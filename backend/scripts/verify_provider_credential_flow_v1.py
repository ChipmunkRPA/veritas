"""Exercise actual provider bytes in isolated local analyses; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.reports import engine, snapshot
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base, MediaObject
from app.storage.objects import ObjectStore
from app.version import VERSION


def verify(study, folder, pdf):
    folder.mkdir(parents=True, exist_ok=False)
    settings.data_dir = folder
    settings.min_free_bytes = 0
    db_engine = create_engine('sqlite:///' + str(folder / 'isolated.sqlite'))
    Base.metadata.create_all(db_engine)
    store = ObjectStore(folder / 'objects')

    @contextmanager
    def sessions():
        with Session(db_engine, expire_on_commit=False) as db:
            yield db

    class NoInference:
        calls = 0
        default_device = 'cpu'

        def activation_bytes_for_items(self, counts):
            return 0

        def lease(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError('Qualification forbids model inference')

    plan = json.loads((study / 'plan.json').read_text())
    outcomes = []
    for case in [c for c in plan['cases'] if c['id'].startswith('provider_source_')]:
        data = (study / case['path']).read_bytes()
        digest, aid = hashlib.sha256(data).hexdigest(), str(uuid.uuid4())
        assert digest == case['sha256']
        with sessions() as db:
            row = Analysis(id=aid, kind='image', filename='public-provider-qualification.png', content_type='image/png',
                           size_bytes=len(data), sha256=digest, status='queued')
            db.add(row)
            db.flush()
            artefact = register_artefact(db, store, data, media_type='image/png', role='original',
                                        created_by='provider-qualification', analysis_id=aid)
            media = MediaObject(id=str(uuid.uuid4()), sha256=digest, size_bytes=len(data), mime_sniffed='image/png',
                                original_filename=row.filename, storage_artefact_id=artefact.id)
            db.add(media)
            row.media_id = media.id
            db.commit()
        models = NoInference()
        run = pipelines.run_analysis(aid, profile='quick', sessions=sessions, store=store, models=models)
        assert run.status == 'done', (run.status, run.error)
        assert run.outputs['provenance_gate']['action'] == 'continue_detection'
        assert models.calls > 0, 'Invalid credentials must not bypass applicable model checks'
        with sessions() as db:
            row = db.get(Analysis, aid)
            result = json.loads(store.path_for(row.result_hash).read_bytes())
            assessment = customer_assessment({**result, 'status': 'done'})
            assert assessment['confidence_pct'] is None
            assert assessment['direction'] in ('inconclusive', 'unavailable')
            assert result['decision'].get('provenance_record') is None
            finding = next(f for f in result['findings'] if 'prov.c2pa:global' in f['evidence_ids'])
            assert 'hash matches the current file bytes' in finding['statement']
            assert 'certificate is expired' in finding['statement']
            assert 'changed after signing' not in finding['statement']
            if not outcomes:
                frozen = snapshot.build(db, row, result, report_id=str(uuid.uuid4()), revision=1,
                    profile='standard', redacted=True, supersedes_report_id=None, revision_reason=None,
                    requested_at=datetime.now(UTC), requested_by='research-qualification',
                    template_set=engine.TEMPLATE_SET, veritas_version=VERSION, store=store)
                (folder / 'report-snapshot.json').write_bytes(snapshot.dump(frozen))
                rendered = engine.render_pdf(frozen)
                pdf.parent.mkdir(parents=True, exist_ok=True)
                pdf.write_bytes(rendered.pdf)
                text = ' '.join(engine.pdf_text(rendered.pdf).split())
                assert 'changed after signing' not in text
                assert 'signing certificate is expired' in text
                assert 'hash matches the current file bytes' in text
                assert 'Credentials were not accepted.' in text
                assert 'Content Credentials were not checked' not in text
                assert not rendered.refused
                assert not rendered.lint.get('banned_terms') and not rendered.lint.get('internal_terms'), rendered.lint
                pdf_result = {'pages': rendered.page_count, 'lint': rendered.lint,
                              'pdf_sha256': hashlib.sha256(rendered.pdf).hexdigest()}
            outcomes.append({'case': case['id'], 'analysis_id': aid, 'result_hash': row.result_hash,
                             'model_lease_attempts': models.calls, 'model_forwards': 0,
                             'assessment': assessment, 'credential_finding': finding})
    assert len(outcomes) == 2
    result = {'cases': outcomes, 'pdf': pdf_result, 'customer_database_used': False, 'provider_requests': 0}
    (folder / 'flow-review.json').write_text(json.dumps(result, indent=2) + '\n')
    return {'flows': len(outcomes), 'pdf': pdf_result, 'customer_database_used': False,
            'model_forwards': 0, 'provider_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('pdf', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.study, args.folder, args.pdf), indent=2))
