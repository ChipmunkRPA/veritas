"""Read-only comparison of the saved counterfactual runs and report presentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.engine.customer_assessment import customer_assessment
from app.reports import engine, snapshot
from app.storage.db import Analysis
from app.storage.objects import ObjectStore
from app.version import VERSION


def review(folder, output):
    output.mkdir(exist_ok=False)
    plan = json.loads((folder / 'plan.json').read_text())
    phases = {phase: json.loads((folder / phase / 'outcomes.json').read_text()) for phase in ['before', 'after']}
    rows = []
    for expected, before, after in zip(plan['cases'], phases['before'], phases['after'], strict=True):
        assert expected['id'] == before['case'] == after['case']
        assert expected['sha256'] == before['source_sha256'] == after['source_sha256']
        assert hashlib.sha256((folder / expected['filename']).read_bytes()).hexdigest() == expected['sha256']
        assert before['gate'] == after['gate'] == 'continue_detection'
        assert before['model_forwards'] == after['model_forwards'] == 0
        assert before['model_lease_attempts'] > 0 and after['model_lease_attempts'] > 0
        assert after['decision']['code'] == 'not_assessable'
        assert after['decision']['probability_ai_uncalibrated'] is None
        assert after['decision']['provenance_record'] is None
        assert after['customer_assessment']['confidence_pct'] is None
        rows.append({'case': expected['id'], 'before_code': before['decision']['code'],
                     'before_uncalibrated_score': before['decision']['probability_ai_uncalibrated'],
                     'after_code': after['decision']['code'], 'after_score': None,
                     'after_confidence': None, 'further_checks_attempted': True})
    reports = []
    for phase in ['before', 'after']:
        settings.data_dir = folder / phase
        store = ObjectStore(settings.data_dir / 'objects')
        db_engine = create_engine('sqlite:///' + str(settings.data_dir / 'isolated.sqlite'))
        with Session(db_engine) as db:
            analysis = db.scalar(select(Analysis).where(Analysis.filename == 'public_watermark.png'))
            source = store.path_for(analysis.result_hash).read_bytes()
            result = json.loads(source)
            assessment = customer_assessment({**result, 'status': 'done'})
            assert assessment['confidence_pct'] is None
            assert assessment['direction'] == ('inconclusive' if phase == 'before' else 'unavailable')
            frozen = snapshot.build(db, analysis, result, report_id=str(uuid.uuid4()), revision=1,
                profile='forensic', redacted=True, supersedes_report_id=None, revision_reason=None,
                requested_at=datetime.now(UTC), requested_by='local-marker-policy-review',
                template_set=engine.TEMPLATE_SET, veritas_version=VERSION, store=store)
            frozen_bytes = snapshot.dump(frozen)
            (output / f'{phase}-snapshot.json').write_bytes(frozen_bytes)
            rendered = engine.render_pdf(frozen)
            (output / f'{phase}.pdf').write_bytes(rendered.pdf)
            text = ' '.join(engine.pdf_text(rendered.pdf).split())
            assert assessment['headline'] in text
            assert 'sets the result' not in text and 'floors the' not in text and 'at least 85' not in text
            assert 'public watermark' in text.lower()
            assert not rendered.refused and not rendered.lint.get('banned_terms') and not rendered.lint.get('internal_terms'), rendered.lint
            assert store.path_for(analysis.result_hash).read_bytes() == source
            assert snapshot.dump(frozen) == frozen_bytes
            reports.append({'phase': phase, 'pages': rendered.page_count,
                            'pdf_sha256': hashlib.sha256(rendered.pdf).hexdigest(), 'lint': rendered.lint,
                            'snapshot_sha256': hashlib.sha256(frozen_bytes).hexdigest(), 'canonical_unchanged': True})
    receipt = {'schema': 'chipmunk.unverified-marker-policy/v1', 'cases': rows, 'reports': reports,
               'model_forwards': 0, 'provider_requests': 0, 'customer_database_used': False,
               'scope': 'Counterfactual policy correctness only; no AI-detection accuracy claim.'}
    (output / 'review.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.folder, args.output), indent=2))
