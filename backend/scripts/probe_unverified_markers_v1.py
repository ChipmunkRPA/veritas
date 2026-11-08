"""Counterfactual source-label policy probe; authored pixels, no learned inference.

This is a correctness reproduction, not an AI-detector accuracy benchmark.
Before/after phases reuse the exact same input bytes in separate isolated DBs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.engine import pipelines
from app.engine.customer_assessment import customer_assessment
from app.provenance import watermarks
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base, MediaObject
from app.storage.objects import ObjectStore
from app.third_party.imwatermark.max_dct import EmbedMaxDct


def digest(data):
    return hashlib.sha256(data).hexdigest()


def png(pixels, **fields):
    output = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    for key, value in fields.items():
        metadata.add_text(key, value)
    Image.fromarray(pixels).save(output, 'PNG', pnginfo=metadata)
    return output.getvalue()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[:512, :512]
    pixels = np.stack((60 + 160*x/511, 60 + 160*y/511, 130 + 45*np.sin(x/19)), axis=-1).astype(np.uint8)
    message = watermarks.BY_KEY['sdxl']
    encoder = EmbedMaxDct(watermarks=list(message.bits), wmLen=len(message.bits))
    marked = encoder.encode(pixels[:, :, ::-1].copy())[:, :, ::-1].copy()
    reading = next(r for r in watermarks.scan(marked) if r.message.key == message.key)
    assert reading.present
    samples = {
        'plain': png(pixels),
        'unsigned_label': png(pixels, AIGC=json.dumps({'Label': '1', 'ContentProducer': 'OpenAI'})),
        'generation_parameters': png(pixels, parameters='A bowl on a desk\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512'),
        'public_watermark': png(marked),
    }
    cases = []
    for name, data in samples.items():
        (folder / (name + '.png')).write_bytes(data)
        cases.append({'id': name, 'filename': name + '.png', 'sha256': digest(data), 'bytes': len(data)})
    plan = {'scope': 'Authored non-neural pixels; copyable metadata/public-message watermark only.',
            'baseline_commit': 'e87dd70a639a4bcfa79ee29770b3a35b9b5fb6c0', 'cases': cases,
            'watermark_bits_matched': reading.equal_bits, 'watermark_bits': reading.n_bits,
            'model_inference': 'forbidden', 'customer_database': 'not used',
            'accuracy_claim': 'none; policy correctness probe, not model evaluation'}
    (folder / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')


def run(folder, phase):
    plan = json.loads((folder / 'plan.json').read_text())
    target = folder / phase
    target.mkdir(exist_ok=False)
    settings.data_dir = target
    settings.min_free_bytes = 0
    db_engine = create_engine('sqlite:///' + str(target / 'isolated.sqlite'))
    Base.metadata.create_all(db_engine)
    store = ObjectStore(target / 'objects')

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
            raise RuntimeError('This policy probe forbids learned inference')

    outcomes = []
    for case in plan['cases']:
        data = (folder / case['filename']).read_bytes()
        assert digest(data) == case['sha256'] and len(data) == case['bytes']
        aid = str(uuid.uuid4())
        with sessions() as db:
            row = Analysis(id=aid, kind='image', filename=case['filename'], content_type='image/png',
                           size_bytes=len(data), sha256=case['sha256'], status='queued')
            db.add(row)
            db.flush()
            artefact = register_artefact(db, store, data, media_type='image/png', role='original',
                                        created_by='authored-marker-policy-probe', analysis_id=aid)
            media = MediaObject(id=str(uuid.uuid4()), sha256=case['sha256'], size_bytes=len(data), mime_sniffed='image/png',
                                original_filename=case['filename'], storage_artefact_id=artefact.id)
            db.add(media)
            row.media_id = media.id
            db.commit()
        models = NoInference()
        outcome = pipelines.run_analysis(aid, profile='quick', sessions=sessions, store=store, models=models)
        assert outcome.status == 'done', (outcome.status, outcome.error)
        with sessions() as db:
            row = db.get(Analysis, aid)
            result = json.loads(store.path_for(row.result_hash).read_bytes())
        customer = customer_assessment({**result, 'status': 'done'})
        record = {'case': case['id'], 'source_sha256': case['sha256'], 'decision': result['decision'],
                  'customer_assessment': customer, 'gate': outcome.outputs['provenance_gate']['action'],
                  'model_lease_attempts': models.calls, 'model_forwards': 0,
                  'signals': [{k:s.get(k) for k in ('detector_id','provenance_override','raw_score','raw_kind','effective_weight')}
                              for s in result.get('signals',[])],
                  'findings': result.get('findings',[]), 'result_hash': row.result_hash}
        outcomes.append(record)
        print(json.dumps({k:record[k] for k in ('case','decision','gate','model_lease_attempts')}), flush=True)
    (target / 'outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    return outcomes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'before', 'after'))
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.folder)
    else:
        run(args.folder, args.phase)
