"""Verify the local known-composite demo against its API, frozen report and source pixels.

Run from backend: .venv/bin/python scripts/verify_image_evidence_demo.py REPORT_ID
This is an engineering fixture check, not a detector-accuracy benchmark.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np
import pymupdf
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evidence.image_packets import packet_hash  # noqa: E402
from app.reports.signing import load_public_key, verify  # noqa: E402
from app.reports.snapshot import load  # noqa: E402
from app.storage.db import Analysis, ReportRow, get_session  # noqa: E402
from app.storage.objects import default_store  # noqa: E402


def main(report_id: str) -> None:
    output = Path('data/image-evidence-demo')
    store = default_store()
    identities = re.compile(r'mesorch|segformer|convnext|depth.anything|dinov2', re.I)
    checks = []
    with httpx.Client(base_url='http://127.0.0.1:8160', timeout=60) as client, get_session() as db:
        for item in json.loads((output / 'analyses.json').read_text()):
            aid = item['analysis_id']
            analysis = db.get(Analysis, aid)
            response = client.get(f'/api/v2/analyses/{aid}/evidence-packets')
            response.raise_for_status()
            evidence = response.json()
            exported = client.get(f'/api/analyses/{aid}/export.json')
            exported.raise_for_status()
            assert evidence == exported.json()['image_evidence']
            assert packet_hash(evidence) == evidence['sha256']
            assert not identities.search(exported.text), 'Internal model identity in public JSON'
            source = ImageOps.exif_transpose(Image.open(io.BytesIO(store.read_bytes(analysis.sha256)))).convert('RGB')
            count, metric = 0, {}
            for packet in evidence['packets']:
                for role, asset in packet['artifacts'].items():
                    response = client.get(asset['url'])
                    response.raise_for_status()
                    assert hashlib.sha256(response.content).hexdigest() == asset['sha256']
                    image = Image.open(io.BytesIO(response.content))
                    assert image.size == (asset['width'], asset['height'])
                    if role == 'crop':
                        crop = packet['crop_region']
                        expected = source.crop((crop['x'], crop['y'], crop['x'] + crop['w'], crop['y'] + crop['h']))
                        np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))
                    if role == 'mask':
                        predicted = np.asarray(image.convert('L')) > 127
                        truth = np.asarray(Image.open(output / 'reference-mask.png')) > 127
                        assert predicted.shape == truth.shape
                        metric = {'mask_iou': float((predicted & truth).sum() / (predicted | truth).sum()),
                                  'scope': 'Known simple compositing demo only, not generative editing accuracy'}
                    count += 1
            (output / (Path(item['name']).stem + '.json')).write_bytes(exported.content)
            checks.append({'analysis': aid, 'name': item['name'], 'evidence_hash': evidence['sha256'],
                           'localised_regions': sum(p['evidence_type'] == 'localisation' for p in evidence['packets']),
                           'exact_pngs_checked': count, 'exact_crops': True, 'json_api_equal': True, **metric})

        report = db.get(ReportRow, report_id)
        response = client.get(f'/api/v2/reports/{report_id}/file')
        response.raise_for_status()
        pdf_bytes = response.content
        assert hashlib.sha256(pdf_bytes).hexdigest() == report.sha256
        public = client.get('/api/v2/instance/public-key')
        public.raise_for_status()
        assert verify(load_public_key(public.json()['public_key_pem'].encode()), pdf_bytes, report.signature)
        frozen = load(store.read_bytes(report.snapshot_hash))
        assert frozen.image_evidence['sha256'] == checks[0]['evidence_hash']
        pdf = pymupdf.open(stream=pdf_bytes, filetype='pdf')
        text = '\n'.join(page.get_text() for page in pdf)
        assert not identities.search(text), 'Internal model identity in PDF'
        manifests = [json.loads(pdf.embfile_get(name)) for name in pdf.embfile_names() if name.endswith('.json')]
        manifest = next(m for m in manifests if 'image_evidence' in m)
        assert manifest['image_evidence']['sha256'] == checks[0]['evidence_hash']
        assert {a['sha256'] for a in manifest['image_evidence']['artifacts']} == {
            a['sha256'] for p in frozen.image_evidence['packets'] for a in p['artifacts'].values()}
        (output / 'chipmunk-verify-image-evidence.pdf').write_bytes(pdf_bytes)
        (output / 'pdf-report-record.json').write_text(json.dumps(client.get(f'/api/v2/reports/{report_id}').json(), indent=2))
        (output / 'report-text.txt').write_text(text)
        pages = [i + 1 for i, p in enumerate(pdf) if any(term in p.get_text() for term in (
            'Local editing candidate', 'Compression difference', 'Noise variation'))]
        baseline = Path('/private/tmp/chipmunk-evidence-baseline.json')
        if baseline.is_file():
            for previous in json.loads(baseline.read_text())['analyses']:
                original = db.get(Analysis, previous['id'])
                assert original.result_hash == previous['result_hash']
                assert hashlib.sha256(original.result_json.encode()).hexdigest() == previous['result_json_sha256']
        assert client.get('/api/v2/reports/VR-2026-000001/file').status_code == 200
        assert client.get('/api/v2/reports/VR-2026-000002/file').status_code == 409
        summary = {'checks': checks, 'pdf': {'id': report_id, 'pages': len(pdf), 'evidence_pages': pages,
                   'signature_verified': True, 'packet_hash_equal': True, 'all_plate_hashes_equal': True},
                   'historical_results_unchanged': True if baseline.is_file() else 'not checked',
                   'original_safe_report_downloadable': True, 'model_name_scan': 'passed'}
        (output / 'verification.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Pass the completed local demo report ID')
    main(sys.argv[1])
