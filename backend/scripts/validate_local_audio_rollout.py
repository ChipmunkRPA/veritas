"""Paid API -> real inference -> evidence/export/PDF integration, isolated database.

Uses consumed research fixtures, not a fresh accuracy evaluation. Only the Stripe
HTTP transport is doubled; the payment gate, signed webhook, queue, models,
calibration, evidence and report pipeline are production code. No real charge.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'backend'), str(Path(__file__).parent)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['legacy', 'candidate'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Run only named engineering cases')
    parser.add_argument('--video-control', type=Path, help='Use a speaking video as the video case in candidate mode')
    parser.add_argument('--video-evidence-only', action='store_true', help='Verify a video evidence fixture without voice-specific assertions')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    if args.video_control is not None and args.mode != 'candidate':
        parser.error('--video-control requires --mode candidate')
    if args.video_evidence_only and (args.video_control is None or args.cases != ['video']):
        parser.error('--video-evidence-only requires --video-control and --cases video')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    data = folder/'data'
    assert not (data/'veritas.db').exists(), 'Use a fresh output directory; completed runs are immutable.'
    os.environ.update(VERITAS_DATA_DIR=str(data), VERITAS_DATABASE_URL=f'sqlite:///{data}/veritas.db',
        VERITAS_MODELS_DIR=str(ROOT/'models'), VERITAS_AUDIO_DETECTOR_MODE=args.mode,
        VERITAS_BILLING_MODE='test', VERITAS_STRIPE_SECRET_KEY='sk_test_isolated_validation',
        VERITAS_STRIPE_WEBHOOK_SECRET='whsec_isolated_validation', VERITAS_API_KEY=secrets.token_hex(32),
        VERITAS_API_KEY_PEPPER=secrets.token_hex(32), VERITAS_SIGNING_KEY=secrets.token_hex(32),
        VERITAS_SINGLE_USER='false', VERITAS_DEVICE=args.device, VERITAS_OFFLINE='true',
        VERITAS_JOB_TIMEOUT_S='900', VERITAS_MIN_FREE_BYTES='0', VERITAS_RUN_WORKER='true')
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import auth
    from app.billing import service
    from app.storage.db import Analysis, Job, get_session
    from app.storage.objects import default_store
    from app.detectors.audio.calibration import SHA256
    from verify_multimedia_exports import run as exports, scan
    remote = {}

    def provider(method, path, *, data=None, params=None, idempotency_key=None):
        if method == 'POST':
            sid = 'cs_test_' + data['client_reference_id'].replace('-', '')
            remote.setdefault(sid, {'id': sid, 'object': 'checkout.session', 'mode': 'payment', 'livemode': False,
                'client_reference_id': data['client_reference_id'], 'amount_total': 999, 'amount_subtotal': 999,
                'currency': 'usd', 'metadata': {k[9:-1]: v for k, v in data.items() if k.startswith('metadata[')},
                'payment_status': 'unpaid', 'status': 'open', 'payment_intent': 'pi_test_' + sid,
                'url': 'https://checkout.stripe.com/c/pay/' + sid,
                'line_items': {'has_more': False, 'data': [{'quantity': 1, 'amount_total': 999, 'amount_subtotal': 999,
                    'currency': 'usd', 'price': {'unit_amount': 999, 'currency': 'usd', 'type': 'one_time'}}]}})
            return copy.deepcopy(remote[sid])
        return copy.deepcopy(remote[path.rsplit('/', 1)[-1]])
    service._stripe = provider
    research = ROOT/'backend/data/independent-evaluation/audio-v3'
    rows = json.loads((research/'predictions.json').read_text())
    chosen = [next(r for r in rows if r['label'] == label and r['condition'] == condition)
              for label in (0, 1) for condition in ('clean', 'noise_0')]
    # Choose the exact recorded condition name without relying on case/spelling.
    cases = [{'name': f"{'ai' if r['label'] else 'real'}-{r['condition']}", 'kind': 'audio',
              'path': research/r['path'], 'expected': r} for r in chosen]
    if args.mode == 'legacy':
        cases = [c for c in cases if c['expected']['condition'] == 'clean']
    else:
        import numpy as np
        import soundfile as sf
        source, rate = sf.read(cases[2]['path'], dtype='float32')
        sf.write(folder/'long.wav', np.tile(source, 3), rate, subtype='FLOAT')
        sf.write(folder/'silence.wav', np.zeros(64000, np.float32), 16000, subtype='FLOAT')
        video = args.video_control.resolve() if args.video_control else folder/'speech-video.mov'
        if not args.video_control:
            # Static slate + real test speech; a plumbing fixture, not a video accuracy test.
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'color=c=0x254d42:s=640x360:r=25:d=4', '-i', str(cases[2]['path']),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-shortest',
                            str(video)], check=True)
        cases += [{'name': 'long', 'kind': 'audio', 'path': folder/'long.wav'},
                  {'name': 'silence', 'kind': 'audio', 'path': folder/'silence.wav'},
                  {'name': 'video', 'kind': 'video', 'path': video}]
    if args.cases:
        assert set(args.cases) <= {c['name'] for c in cases}
        cases = [c for c in cases if c['name'] in args.cases]
    results = []
    with TestClient(app) as client:
        with get_session() as db:
            owner = auth.new_principal('rollout-validation', 'examiner')
            db.add(owner); db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers['X-API-Key'] = key
        public_key = client.get('/api/v2/instance/public-key').json()['public_key_pem'].encode()
        for case in cases:
            started = time.monotonic()
            with get_session() as db:
                before = (db.query(Analysis).count(), db.query(Job).count())
            response = client.post('/api/v2/billing/checkout', headers={'Idempotency-Key': str(uuid.uuid4())},
                data={'kind': case['kind'], 'profile': 'full'},
                files={'file': (case['path'].name, case['path'].read_bytes(), 'application/octet-stream')})
            assert response.status_code == 201, response.text
            oid = response.json()['order_id']
            with get_session() as db:
                assert before == (db.query(Analysis).count(), db.query(Job).count()), 'Analysis ran before payment'
            sid = next(s for s, r in remote.items() if r['client_reference_id'] == oid)
            remote[sid].update(payment_status='paid', status='complete')
            event = {'id': 'evt_test_' + oid.replace('-', ''), 'type': 'checkout.session.completed', 'livemode': False,
                     'data': {'object': {'id': sid}}}
            payload = json.dumps(event, separators=(',', ':')).encode(); stamp = str(int(time.time()))
            sig = hmac.new(b'whsec_isolated_validation', stamp.encode()+b'.'+payload, hashlib.sha256).hexdigest()
            paid = client.post('/api/v2/billing/webhook', content=payload, headers={'Stripe-Signature': f't={stamp},v1={sig}'})
            assert paid.status_code == 200, paid.text
            order = client.get(f'/api/v2/billing/orders/{oid}').json(); aid = order['analysis_id']
            print(json.dumps({'case': case['name'], 'analysis_id': aid, 'stage': 'paid_and_queued'}), flush=True)
            until = time.monotonic()+950
            while True:
                with get_session() as db:
                    row = db.get(Analysis, aid)
                    status, error, result_hash = row.status, row.error, row.result_hash
                if status in ('done', 'failed', 'cancelled'):
                    break
                assert time.monotonic() < until, 'Analysis timeout'
                time.sleep(1)
            assert status == 'done', (case['name'], status, error)
            result = json.loads(default_store().read_bytes(result_hash))
            signals = result['signals']
            primary = next((s for s in signals if s['detector_id'] == 'aud.pella_v2'), None)
            screen = primary['details'].get('audio_screening') if primary else None
            if args.video_evidence_only:
                assert result['kind'] == 'video'
            elif args.mode == 'candidate':
                assert not any(s['detector_id'] == 'aud.gustking_xlsr' for s in signals)
                if case['name'] not in ('silence',):
                    assert screen, 'Real candidate did not run'
                if case['kind'] == 'audio':
                    assert result['decision']['fuser_sha256'] == SHA256
                else:
                    assert primary['effective_weight'] == 0
                    assert result['decision']['fuser_sha256'] != SHA256
                    assert 'Gustking' not in result['decision']['reason']
                if expected := case.get('expected'):
                    assert abs(screen['probability_ai']-expected['new_p']) < 1e-5, (screen, expected['new_p'])
                    assert screen['decision'] == expected['new_guarded'], (screen, expected['new_guarded'])
                if case['name'] == 'long':
                    assert screen['probability_ai'] is None and screen['decision'] == 'inconclusive'
                if case['name'] == 'silence':
                    assert result['decision']['probability_ai_calibrated'] is None
            else:
                assert any(s['detector_id'] == 'aud.gustking_xlsr' for s in signals)
                assert result['decision']['fuser_sha256'] != SHA256
                assert screen is None
            detail = client.get(f'/api/v2/analyses/{aid}'); detail.raise_for_status(); scan(detail.text)
            case_folder = folder/case['name']; case_folder.mkdir()
            (case_folder/'api.json').write_bytes(detail.content)
            # Complete export, evidence asset/range, PDF, signature and package correspondence.
            check = exports(client, aid, case_folder, default_store(), public_key)
            if args.mode == 'candidate' and case['kind'] == 'video' and not args.video_evidence_only:
                soundtrack = detail.json()['customer_assessment']['soundtrack']
                assert soundtrack == json.loads((case_folder/'export.json').read_bytes())['customer_assessment']['soundtrack']
                assert soundtrack['direction'] == screen['decision']
                import pymupdf

                with pymupdf.open(case_folder/'report.pdf') as report:
                    cover = ' '.join(report[0].get_text().split())
                    assert soundtrack['headline'] in cover
                    if soundtrack['confidence_pct'] is not None:
                        assert f"{soundtrack['confidence_pct']:g}%" in cover
                check['soundtrack'] = soundtrack
            check.update(case=case['name'], seconds=round(time.monotonic()-started, 2),
                         decision=result['decision']['code'], probability_ai=result['decision']['probability_ai_calibrated'],
                         payment_before_analysis=True, candidate_calibration=screen)
            results.append(check)
            (folder/'results.json').write_text(json.dumps({'mode': args.mode, 'purpose': __doc__, 'results': results}, indent=2))
            print(json.dumps({'case': case['name'], 'status': 'verified', 'seconds': check['seconds'],
                              'decision': check['decision'], 'p': check['probability_ai']}), flush=True)
    print('LOCAL AUDIO ROLLOUT VALIDATION PASSED', args.mode, flush=True)


if __name__ == '__main__':
    main()
