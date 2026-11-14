# SPDX-License-Identifier: Apache-2.0
"""Replay the default-off four-second H.264 face review through the real local worker."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from scripts.video_short_return_encoded_app import (
    ALLOW,
    PaymentStub,
    body,
    checkout,
    configure,
    fetch,
    read_result,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / 'docs/evaluation-panels/video-brief-face-120-carrier-2026-11-13.json'
CASES = ('two_swaps', 'tandem_two_swaps', 'original', 'cross_recording_cuts')


def run(bank, output):
    ledger = {row['delivery_sha256']: row for row in json.loads(LEDGER.read_text())['held']['avc']}
    media = {row['file']: row for row in json.loads((bank / 'media.json').read_text())['rows']}
    configure(output, ROOT)

    from scripts.video_brief_face_avc_confirmation import BASE_FLAGS

    os.environ.update(BASE_FLAGS, VERITAS_VIDEO_SCENE_RETURN_REVIEW='true', VERITAS_VIDEO_WHOLE_FACE_REVIEW='false',
                      VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW='false')

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.video.adapters import face_appearance
    from app.engine.registry import get_registry
    from app.evidence import face_return_coverage, face_short_proposal_binding, scene_return_binding
    from app.main import app
    from app.storage.db import get_session
    from scripts.video_face_unrouted_short_app import verified_surfaces

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    assert set(registry.enabled('video', 'full')) == ALLOW
    stub = PaymentStub()
    service._stripe = stub
    timings = []
    propose = face_appearance._propose_short

    def measured_proposal(*args, **kwargs):
        started = time.perf_counter()
        try:
            return propose(*args, **kwargs)
        finally:
            timings.append(time.perf_counter() - started)

    face_appearance._propose_short = measured_proposal
    baseline = {}
    results = {}
    with TestClient(app, client=('127.0.0.1', 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal('local-avc-trial', 'examiner')
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers['X-API-Key'] = key
        for active in (False, True):
            settings.video_face_short_avc_review = active
            for case in CASES:
                filename = f'g1-{case}.mp4'
                reference = media[filename]
                expected = ledger[reference['sha256']]
                source = bank / 'media' / filename
                aid = checkout(client, stub, source, filename='trial-four-second-h264.mp4')
                decision, entry = read_result(aid, reference['sha256'])
                evidence = body(client.get(f'/api/v2/analyses/{aid}/evidence-packets'))
                scene = evidence['coverage'][scene_return_binding.COVERAGE]
                packets = [p for p in evidence['packets'] if p.get('semantics') == face_short_proposal_binding.AVC_METHOD]
                assert len(packets) == int(active and expected['proposal'] is not None)
                if not active:
                    assert 'face_short_proposal_check' not in entry['details'] and not timings
                    baseline[case] = decision, scene
                    continue
                assert baseline[case] == (decision, scene)
                coverage = evidence['coverage'][face_return_coverage.KEY]
                bounded = coverage['passes'][face_return_coverage.BOUNDED]
                assert coverage['affects_overall_confidence'] is False
                assert bounded['status'] == ('localized' if packets else 'no_distinct_range')
                assert len(timings) == len(results) + 1
                assert timings[-1] <= face_appearance.SHORT_PROPOSAL_BUDGET_S
                result = {'bounded_proposal': bounded, 'scene_return_status': scene['status'],
                          'overall_assessment_equal': True, 'bounded_proposal_seconds': round(timings[-1], 3)}
                if packets:
                    [packet] = packets
                    proof = packet['face_return']
                    assert [proof['start_frame_index'], proof['end_frame_index']] == expected['proposal']['frames']
                    assert sum('face change above' in m['label'] for m in packet['measurements']) == 2
                    assert sum('nearby regions' in m['label'] for m in packet['measurements']) == 2
                    assert evidence['version'] == 'video-evidence-26'
                    representatives = {role: x['frame_index'] for role, x in
                                       zip(('before', 'during', 'after'), expected['proposal']['anchors'], strict=True)}
                    verified = verified_surfaces(client, aid, source, reference, evidence, packet, output / case,
                                                 representatives=representatives, initial_coverage=None)
                    result.update(range_frames=expected['proposal']['frames'], source_frames=list(representatives.values()),
                                  exact_source_exhibit_count=len(verified['exact_source_images']),
                                  pdf_api_json_agree=verified['pdf_api_json_agree'],
                                  signed_original_replay=verified['signed_original_replay'],
                                  different_original_rejected=verified['different_original_rejected'])
                else:
                    assert json.loads(fetch(client, f'/api/analyses/{aid}/export.json'))['evidence'] == evidence
                results[case] = result
                print(case, bounded['status'], result['bounded_proposal_seconds'], scene['status'], flush=True)
    receipt = {'schema': 'chipmunk.video-brief-face-avc-app-trial/1', 'cases': results,
               'simulated_paid_runs': 2 * len(CASES), 'unpaid_jobs': 0, 'duplicate_paid_webhooks_checked': True,
               'default_off': True, 'bounded_proposal_budget_seconds': face_appearance.SHORT_PROPOSAL_BUDGET_S,
               'scope': 'Two encodes of one previously evaluated edit and two matching ordinary controls.'}
    (output / 'app-trial-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path, help='Retained four-second source bank with media.json and media/')
    parser.add_argument('output', type=Path, help='New private output directory')
    args = parser.parse_args()
    run(args.bank.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
