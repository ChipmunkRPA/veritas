"""Publish audited natural-media aggregates; exclude raw sources and case pairings."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts import run_public_visual_natural_v2 as study

m = study.m
FREEZE_COMMIT = '257e04aecfc70e980f474fbd05bd3d951ab673e7'


def publish(trial, output):
    lock, sources = study.check(trial)
    public_freeze = m.read(study.ROOT / 'docs/evaluation-panels/public-visual-natural-v2-freeze.json')
    m.require(public_freeze['prospective_freeze_sha256'] == m.digest(trial / 'freeze.json'),
              'The published original source-panel freeze is required')
    for path, expected in lock['code_sha256'].items():
        source = subprocess.check_output(['git', 'show', f'{FREEZE_COMMIT}:{path}'], cwd=study.ROOT)
        m.require(hashlib.sha256(source).hexdigest() == expected, 'Published prospective source identity differs')
    result, audit = m.read(trial / 'result.json'), m.read(trial / 'audit/audit.json')
    m.require(result['case_count'] == audit['cases'] == 640 and result['generator_forwards'] == 64
        and result['detector_forwards'] == 645 and audit['source_assets_redecoded'] == 40
        and audit['source_sampled_frames_replayed'] == 64 and audit['neural_forwards'] == 0
        and audit['presence_pixels_replayed'] == 41943040 and audit['binary_bit_predictions_replayed'] == 1342177280
        and audit['h264_clips_reencoded_and_redecoded'] == 24
        and audit['original_and_derived_source_clocks_verified'] is audit['lossless_f32_replay_verified'] is True,
        'Complete original native and model-free audit required')
    m.require(all(result[k] is False for k in ('provider_authenticated', 'used_in_confidence', 'early_stop_eligible', 'serving_admitted')),
              'Research observations cannot be promoted by publication')
    profiles = {}
    for name in ('acquisition-profile', 'profile', 'audit-profile'):
        profile = m.read(trial / name / 'profile.json')
        m.require(profile['child_exit_code'] == 0 and profile['guard'] is None, 'A failed or guarded phase cannot be omitted')
        profiles[name] = {k: v for k, v in profile.items() if k != 'arguments'}
        profiles[name]['arguments_paths_withheld'] = True
    out = {'schema': 'chipmunk.public-natural-visual-watermark-evaluation/2',
        'prospective_freeze_commit': FREEZE_COMMIT, 'prospective_freeze_sha256': m.digest(trial / 'freeze.json'),
        'native': result, 'source_audit': audit, 'resource_profiles': profiles,
        'source_panel': {'images': 32, 'clips': 8,
            'photo_creators': len({r['creator'] for r in sources if r['kind'] == 'image'}),
            'video_creators': len({r['creator'] for r in sources if r['kind'] == 'video'}),
            'licenses': dict(Counter(r['license'] for r in sources)), 'photo_strata': study.sources.QUOTAS,
            'raw_media_published': False, 'case_pairings_published': False,
            'raw_predictions_published': False, 'metadata_authenticates_capture': False,
            'scope': 'No-added-mark/known added-mark controls; photographic development eligibility review, not AI-authorship truth or production-provider compatibility.'},
        'serving_changed': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'customer_inputs': 0, 'external_provider_verification_requests': 0, 'outside_human_reviewers': 0,
        'canonical_retained_records_rewritten': False,
        'publication_redactions': 'Resource-profile argument paths withheld. Native statistics, policy, audit facts and sampling-error counts unchanged.',
        'limitations': ['Public watermark presence cannot authenticate a provider or AI authorship.',
            'Frames/variants share sources; no independent-population accuracy or confidence is established.',
            'An outside-interval compressed response can be codec propagation or a false-family response; not proof of an AI edit at that timestamp.',
            f"The audit recorded {profiles['audit-profile']['sample_errors']} process-sampling error(s); sampled memory is a lower bound, not an exact peak. The caught exception type was not separately recorded."],
        'exporter_sha256': m.digest(Path(__file__))}
    m.write(output, out)
    return {'published_cases': 640, 'native_forwards': 709, 'audit_neural_forwards': 0,
            'audit_process_sampling_errors': profiles['audit-profile']['sample_errors'], 'serving_admitted': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.trial, a.output), indent=2))
