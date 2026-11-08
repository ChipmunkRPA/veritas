"""Export rejected development, retained strict failure and complete diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from scripts import diagnose_image_patch_head_v14 as diagnostics
from scripts import run_image_patch_head_v14 as study

FREEZE_COMMIT = '479b7cca66dec2d8313f7cc977d0b212b6179477'
DIAGNOSTIC_FREEZE_COMMIT = '498023d79f83570702694ce5348cd63c7d5ffede'


def publish(trial, output):
    study.check(trial)
    public = study.read(study.ROOT / 'docs/evaluation-panels/image-patch-head-v14-freeze.json')
    study.method.require(study.digest(trial / 'freeze.json') == public['private_freeze_sha256'],
                         'Published prospective source-panel freeze required')
    for path, expected in public['code_sha256'].items():
        frozen = subprocess.check_output(['git', 'show', f'{FREEZE_COMMIT}:{path}'], cwd=study.ROOT)
        study.method.require(hashlib.sha256(frozen).hexdigest() == expected,
                             'Prospectively published method differs')
    diagnostics.check(trial)
    diagnostic_freeze = study.read(study.ROOT / 'docs/evaluation-panels/image-patch-head-v14-diagnostic-freeze.json')
    study.method.require(diagnostic_freeze == study.read(trial / 'diagnostic-freeze.json'),
                         'Public diagnostic plan differs from original private plan')
    for path, expected in diagnostic_freeze['source_sha256'].items():
        frozen = subprocess.check_output(['git', 'show', f'{DIAGNOSTIC_FREEZE_COMMIT}:{path}'], cwd=study.ROOT)
        study.method.require(hashlib.sha256(frozen).hexdigest() == expected, 'Published diagnostic source differs')
    result, audit = study.read(trial / 'result.json'), study.read(trial / 'diagnostic-replay.json')
    study.method.require(result['cases'] == audit['cases'] == 1344
        and result['backbone_forwards'] == 1363 and audit['backbone_forwards'] == 1362
        and result['head_training']['steps'] == 3150 and result['head_inference_cases'] == 504
        and audit['fit_sample_rows_replayed'] == 840 and audit['sampled_patch_vectors_replayed'] == 53760
        and audit['native_response_cells_replayed'] == 516096
        and audit['native_features_and_maps_exact'] is audit['original_genuine_point_and_aggregates_exact'] is True
        and audit['source_truth_samples_and_projection_exact'] is True
        and audit['original_strict_audit_passed'] is False,
        'Complete original scoring and separate full native diagnostic replay required')
    study.method.require(all(r['serving_admitted'] is r['used_in_confidence'] is False for r in (result, audit)),
                         'Research result cannot be promoted by an export')
    profiles = {}
    for name in ('profile', 'diagnostic-profile'):
        raw = study.read(trial / name / 'profile.json')
        study.method.require(raw['child_exit_code'] == 0 and raw['guard'] is None,
                             'Failed or guarded phases cannot be omitted')
        profiles[name] = {k: v for k, v in raw.items() if k != 'arguments'}
        profiles[name]['arguments_paths_withheld'] = True
    failed = study.read(trial / 'audit-profile/profile.json')
    study.method.require(failed['child_exit_code'] == 1 and failed['guard'] is None,
                         'Original failed strict audit must remain visible')
    profiles['failed-strict-audit'] = {k: v for k, v in failed.items() if k != 'arguments'}
    profiles['failed-strict-audit']['arguments_paths_withheld'] = True
    contract = study.read(study.ROOT / 'backend/data/independent-evaluation/image-patch-head-v14-contract-profile/profile.json')
    study.method.require(contract['child_exit_code'] == 0 and contract['guard'] is None,
                         'Complete authored contract resource receipt required')
    profiles['authored-contract'] = {k: v for k, v in contract.items() if k != 'arguments'}
    profiles['authored-contract']['arguments_paths_withheld'] = True
    output_record = {
        'schema': 'chipmunk.image-supervised-patch-head-public-result/14',
        'prospective_freeze_commit': FREEZE_COMMIT, 'prospective_freeze_sha256': public['private_freeze_sha256'],
        'diagnostic_freeze_commit': DIAGNOSTIC_FREEZE_COMMIT,
        'diagnostic_freeze_sha256': study.digest(trial / 'diagnostic-freeze.json'),
        'native_development': result, 'complete_native_diagnostic_replay': audit, 'resource_profiles': profiles,
        'retained_strict_audit': {'completed': False, 'passed': False,
            'stopped_check': 'Independent full-image mean exceeds frozen arithmetic bound',
            'original_bound': 1e-5, 'bound_changed': False,
            'failed_log_sha256': diagnostic_freeze['failed_audit_log_sha256'],
            'failed_profile_sha256': diagnostic_freeze['failed_audit_profile_sha256'],
            'partial_backbone_forward_count': 'not recorded; not inferred from duration'},
        'source_groups': public['source_groups'], 'case_roles': public['cases'], 'assets': public['assets'],
        'authored_contract_backbone_forwards': 9,
        'completed_project_scoring_and_diagnostic_backbone_forwards': 2725,
        'completed_phase_count_excludes_unrecorded_partial_strict_audit': True,
        'outside_human_reviewers': 0, 'customer_inputs': 0, 'external_provider_requests': 0,
        'serving_changed': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'raw_sources_published': False, 'source_pairings_published': False,
        'fit_vectors_or_weights_published': False, 'canonical_records_rewritten': False,
        'limitations': [
            'Consumed-source development, not fresh accuracy, independent capture truth or unseen-generator validation.',
            'Ranking sigmoid and genuine-only comparison point are not calibrated AI-authorship confidence.',
            'Baseline ordinary-control native inference is unassessed here and would be required before fresh confirmation.',
            'A native/source/NumPy replay is not independent optimizer refitting or outside human review.',
            'The original strict mean audit failed and remains failed; diagnostic completion does not waive it.',
            'Accurate-mean response sensitivity retains the original trained head and point; it is not a corrected-model evaluation.',
            'Keep every original quality-rule failure and subgroup loss; no retuned point or selected subgroup.',
            'Process memory is a sampled lower bound. All original sampling-error counts remain visible.',
        ],
        'exporter_sha256': study.digest(Path(__file__)),
    }
    study.write(output, output_record)
    return {'published_cases': 1344, 'completed_scoring_and_diagnostic_backbone_forwards': 2725,
            'original_strict_audit_passed': False,
            'serving_changed': False, 'quality_rules_pass': result['all']['quality_rules_pass']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(publish(args.trial, args.output))
