"""Publish all four development arms and complete audit without private inputs."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from scripts import run_image_region_point_v15 as study

FREEZE_COMMIT = '730e11e75e839c8f4650ec45d6638e6d48bad9d5'


def publish(trial, output):
    study.check(trial)
    study.previous_diagnostics.check(study.prior_trial())
    public = study.read(study.ROOT / 'docs/evaluation-panels/image-region-point-v15-freeze.json')
    check = study.old.method.require
    check(public['private_freeze_sha256'] == study.digest(trial / 'freeze.json'), 'Original published freeze required')
    for path, expected in public['code_sha256'].items():
        original = subprocess.check_output(['git', 'show', f'{FREEZE_COMMIT}:{path}'], cwd=study.ROOT)
        check(hashlib.sha256(original).hexdigest() == expected, 'Prospectively published source differs')
    native, audit = study.read(trial / 'result.json'), study.read(trial / 'audit.json')
    check(native['cases'] == audit['cases'] == 1344 and native['backbone_forwards'] == 1363
          and audit['backbone_forwards'] == 1362 and native['head_training']['steps'] == 3150
          and native['head_inference_cases'] == 504 and native['outcome_records'] == 2016
          and audit['fit_sample_rows_replayed'] == 840 and audit['sampled_patch_vectors_replayed'] == 53760
          and audit['activation_levels_replayed'] == 1008 and audit['source_mask_metric_sets_replayed'] == 2016
          and audit['native_response_cells_replayed'] == 1032192
          and audit['new_mean_strict_criterion_passed'] is True
          and audit['old_mean_bound_failures_retained'] == 1302 and audit['old_strict_audit_remains_failed'] is True
          and audit['native_features_maps_and_fit_samples_exact'] is audit['source_projection_metrics_exact'] is True
          and audit['independent_activation_and_point_refits_exact'] is True,
          'All original outcomes and complete native/mean/activation/source replay required')
    check(native['primary_candidate'] == 'stable_regions'
          and native['original_canonical_outcomes_unchanged'] and not native['fit_vectors_copied']
          and all(r['serving_admitted'] is r['used_in_confidence'] is False for r in (native, audit)),
          'Publication cannot choose another primary arm or admit a serving model')
    profiles = {}
    locations = {'native': trial / 'profile/profile.json', 'audit': trial / 'audit-profile/profile.json',
                 'authored-contract': study.ROOT / 'backend/data/independent-evaluation/image-region-point-v15-contract-profile/profile.json'}
    for name, path in locations.items():
        raw = study.read(path)
        check(raw['child_exit_code'] == 0 and raw['guard'] is None, 'Failed/guarded phases cannot be omitted')
        profiles[name] = {k: v for k, v in raw.items() if k != 'arguments'}
        profiles[name]['arguments_paths_withheld'] = True
    eligible = native['all']['quality_rules']['stable_regions']['pass']
    record = {'schema': 'chipmunk.image-region-point-public-development/15',
        'prospective_freeze_commit': FREEZE_COMMIT, 'prospective_freeze_sha256': public['private_freeze_sha256'],
        'primary_candidate': 'stable_regions', 'native_development': native, 'complete_native_audit': audit,
        'resource_profiles': profiles, 'source_groups': public['source_groups'], 'case_roles': public['case_roles'],
        'assets': public['assets'], 'primary_quality_rules_pass': eligible,
        'development_continuation_eligible': eligible, 'fresh_confirmation': False,
        'serving_changed': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'old_failure_and_canonical_results_unchanged': True,
        'completed_project_backbone_forwards_including_audit': 2725, 'authored_backbone_forwards': 9,
        'outside_human_reviewers': 0, 'customer_inputs': 0, 'external_provider_requests': 0,
        'raw_sources_published': False, 'source_pairings_published': False,
        'feature_vectors_or_weights_published': False,
        'limits': [
            'All sources are consumed development data, not fresh project confirmation or an unseen-generator test.',
            'Four arms share source cases; they are not four independent datasets or independent corroborating votes.',
            'Region points control in-sample calibration group flags only, not a population false-alarm guarantee.',
            'Sigmoid ranks, component activation levels and empty masks are not AI-confidence or originality verdicts.',
            'Old-head arms retain the original failed mean criterion and are not selectable serving candidates.',
            'Native/source/array replay does not independently refit the optimizer or constitute outside human review.',
            'Current-selector ordinary-control inference, fresh confirmation and full-app/export tests remain before admission.',
            'All source/feature data and learned weights remain private; process memory is a sampled lower bound.'
        ], 'exporter_sha256': study.digest(Path(__file__))}
    study.write(output, record)
    return {'published_cases': 1344, 'arms': 4, 'primary_quality_rules_pass': eligible,
            'new_mean_criterion_passed': True, 'serving_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(publish(args.trial, args.output))
