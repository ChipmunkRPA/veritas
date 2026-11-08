"""Investigate the retained strict-mean failure; never converts it into a pass."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_patch_head_v14 as method
from scripts import review_image_patch_head_v14 as original
from scripts import run_image_patch_head_v14 as study

CODE = ('backend/scripts/diagnose_image_patch_head_v14.py',
        'backend/scripts/image_patch_head_diagnostics_runtime_v14.py',
        'backend/tests/test_image_patch_head_v14_diagnostics.py',
        'docs/image-patch-head-v14-diagnostic-plan.md')


def assess_mean(features, average):
    accurate = features.astype(np.float64).mean(axis=0)
    difference = float(np.abs(accurate - average).max())
    return {'maximum_difference': difference, 'within_original_1e_5_bound': difference <= 1e-5}, accurate.astype(np.float32)


def admission(mean_failures):
    return {'original_strict_audit_passed': False, 'original_mean_bound_failures': mean_failures,
            'serving_admitted': False, 'used_in_confidence': False,
            'meaning': 'Diagnostic completion never overrides the retained strict-audit failure or original quality rules.'}


def prepare(trial):
    study.check(trial)
    failed = study.read(trial / 'audit-profile/profile.json')
    log = trial / 'audit-profile/worker.log'
    method.require(failed['child_exit_code'] == 1 and failed['guard'] is None
                   and 'Independent full-image mean exceeds frozen arithmetic bound' in log.read_text()
                   and not (trial / 'audit.json').exists(), 'Original terminal failed audit must remain unamended')
    freeze = {'schema': 'chipmunk.image-head-mean-diagnostics-freeze/14',
        'original_freeze_sha256': study.digest(trial / 'freeze.json'),
        'original_result_sha256': study.digest(trial / 'result.json'),
        'fitted_point_sha256': study.digest(trial / 'fitted-point-private.json'),
        'native_journal_sha256': study.digest(trial / 'native-private.jsonl'),
        'outcomes_sha256': study.digest(trial / 'outcomes-private.json'),
        'failed_audit_profile_sha256': study.digest(trial / 'audit-profile/profile.json'),
        'failed_audit_log_sha256': study.digest(log),
        'source_sha256': {p: study.digest(study.ROOT / p) for p in CODE},
        'plan': 'All1344 native cases/18 full hooks;exact original fit vectors/maps/masks;record every original1e-5 mean failure;retain3e-5 dense arithmetic bound;fixed-head accurate-mean sensitivity only,no refit or point changes.',
        'diagnostic_backbone_forwards_planned': 1362, 'diagnostic_forwards_before_freeze': 0,
        'original_strict_audit_passed': False, 'serving_admitted': False, 'used_in_confidence': False}
    study.write(trial / 'diagnostic-freeze.json', freeze)
    study.write(study.ROOT / 'docs/evaluation-panels/image-patch-head-v14-diagnostic-freeze.json', freeze)
    print({'diagnostic_cases': 1344, 'original_strict_audit_passed': False})


def check(trial):
    plan, freeze = study.check(trial), study.read(trial / 'diagnostic-freeze.json')
    method.require(freeze['source_sha256'] == {p: study.digest(study.ROOT / p) for p in CODE}
                   and freeze['original_freeze_sha256'] == study.digest(trial / 'freeze.json')
                   and freeze['original_result_sha256'] == study.digest(trial / 'result.json')
                   and freeze['fitted_point_sha256'] == study.digest(trial / 'fitted-point-private.json')
                   and freeze['native_journal_sha256'] == study.digest(trial / 'native-private.jsonl')
                   and freeze['outcomes_sha256'] == study.digest(trial / 'outcomes-private.json')
                   and freeze['failed_audit_profile_sha256'] == study.digest(trial / 'audit-profile/profile.json')
                   and freeze['failed_audit_log_sha256'] == study.digest(trial / 'audit-profile/worker.log')
                   and not (trial / 'audit.json').exists(), 'Original failed evidence or diagnostic plan changed')
    return plan


def run(trial):
    from safetensors.numpy import load_file as load_numpy
    from safetensors.torch import load_file

    plan, frozen = check(trial), study.read(trial / 'fitted-point-private.json')
    private = trial / 'private'
    method.require(frozen['cases_before_point'] == 960 and frozen['point_ai_cases_scored'] == 0
                   and frozen['diagnostic_cases_scored'] == 0, 'Original genuine-only point snapshot required')
    for path, expected in frozen['fit_arrays_sha256'].items():
        method.require(study.digest(private / path) == expected, 'Original fit array changed')
    method.require(study.digest(private / 'head.safetensors') == frozen['head_sha256'], 'Original fixed head required')
    native = study.read(trial / 'result.json')
    method.require(study.digest(trial / 'native-private.jsonl') == native['journal_sha256'], 'Original complete native journal required')
    rows = [json.loads(line) for line in (trial / 'native-private.jsonl').read_text().splitlines()]
    expected = {c['id']: c for c in plan['cases']}
    outcomes = {r['id']: r for r in study.read(trial / 'outcomes-private.json')}
    method.require(len(rows) == len(expected) == len({r['id'] for r in rows}) == 1344 and len(outcomes) == 504,
                   'All original cases and outcomes required')
    model, head = study.load_backbone(), method.new_head()
    head.load_state_dict(load_file(str(private / 'head.safetensors')), strict=True)
    head.eval().requires_grad_(False)
    weights = {k: v.astype(np.float64) for k, v in load_numpy(private / 'head.safetensors').items()}
    samples = np.load(private / 'fit-vectors.npy', mmap_mode='r', allow_pickle=False)
    means = np.load(private / 'fit-means.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(private / 'fit-targets.npy', mmap_mode='r', allow_pickle=False)
    failures, changed, fit_seen, maxima = Counter(), Counter(), set(), {}
    largest_mean, largest_head, largest_effect, forwards, score_cells, source_pixels, near = 0., 0., 0., 0, 0, 0, 0
    first_failure_index = None
    with (trial / 'mean-diagnostics-private.jsonl').open('x') as journal:
        for index, row in enumerate(rows):
            case = expected[row['id']]
            method.require(all(row[k] == v for k, v in case.items()), 'Original case identity or role changed')
            entry, rgb, truth = study.pixels_for(plan, case)
            method.require(method.sha(rgb) == row['pixels_sha256'] and method.sha(truth) == row['truth_sha256'],
                           'Original source and intervention pixels changed')
            probe = row['id'] in plan['probe_cases']
            values, average, tensor = method.extract(model, rgb, hook=probe)
            forwards += 1 + int(probe)
            method.require(method.sha(values) == row['features_sha256'] and method.sha(average) == row['mean_sha256']
                           and tensor == row['tensor_sha256'] and row['hook_exact'] == probe,
                           'Complete native feature/input/mean replay differs')
            diagnostic, accurate_average = assess_mean(values, average)
            largest_mean = max(largest_mean, diagnostic['maximum_difference'])
            if not diagnostic['within_original_1e_5_bound']:
                failures[row['role']] += 1
                if first_failure_index is None:
                    first_failure_index = index
            item = {'id': row['id'], 'role': row['role'], **diagnostic}
            source_pixels += truth.size
            if row['role'] == 'fit':
                soft = original.independent_truth(truth)
                chosen, fit = original.independent_indices(soft, entry['key']), row['fit_index']
                method.require(fit not in fit_seen and np.array_equal(samples[fit], values[chosen])
                               and np.array_equal(means[fit], average) and np.array_equal(targets[fit], soft.ravel()[chosen])
                               and method.sha(chosen) == row['sampled_indices_sha256'], 'Sampled native training features/labels differ')
                fit_seen.add(fit)
            else:
                scores = np.load(private / row['scores_file'], allow_pickle=False)
                method.require(study.digest(private / row['scores_file']) == row['scores_file_sha256']
                               and method.sha(scores) == row['scores_sha256']
                               and np.array_equal(method.response(head, values, average), scores), 'Exact original response replay differs')
                reference = original.numpy_scores(values, average, weights)
                largest_head = max(largest_head, float(np.abs(reference - scores).max()))
                method.require(largest_head <= 3e-5, 'Original separate dense arithmetic bound fails')
                point = frozen['point']['value']
                near += int((np.abs(reference - point) <= 3e-5).sum())
                mask = original.independent_mask(scores, point, row['width'], row['height'])
                method.require(original.metrics(mask, truth) == outcomes[row['id']]['metrics']
                               and method.sha(mask) == outcomes[row['id']]['mask_sha256'], 'Original source mask/metric replay differs')
                accurate_response = original.numpy_scores(values, accurate_average, weights)
                # Compare float64 paths with one changed input. This is a fixed-
                # head sensitivity probe, not training or an alternative result.
                effect = float(np.abs(reference - accurate_response).max())
                largest_effect = max(largest_effect, effect)
                probe_mask = original.independent_mask(accurate_response, point, row['width'], row['height'])
                original_reference_mask = original.independent_mask(reference, point, row['width'], row['height'])
                if not np.array_equal(probe_mask, original_reference_mask):
                    changed[row['role']] += 1
                item['fixed_head_accurate_mean_response_effect'] = effect
                item['float64_sensitivity_mask_changed'] = not np.array_equal(probe_mask, original_reference_mask)
                score_cells += scores.size
                if row['role'] == 'point' and row['label'] == 0:
                    maxima.setdefault(row['group'], []).append(float(scores.max()))
            journal.write(json.dumps(item) + '\n')
            journal.flush()
            if (index + 1) % 160 == 0:
                print({'diagnostic_replay_cases': index + 1, 'planned': 1344, 'original_mean_failures': sum(failures.values())}, flush=True)
    ordered = sorted((max(v) for v in maxima.values()), reverse=True)
    method.require(len(maxima) == 60 and all(len(v) == 2 for v in maxima.values())
                   and ordered[3] == frozen['point']['value'] == native['point']['value']
                   and forwards == 1362 and len(fit_seen) == 840 and score_cells == 516096,
                   'Complete native counts, all fit samples and unchanged genuine point required')
    result = study.summarize(plan, list(outcomes.values()))
    method.require(all(result[k] == native[k] for k in result), 'Original outcome aggregates differ')
    check(trial)
    study.write(trial / 'diagnostic-replay.json', {
        'schema': 'chipmunk.image-head-mean-diagnostic-replay/14', 'cases': 1344,
        'backbone_forwards': forwards, 'native_hook_probes': 18,
        'native_feature_values_replayed': 1344 * 1024 * 2304,
        'fit_sample_rows_replayed': 840, 'sampled_patch_vectors_replayed': 53760,
        'source_pixels_replayed_including_variants': source_pixels, 'native_response_cells_replayed': score_cells,
        'native_features_and_maps_exact': True, 'source_truth_samples_and_projection_exact': True,
        'original_genuine_point_and_aggregates_exact': True,
        'original_mean_failure_counts_by_role': dict(failures), 'first_replayed_mean_failure_index': first_failure_index,
        'independent_mean_max_difference': largest_mean, 'numpy_head_response_max_difference': largest_head,
        'fixed_head_accurate_mean_max_response_effect': largest_effect,
        'fixed_head_float64_sensitivity_mask_changes_by_role': dict(changed),
        'response_cells_within_original_dense_tolerance_of_point': near,
        'sensitivity_is_a_refitted_candidate': False, 'optimizer_independently_refit': False,
        'head_or_point_changed': False, 'outside_human_reviewers': 0,
        **admission(sum(failures.values()))})
    print({'completed_diagnostic_replay': 1344, 'original_mean_failures': sum(failures.values()),
           'mean_max_difference': largest_mean, 'head_arithmetic_max_difference': largest_head,
           'original_strict_audit_passed': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.trial)
