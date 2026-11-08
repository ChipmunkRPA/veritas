"""Independent activation search, source geometry, native and head replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import image_region_point_v15 as method
from scripts import review_image_patch_head_v14 as reference
from scripts import run_image_region_point_v15 as study


def has_region(binary, minimum=4):
    height, width = binary.shape
    seen = set()
    for y, x in zip(*np.where(binary), strict=True):
        start = (int(y), int(x))
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            row, col = stack.pop()
            size += 1
            if size >= minimum:
                return True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    other = (row + dy, col + dx)
                    if 0 <= other[0] < height and 0 <= other[1] < width and other not in seen and binary[other]:
                        seen.add(other)
                        stack.append(other)
    return False


def activation(scores, minimum=4):
    """Binary search and separate flood fill, not the primary union algorithm."""
    values = sorted(set(map(float, scores.ravel())))
    low, high, answer = 0, len(values) - 1, None
    while low <= high:
        mid = (low + high) // 2
        if has_region(scores >= values[mid], minimum):
            answer, low = values[mid], mid + 1
        else:
            high = mid - 1
    study.old.method.require(answer is not None, 'Complete feasible reference grid required')
    return answer


def mean_reference(features):
    return features.astype(np.float64).reshape(16, 64, -1).sum(axis=1).sum(axis=0) / 1024


def review(trial, output):
    from safetensors.numpy import load_file as load_numpy
    from safetensors.torch import load_file

    plan, previous = study.check(trial), study.prior_trial()
    fixed = study.read(trial / 'fixed-points-private.json')
    native = study.read(trial / 'result.json')
    check = study.old.method.require
    private = trial / 'private'
    check(fixed['new_cases_before_points'] == 960 and fixed['new_ai_point_cases_scored'] == 0
          and fixed['new_diagnostic_cases_scored'] == 0, 'Genuine-only point isolation must remain exact')
    check(study.digest(private / 'head.safetensors') == fixed['head_sha256']
          and study.digest(private / 'stable-fit-means.npy') == fixed['stable_means_sha256']
          and study.digest(trial / 'native-private.jsonl') == native['native_journal_sha256'], 'Complete original new-head evidence required')
    rows = [json.loads(line) for line in (trial / 'native-private.jsonl').read_text().splitlines()]
    planned, previous_rows = {c['id']: c for c in plan['cases']}, study.old_rows()
    check(len(rows) == len(planned) == len({r['id'] for r in rows}) == 1344, 'Every original planned input required')
    outcomes = {(r['id'], r['arm']): r for r in study.read(trial / 'outcomes-private.json')}
    check(len(outcomes) == 2016, 'All4 arms and every point/diagnostic outcome required')
    model = study.old.load_backbone()
    heads, weights = {}, {}
    for name, path in (('old', previous / 'private/head.safetensors'), ('stable', private / 'head.safetensors')):
        head = study.old.method.new_head()
        head.load_state_dict(load_file(str(path)), strict=True)
        heads[name] = head.eval().requires_grad_(False)
        weights[name] = {k: v.astype(np.float64) for k, v in load_numpy(path).items()}
    samples = np.load(previous / 'private/fit-vectors.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(previous / 'private/fit-targets.npy', mmap_mode='r', allow_pickle=False)
    means = np.load(private / 'stable-fit-means.npy', mmap_mode='r', allow_pickle=False)
    counts = defaultdict(int)
    calibration = defaultdict(list)
    largest_mean, largest_head, old_mean_failures, forwards, source_pixels = 0., 0., 0, 0, 0
    seen_fit = set()
    for index, row in enumerate(rows):
        case = planned[row['id']]
        check(all(row[k] == v for k, v in case.items()), 'New case identity or role changed')
        entry, rgb, truth = study.old.pixels_for(plan, case)
        check(study.old.method.sha(rgb) == row['pixels_sha256'] and study.old.method.sha(truth) == row['truth_sha256'],
              'Exact source/intervention pixels and truth required')
        probe = row['id'] in plan['probe_cases']
        features, average, tensor = method.extract(model, rgb, hook=probe)
        forwards += 1 + int(probe)
        check(study.old.method.sha(features) == row['features_sha256'] == previous_rows[row['id']]['features_sha256']
              and study.old.method.sha(average) == row['stable_mean_sha256'] and tensor == row['tensor_sha256'],
              'Complete exact native feature/context replay required')
        accurate = mean_reference(features)
        largest_mean = max(largest_mean, float(np.abs(average - accurate).max()))
        check(largest_mean <= 1e-5, 'New stable context must satisfy the unchanged strict mean bound')
        old_average = features.mean(axis=0, dtype=np.float32)
        old_mean_failures += int(np.abs(old_average - accurate).max() > 1e-5)
        check(study.old.method.sha(old_average) == previous_rows[row['id']]['mean_sha256'], 'Original old mean changed')
        source_pixels += truth.size
        if row['role'] == 'fit':
            soft = reference.independent_truth(truth)
            chosen, fit = reference.independent_indices(soft, entry['key']), row['fit_index']
            check(fit not in seen_fit and np.array_equal(features[chosen], samples[fit])
                  and np.array_equal(average, means[fit]) and np.array_equal(soft.ravel()[chosen], targets[fit]),
                  'Every reused fit vector, new mean and exact target must reproduce')
            seen_fit.add(fit)
        else:
            new_scores = np.load(private / row['scores_file'], allow_pickle=False)
            check(study.digest(private / row['scores_file']) == row['scores_file_sha256']
                  and study.old.method.sha(new_scores) == row['scores_sha256'], 'Retained new grid changed')
            for name, scores, context in (('old', study.old_scores(previous_rows[row['id']]), old_average),
                                          ('stable', new_scores, average)):
                check(scores.dtype == np.float32 and np.array_equal(study.old.method.response(heads[name], features, context), scores),
                      'Every old/new native ranking grid must repeat exactly')
                separate = reference.numpy_scores(features, context, weights[name])
                largest_head = max(largest_head, float(np.abs(separate - scores).max()))
                check(largest_head <= 3e-5, 'Unchanged independent dense-head arithmetic bound failed')
                independent_level = activation(scores)
                check(independent_level == method.activation(scores), 'Union activation and binary-search flood fill differ')
                counts['activation_levels_replayed'] += 1
                counts['native_response_cells_replayed'] += scores.size
                if row['role'] == 'point' and row['label'] == 0:
                    calibration[name].append({**{k: row[k] for k in ('id', 'group', 'role', 'label', 'variant')},
                                              'maximum': float(scores.max()), 'activation': independent_level})
                for suffix in ('maximum', 'regions'):
                    arm = name + '_' + suffix
                    point = fixed['points'][arm]['value']
                    mask = reference.independent_mask(scores, point, row['width'], row['height'])
                    check(bool(mask.any()) == (independent_level > point)
                          and reference.metrics(mask, truth) == outcomes[row['id'], arm]['metrics']
                          and study.old.method.sha(mask) == outcomes[row['id'], arm]['mask_sha256'],
                          'Separate components/source geometry/counts or strict activation equivalence failed')
                    counts['source_mask_metric_sets_replayed'] += 1
        if (index + 1) % 160 == 0:
            print({'replayed_cases': index + 1, 'planned': 1344, 'stable_mean_max_difference': largest_mean}, flush=True)
    for name, records in calibration.items():
        for suffix, kind in (('maximum', 'maximum'), ('regions', 'regions')):
            groups = defaultdict(list)
            for row in records:
                groups[row['group']].append(row['maximum'] if kind == 'maximum' else row['activation'])
            values = sorted((max(v) for v in groups.values()), reverse=True)
            check(len(groups) == 60 and all(len(v) == 2 for v in groups.values())
                  and values[3] == fixed['points'][name + '_' + suffix]['value'], 'Independent genuine-only point refit differs')
    check(forwards == 1362 and len(seen_fit) == 840 and counts['activation_levels_replayed'] == 1008
          and counts['source_mask_metric_sets_replayed'] == 2016 and counts['native_response_cells_replayed'] == 1032192
          and old_mean_failures == 1302, 'Complete declared counts and retained old mean failures required')
    recomputed = study.summarize(plan, list(outcomes.values()))
    check(all(native[k] == recomputed[k] for k in recomputed), 'All aggregate denominators and quality rules must agree')
    study.check(trial)
    study.write(output, {'schema': 'chipmunk.image-region-point-native-audit/15', 'cases': 1344,
        'backbone_forwards': forwards, 'native_hook_probes': 18, 'fit_sample_rows_replayed': 840,
        'sampled_patch_vectors_replayed': 53760, 'native_feature_values_replayed': 1344 * 1024 * 2304,
        'source_pixels_replayed_including_variants': source_pixels, **dict(counts),
        'new_mean_strict_bound': 1e-5, 'stable_mean_max_difference': largest_mean,
        'new_mean_strict_criterion_passed': True, 'old_mean_bound_failures_retained': old_mean_failures,
        'old_strict_audit_remains_failed': True, 'numpy_head_response_max_difference': largest_head,
        'native_features_maps_and_fit_samples_exact': True, 'source_projection_metrics_exact': True,
        'independent_activation_and_point_refits_exact': True, 'optimizer_independently_refit': False,
        'outside_human_reviewers': 0, 'serving_admitted': False, 'used_in_confidence': False})
    print({'audited_cases': 1344, 'new_mean_max_difference': largest_mean, 'old_mean_failures_retained': old_mean_failures}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
