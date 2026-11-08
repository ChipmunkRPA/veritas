"""Separate complete native/source replay and NumPy head/coordinate review."""
from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts import image_patch_head_v14 as method
from scripts import run_image_patch_head_v14 as study


def independent_truth(truth):
    height, width = truth.shape
    integral = np.pad(truth.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    out = np.empty((32, 32), np.float32)
    for y in range(32):
        a, b = math.ceil(y * height / 32), math.ceil((y + 1) * height / 32)
        for x in range(32):
            c, d = math.ceil(x * width / 32), math.ceil((x + 1) * width / 32)
            out[y, x] = (integral[b, d] - integral[a, d] - integral[b, c] + integral[a, c]) / ((b - a) * (d - c))
    return out


def independent_indices(truth, key):
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    a, b = np.where(truth.flatten() > 0)[0], np.where(truth.flatten() == 0)[0]
    if len(a) and len(b):
        return np.concatenate((rng.choice(a, 32, replace=len(a) < 32), rng.choice(b, 32, replace=len(b) < 32)))
    return rng.choice(1024, 64, replace=False)


def numpy_scores(features, average, weights):
    values = np.concatenate((features, features - average), axis=-1).astype(np.float64)
    hidden = np.maximum(values @ weights['0.weight'].T + weights['0.bias'], 0)
    response = expit(hidden @ weights['2.weight'].T + weights['2.bias']).reshape(32, 32)
    padded = np.pad(response, 1, mode='symmetric')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1))


def independent_mask(scores, point, width, height):
    alive, seen, keep = scores > point, set(), set()
    for row, col in zip(*np.where(alive), strict=True):
        start = (int(row), int(col))
        if start in seen:
            continue
        stack, region = [start], []
        seen.add(start)
        while stack:
            y, x = stack.pop()
            region.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if 0 <= neighbor[0] < 32 and 0 <= neighbor[1] < 32 and neighbor not in seen and alive[neighbor]:
                        seen.add(neighbor)
                        stack.append(neighbor)
        if len(region) >= 4:
            keep.update(region)
    out = np.zeros((height, width), bool)
    for y, x in keep:
        out[math.ceil(y * height / 32):math.ceil((y + 1) * height / 32),
            math.ceil(x * width / 32):math.ceil((x + 1) * width / 32)] = True
    return out


def metrics(mask, truth):
    a, b, c = (int(np.count_nonzero(mask & truth)), int(np.count_nonzero(mask & ~truth)),
               int(np.count_nonzero(~mask & truth)))
    return {'intersection': a, 'false_positive_pixels': b, 'false_negative_pixels': c,
            'reference_positive_pixels': int(np.count_nonzero(truth)),
            'predicted_positive_pixels': int(np.count_nonzero(mask)), 'pixels': truth.size,
            'iou': a / (a + b + c) if a + b + c else None,
            'pixel_f1': 2 * a / (2 * a + b + c) if 2 * a + b + c else None,
            'flag': bool(np.any(mask)), 'selected_area_fraction': np.count_nonzero(mask) / truth.size}


def review(trial, output):
    import json

    from safetensors.numpy import load_file as load_numpy
    from safetensors.torch import load_file

    plan = study.check(trial)
    result, frozen = study.read(trial / 'result.json'), study.read(trial / 'fitted-point-private.json')
    private = trial / 'private'
    method.require(frozen['cases_before_point'] == 960 and frozen['point_ai_cases_scored'] == 0
                   and frozen['diagnostic_cases_scored'] == 0, 'Genuine-only point must precede held-role AI scoring')
    for path, expected in frozen['fit_arrays_sha256'].items():
        method.require(study.digest(private / path) == expected, 'Original fit sample array changed')
    method.require(study.digest(private / 'head.safetensors') == frozen['head_sha256']
                   and study.digest(trial / 'native-private.jsonl') == result['journal_sha256'],
                   'Original fitted head and complete native journal required')
    rows = [json.loads(line) for line in (trial / 'native-private.jsonl').read_text().splitlines()]
    planned = {c['id']: c for c in plan['cases']}
    method.require(len(rows) == len(planned) == 1344 and len({r['id'] for r in rows}) == 1344,
                   'No missing, duplicated or omitted cases')
    outcomes = {r['id']: r for r in study.read(trial / 'outcomes-private.json')}
    method.require(len(outcomes) == 504, 'All point and diagnostic outcomes required')
    backbone, head = study.load_backbone(), method.new_head()
    head.load_state_dict(load_file(str(private / 'head.safetensors')), strict=True)
    head.eval().requires_grad_(False)
    weights = {k: v.astype(np.float64) for k, v in load_numpy(private / 'head.safetensors').items()}
    samples = np.load(private / 'fit-vectors.npy', mmap_mode='r', allow_pickle=False)
    means = np.load(private / 'fit-means.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(private / 'fit-targets.npy', mmap_mode='r', allow_pickle=False)
    largest, mean_difference, forwards, score_cells, source_pixels, near_point = 0., 0., 0, 0, 0, 0
    group_maxima, fit_seen = defaultdict(list), set()
    for index, row in enumerate(rows):
        case = planned[row['id']]
        method.require(all(row[k] == v for k, v in case.items()), 'Case identity or source-role assignment changed')
        entry, rgb, truth = study.pixels_for(plan, case)
        method.require(method.sha(rgb) == row['pixels_sha256'] and method.sha(truth) == row['truth_sha256'],
                       'Exact source/intervention pixels and truth required')
        probe = case['id'] in plan['probe_cases']
        values, average, tensor = method.extract(backbone, rgb, hook=probe)
        forwards += 1 + int(probe)
        method.require(tensor == row['tensor_sha256'] and method.sha(values) == row['features_sha256']
                       and method.sha(average) == row['mean_sha256'] and row['hook_exact'] == probe,
                       'Native source/tensor/layer-feature replay differs')
        mean_difference = max(mean_difference, float(np.abs(values.astype(np.float64).mean(0) - average).max()))
        method.require(mean_difference <= 1e-5, 'Independent full-image mean exceeds frozen arithmetic bound')
        source_pixels += truth.size
        if row['role'] == 'fit':
            soft = independent_truth(truth)
            chosen = independent_indices(soft, entry['key'])
            fit = row['fit_index']
            method.require(fit not in fit_seen and np.array_equal(values[chosen], samples[fit])
                           and np.array_equal(average, means[fit]) and np.array_equal(soft.ravel()[chosen], targets[fit])
                           and method.sha(chosen) == row['sampled_indices_sha256'],
                           'Complete sampled fit features, soft truth and coordinates must replay exactly')
            fit_seen.add(fit)
        else:
            scores = np.load(private / row['scores_file'], allow_pickle=False)
            method.require(study.digest(private / row['scores_file']) == row['scores_file_sha256']
                           and method.sha(scores) == row['scores_sha256']
                           and np.array_equal(method.response(head, values, average), scores),
                           'Retained full response grid and independent native repeat must agree exactly')
            reference = numpy_scores(values, average, weights)
            largest = max(largest, float(np.abs(reference - scores).max()))
            method.require(largest <= 3e-5, 'Separate float64 dense/ReLU/sigmoid/median arithmetic differs')
            point = frozen['point']['value']
            near_point += int((np.abs(reference - point) <= 3e-5).sum())
            mask = independent_mask(scores, point, row['width'], row['height'])
            method.require(method.sha(mask) == outcomes[row['id']]['mask_sha256']
                           and metrics(mask, truth) == outcomes[row['id']]['metrics'],
                           'Separate components, source projection or pixel metrics differ')
            score_cells += scores.size
            if row['role'] == 'point' and row['label'] == 0:
                group_maxima[row['group']].append(float(scores.max()))
        if (index + 1) % 160 == 0:
            print({'replayed_cases': index + 1, 'planned': 1344, 'backbone_forwards': forwards}, flush=True)
    values = sorted((max(v) for v in group_maxima.values()), reverse=True)
    method.require(len(values) == 60 and all(len(v) == 2 for v in group_maxima.values())
                   and values[3] == frozen['point']['value'] == result['point']['value']
                   and len(fit_seen) == 840 and forwards == 1362 and score_cells == 516096,
                   'All fit rows, independent genuine-only point and complete native/grid counts required')
    recomputed = study.summarize(plan, list(outcomes.values()))
    method.require(all(recomputed[k] == result[k] for k in recomputed), 'Aggregate denominators or continuation rules differ')
    study.check(trial)
    study.write(output, {'schema': 'chipmunk.image-patch-head-native-audit/14', 'cases': 1344,
        'source_pixels_replayed_including_variants': source_pixels, 'fit_sample_rows_replayed': 840,
        'sampled_patch_vectors_replayed': 53760, 'native_hook_probes': 18, 'backbone_forwards': forwards,
        'native_feature_values_replayed': 1344 * 1024 * 2304, 'native_response_cells_replayed': score_cells,
        'numpy_head_response_max_difference': largest, 'independent_mean_max_difference': mean_difference,
        'response_cells_within_arithmetic_tolerance_of_point': near_point,
        'native_feature_and_map_replay_exact': True, 'source_truth_and_projection_replay_exact': True,
        'genuine_point_refit_exact': True, 'optimizer_independently_refit': False,
        'outside_human_reviewers': 0, 'serving_admitted': False, 'used_in_confidence': False})
    print({'audited_cases': 1344, 'backbone_forwards': forwards, 'head_arithmetic_max_difference': largest}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
