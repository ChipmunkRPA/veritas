"""Separate numerical/region audit of every patch-drift development outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def median_reference(value):
    padded = np.pad(value, 1, mode='reflect')
    return np.median(np.stack([padded[y:y + value.shape[0], x:x + value.shape[1]]
        for y in range(3) for x in range(3)]), axis=0).astype(np.float32)


def readout_reference(value, arm):
    if arm == 'raw':
        return value.copy()
    if arm != 'within_image':
        raise ValueError('Unexpected readout')
    ordered = sorted(float(v) for v in value.flat)

    def quantile(q):
        position = (len(ordered) - 1) * q
        left = int(position)
        return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)

    median = (ordered[511] + ordered[512]) / 2
    # The production primitive takes the float32 np.median result before float64
    # subtraction, so retain that rounding rather than a different estimator.
    center = float(np.float32(median))
    scale = max(quantile(.75) - quantile(.25), 2**-23)
    return ((value.astype(np.float64) - center) / scale).astype(np.float32)


def display_reference(floating, reliable, point, rule, width, height):
    if not math.isfinite(point) or point <= 0:
        raise ValueError('Positive genuine-only point required')
    eligible = (floating > point) & reliable
    values = np.rint(np.clip(floating.astype(np.float64) / (2 * point), 0, 1) * 255).astype(np.uint8)
    values[floating > point] = np.maximum(values[floating > point], 128)
    values[floating <= point] = np.minimum(values[floating <= point], 127)
    # Independent iterative8-neighbour traversal, not cv2 component labels.
    unseen = set(zip(*np.nonzero(eligible), strict=True))
    components = []
    minimum = max(1, math.ceil(rule['minimum_region_fraction'] * eligible.size))
    gh, gw = eligible.shape
    while unseen:
        todo = [unseen.pop()]
        members = []
        while todo:
            y, x = todo.pop()
            members.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        todo.append(neighbor)
        if len(members) >= minimum:
            yy, xx = np.asarray(members).T
            mean = float((values[yy, xx].astype(np.float64) / 255).mean())
            components.append(((-round(mean, 4), -round(len(members) / eligible.size, 4),
                round(int(yy.min()) * height / gh, 1), round(int(xx.min()) * width / gw, 1)), members))
    selected = np.zeros_like(eligible)
    cap = min(rule['localiser_max_regions'], rule['per_map_display_max_regions'])
    for _, members in sorted(components, key=lambda entry: entry[0])[:cap]:
        yy, xx = np.asarray(members).T
        selected[yy, xx] = True
    return values, selected


def pixel_reference(grid, truth):
    height, width = truth.shape
    gh, gw = grid.shape
    projected = grid[(np.arange(height) * gh // height)[:, None], np.arange(width) * gw // width]
    both = int(np.count_nonzero(projected & truth))
    false = int(np.count_nonzero(projected & ~truth))
    missed = int(np.count_nonzero(~projected & truth))
    union, denominator = both + false + missed, 2 * both + false + missed
    return {'intersection': both, 'false_positive_pixels': false, 'false_negative_pixels': missed,
        'pixels': int(truth.size), 'reference_positive_pixels': both + missed,
        'predicted_positive_pixels': both + false, 'iou': both / union if union else 1.,
        'pixel_f1': 2 * both / denominator if denominator else 1.,
        'precision': both / (both + false) if both + false else None,
        'recall': both / (both + missed) if both + missed else None,
        'flagged_area_fraction': (both + false) / truth.size}


def summarize(rows, arm):
    edited, genuine = [r for r in rows if r['label']], [r for r in rows if not r['label']]

    def one(group):
        edits = [r['baseline'] if arm == 'baseline' else r['arms'][arm]['metrics'] for r in group if r['label']]
        controls = [r['baseline'] if arm == 'baseline' else r['arms'][arm]['metrics'] for r in group if not r['label']]
        return {'edited_count': len(edits), 'genuine_count': len(controls),
            'overlapping_edits': sum(r['intersection'] > 0 for r in edits),
            'missed_edits': sum(r['intersection'] == 0 for r in edits),
            'mean_edit_iou': sum(r['iou'] for r in edits) / len(edits),
            'genuine_highlights': sum(r['predicted_positive_pixels'] > 0 for r in controls),
            'genuine_mean_highlight_fraction': sum(r['flagged_area_fraction'] for r in controls) / len(controls)}

    if not edited or not genuine:
        raise ValueError('Complete edited and genuine denominators required')
    return {'overall': one(rows), 'families': {f: one([r for r in rows if r['family'] == f])
        for f in sorted({r['family'] for r in rows})}}


def gates(baseline, candidate, complete, verified):
    b, c = baseline['overall'], candidate['overall']
    return {'complete': complete, 'independent_checks': verified,
        'at_least_four_more_overlapping_edits': c['overlapping_edits'] >= b['overlapping_edits'] + 4,
        'mean_edit_iou_not_lower': c['mean_edit_iou'] >= b['mean_edit_iou'],
        'genuine_highlights_within_one': c['genuine_highlights'] <= b['genuine_highlights'] + 1,
        'no_family_overlap_loss': all(candidate['families'][f]['overlapping_edits'] >=
            baseline['families'][f]['overlapping_edits'] for f in baseline['families'])}


def review(main, trial, expected, points_sha):
    from scripts import run_image_patch_drift_v8 as runner
    from scripts.localization_benchmark import source

    lock, panel = runner.check(main, trial, expected)
    if digest(trial / 'operating-points.json') != points_sha:
        raise ValueError('Points receipt changed')
    points = json.loads((trial / 'operating-points.json').read_text())
    verified, vectors, worst_cosine = 0, 0, 0.
    all_rows = []
    for role, count in (('calibration', 120), ('development', 192)):
        file = trial / role / 'predictions.jsonl'
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        items = {(r['id'], r['condition']): r for r in panel if r['role'] == role}
        if len(rows) != count or len({(r['id'], r['condition']) for r in rows}) != count:
            raise ValueError('Missing or duplicate original outcome')
        for row in rows:
            item = items[row['id'], row['condition']]
            if row['status'] != 'ok' or any(row[k] != item[k] for k in ('id', 'source_id', 'family', 'label',
                    'role', 'condition', 'input_sha256', 'input_rgb_sha256', 'geometry', 'native_probe')):
                raise ValueError('Incomplete or changed frozen case')
            path = trial / role / row['arrays']
            if digest(path) != row['arrays_sha256']:
                raise ValueError('Saved response arrays changed')
            original, rgb, truth = source(runner.paths(main)[role], item)
            data, pixels = runner.variant(rgb, original, row['condition'])
            if hashlib.sha256(data).hexdigest() != row['input_sha256']:
                raise ValueError('Input byte binding failed')
            clean, changed, normalized = __import__('scripts.image_patch_drift_v8', fromlist=['views']).views(pixels)
            for name, value in (('clean_float32', clean), ('changed_float32', changed), ('normalized_nchw_float32', normalized)):
                if hashlib.sha256(value.tobytes()).hexdigest() != row['native'][name + '_sha256']:
                    raise ValueError('Native input tensor binding failed')
            with np.load(path, allow_pickle=False) as arrays:
                for name, field in (('raw', 'native_raw_map_float32_sha256'), ('filtered', 'filtered_map_float32_sha256')):
                    if hashlib.sha256(arrays[name].tobytes()).hexdigest() != row['native'][field]:
                        raise ValueError('Native map byte binding failed')
                np.testing.assert_array_equal(median_reference(arrays['raw']), arrays['filtered'])
                if row['native_probe']:
                    features = arrays['private_features']
                    if (features.shape != (2, 1024, 768) or features.dtype != np.float32
                            or not row['native']['reference_compared']
                            or row['native']['native_hook_feature_max_difference'] != 0
                            or hashlib.sha256(features.tobytes()).hexdigest() != row['native']['native_features_float32_sha256']):
                        raise ValueError('Private native feature binding failed')
                    a, b = features.astype(np.float64)
                    independent = 1 - np.clip(np.einsum('ij,ij->i', a, b) /
                        np.maximum(np.sqrt(np.einsum('ij,ij->i', a, a) * np.einsum('ij,ij->i', b, b)), 2**-23), -1, 1)
                    difference = float(np.max(np.abs(independent.reshape(32, 32) - arrays['raw'])))
                    if difference > 5e-7:
                        raise ValueError('Separate float64 vector cosine exceeds frozen tolerance')
                    worst_cosine = max(worst_cosine, difference)
                    vectors += 1
                from app.detectors.image import jpeg_bitstream
                from app.detectors.image import localiser as loc

                structure = jpeg_bitstream.read(data)
                excluded, _ = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
                reliable = loc.reliable_grid(excluded, loc.grid_shape(pixels.shape[1], pixels.shape[0],
                    loc.GRID_MAX_SIDE['screening']))
                np.testing.assert_array_equal(reliable, arrays['reliable'])
                for arm in ('raw', 'within_image'):
                    floating = cv2.resize(readout_reference(arrays['filtered'], arm),
                        (reliable.shape[1], reliable.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    np.testing.assert_array_equal(floating, arrays[arm + '_screening'])
                    maximum = float(floating[reliable].max()) if reliable.any() else None
                    if maximum != row['arms'][arm]['eligible_maximum']:
                        raise ValueError('Eligible genuine maximum changed')
                    if role == 'development':
                        encoded, selected = display_reference(floating, reliable, points['points'][row['condition']][arm],
                            lock['component_policy'], pixels.shape[1], pixels.shape[0])
                        np.testing.assert_array_equal(encoded, arrays[arm + '_display_grid'])
                        np.testing.assert_array_equal(selected, arrays[arm + '_selected_grid'])
                        if pixel_reference(selected, truth) != row['arms'][arm]['metrics']:
                            raise ValueError('Independently counted source pixels differ')
            if role == 'development' and row['baseline'] != item['baseline']:
                raise ValueError('Frozen current-check metrics changed')
            verified += 1
        if role == 'calibration':
            if digest(file) != points['calibration_predictions_sha256']:
                raise ValueError('Point-fitting source receipt changed')
            for condition in runner.FORMATS:
                for arm in ('raw', 'within_image'):
                    maxima = sorted((r['arms'][arm]['eligible_maximum'] for r in rows if r['condition'] == condition
                        and r['arms'][arm]['eligible_maximum'] is not None), reverse=True)
                    if len(maxima) < 48 or maxima[1] != points['points'][condition][arm]:
                        raise ValueError('Genuine-only point refit differs')
        all_rows.extend(rows)
    profiles = {role: json.loads((trial / (role + '-profile') / 'profile.json').read_text())
        for role in ('calibration', 'development')}
    successful = all(p['guard'] is None and p['child_exit_code'] == 0 for p in profiles.values())
    conditions = {}
    for condition in runner.FORMATS:
        rows = [r for r in all_rows if r['role'] == 'development' and r['condition'] == condition]
        baseline = summarize(rows, 'baseline')
        arms = {}
        for arm in ('raw', 'within_image'):
            stats = summarize(rows, arm)
            requirements = gates(baseline, stats, len(rows) == 96 and successful, verified == 312 and vectors == 12)
            arms[arm] = {'statistics': stats, 'requirements': requirements, 'passed': all(requirements.values())}
        conditions[condition] = {'baseline': baseline, 'arms': arms,
            'selected_arm': next((arm for arm in ('raw', 'within_image') if arms[arm]['passed']), None)}
    result = {'schema': 'chipmunk.patch-drift-development-review/8', 'freeze_sha256': expected,
        'points_sha256': points_sha, 'verified_cases': verified, 'private_native_probes': vectors,
        'worst_float64_cosine_difference': worst_cosine, 'profiles': profiles, 'conditions': conditions,
        'source_roles': 'Consumed development only; no fresh confirmation or outside human review.',
        'serving_changed': False, 'overall_confidence_weight': 0}
    runner.native.write(trial / 'development-review.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--points-sha256', required=True)
    args = parser.parse_args()
    review(args.main, args.trial, args.freeze_sha256, args.points_sha256)
