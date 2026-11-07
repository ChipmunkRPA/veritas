"""Standard-library replay of published scalar counts; not neural inference.

Private source photos, pixel masks and image feature vectors are deliberately not
redistributed. The separate local audit checks those. This portable review binds
source roles, refits genuine-only points, recomputes metrics and freezes decisions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'docs/evaluation-panels'
PREFIX = 'image-v8-patch-drift-'
FORMATS, ARMS = ('native', 'jpeg75'), ('raw', 'within_image')


def read(name):
    return json.loads((PUBLIC / (PREFIX + name + '.json')).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is float or type(b) is float:
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-15
    return type(a) is type(b) and a == b


def pixel_counts(metrics):
    fields = ('intersection', 'false_positive_pixels', 'false_negative_pixels', 'pixels',
        'reference_positive_pixels', 'predicted_positive_pixels')
    require(all(type(metrics.get(k)) is int and metrics[k] >= 0 for k in fields), 'Missing/noninteger pixel denominator')
    a, b, c, n = (metrics[k] for k in fields[:4])
    require(n > 0 and a + b + c <= n and metrics['reference_positive_pixels'] == a + c
        and metrics['predicted_positive_pixels'] == a + b, 'Inconsistent source pixel counts')
    expected = {**{k: metrics[k] for k in fields}, 'iou': a / (a + b + c) if a + b + c else 1.,
        'pixel_f1': 2 * a / (2 * a + b + c) if 2 * a + b + c else 1.,
        'precision': a / (a + b) if a + b else None, 'recall': a / (a + c) if a + c else None,
        'flagged_area_fraction': (a + b) / n}
    require(close(metrics, expected), 'Derived pixel ratios disagree with integer counts')


def summary(rows, arm):
    def one(selected):
        edited = [r['baseline'] if arm == 'baseline' else r['arms'][arm]['metrics'] for r in selected if r['label']]
        originals = [r['baseline'] if arm == 'baseline' else r['arms'][arm]['metrics'] for r in selected if not r['label']]
        require(bool(edited) and bool(originals), 'Missing family denominator')
        return {'edited_count': len(edited), 'genuine_count': len(originals),
            'overlapping_edits': sum(x['intersection'] > 0 for x in edited),
            'missed_edits': sum(x['intersection'] == 0 for x in edited),
            'mean_edit_iou': sum(x['iou'] for x in edited) / len(edited),
            'genuine_highlights': sum(x['predicted_positive_pixels'] > 0 for x in originals),
            'genuine_mean_highlight_fraction': sum(x['flagged_area_fraction'] for x in originals) / len(originals)}

    return {'overall': one(rows), 'families': {f: one([r for r in rows if r['family'] == f])
        for f in sorted({r['family'] for r in rows})}}


def verify(packet):
    freeze, panel, points = read('development-freeze'), read('development-panel'), read('operating-points')
    expected = digest(PUBLIC / (PREFIX + 'development-freeze.json'))
    require(freeze['project_response_count_before_freeze'] == 0 and not freeze['serving_changed'], 'Prospective development contract required')
    require(freeze['minimum_added_overlapping_edits'] == 4 and freeze['maximum_added_genuine_highlights'] == 1
        and freeze['minimum_family_overlap_delta'] == 0 and freeze['minimum_mean_edit_iou_delta'] == 0
        and freeze['arm_priority'] == list(ARMS), 'Unchanged advancement rules required')
    require(digest(PUBLIC / (PREFIX + 'development-panel.json')) == freeze['panel_sha256'], 'Frozen panel bytes changed')
    require(packet['freeze_sha256'] == expected == points['freeze_sha256'], 'Wrong freeze binding')
    require(packet['points_sha256'] == digest(PUBLIC / (PREFIX + 'operating-points.json')) and points['passed']
        and points['edited_images_used'] == 0, 'Genuine-only point receipt changed')
    require(packet['schema'] == 'chipmunk.patch-drift-public-scalars/8'
        and packet['serving_changed'] is False and packet['overall_confidence_weight'] == 0, 'Unsupported service/score claim')
    repeat, amendment, synthetic = read('retry-repeat'), read('operational-amendment'), read('synthetic-native')
    require(packet['resource_retry_repeat'] == repeat and repeat['passed'] is True
        and repeat['prior_complete_records'] == repeat['exact_input_native_map_feature_proofs'] ==
        repeat['exact_arm_metrics_and_baseline'] == 34 and repeat['candidate_code_models_points_and_data_changed'] is False,
        'Missing or changed resource-retry provenance')
    require(amendment['development_freeze_sha256'] == expected and amendment['points_sha256'] == packet['points_sha256']
        and amendment['original_development_attempt']['complete_prediction_records'] == 34
        and amendment['original_development_attempt']['predictions_sha256'] == repeat['previous_predictions_sha256']
        and amendment['retry_plan']['changed_code_models_points_masks_or_advancement_rules'] is False,
        'Unchanged prospective retry contract required')
    require(synthetic['passed'] and len(synthetic['cases']) == 6 and all(r['passed'] for r in synthetic['cases'])
        and synthetic['max_normalization_difference'] == synthetic['max_feature_difference'] == 0,
        'Complete native synthetic receipt required')
    items = {(r['role'], r['id'], r['condition']): r for r in panel['items']}
    rows = packet['cases']
    require(len(items) == len(rows) == 312 and len({(r['role'], r['id'], r['condition']) for r in rows}) == 312,
        'Incomplete or duplicate case denominator')
    require({(r['role'], r['id'], r['condition']) for r in rows} == set(items), 'Unplanned case identity')
    probes = 0
    for row in rows:
        item = items[row['role'], row['id'], row['condition']]
        require(row['status'] == 'ok' and row['freeze_sha256'] == expected, 'Incomplete or wrong-contract case')
        for key in ('source_id', 'family', 'label', 'input_sha256', 'input_rgb_sha256', 'geometry', 'native_probe'):
            require(row[key] == item[key], 'Changed frozen source/role/geometry: ' + key)
        require(set(row['arms']) == set(ARMS), 'Missing predeclared readout')
        for arm in ARMS:
            value = row['arms'][arm]['eligible_maximum']
            require(value is None or (type(value) is float and math.isfinite(value)), 'Invalid scalar response')
        native = row['native']
        require(native['native_shape'] == [2, 1024, 768] and native['dtype'] == 'float32'
            and native['readout_one_based_block'] == 10, 'Changed native model readout')
        if row['native_probe']:
            require(native['reference_compared'] and native['native_hook_feature_max_difference'] == 0., 'Failed native probe')
            probes += 1
        if row['role'] == 'calibration':
            require(row['label'] == 0 and all('metrics' not in a for a in row['arms'].values()), 'Edit outcomes used for calibration')
        else:
            require(row['baseline'] == item['baseline'], 'Changed frozen current-check outcome')
            for metrics in [row['baseline']] + [row['arms'][arm]['metrics'] for arm in ARMS]:
                pixel_counts(metrics)
                require(metrics['pixels'] == row['geometry'][0] * row['geometry'][1], 'Source-pixel denominator differs')
                require((metrics['reference_positive_pixels'] > 0) == bool(row['label']), 'Reference mask/label mismatch')
    require(probes == 12, 'Missing planned native probes')
    for condition in FORMATS:
        selected = [r for r in rows if r['role'] == 'calibration' and r['condition'] == condition]
        require(len(selected) == 60, 'Missing genuine calibration denominator')
        for arm in ARMS:
            maxima = sorted((r['arms'][arm]['eligible_maximum'] for r in selected
                if r['arms'][arm]['eligible_maximum'] is not None), reverse=True)
            require(len(maxima) >= 48 and maxima[1] > 0 and maxima[1] == points['points'][condition][arm], 'Genuine-only point refit failed')
            require(sum(v > maxima[1] for v in maxima) == points['genuine_maxima_above_point'][condition][arm] <= 1,
                'Strict exceedance/calibration budget failed')
    local = packet['local_audit']
    require(local['verified_cases'] == 312 and local['private_native_probes'] == 12
        and local['worst_float64_cosine_difference'] <= 5e-7, 'Local independent arithmetic incomplete')
    require(local['freeze_sha256'] == expected and local['points_sha256'] == packet['points_sha256'], 'Local review binding changed')
    require(all(p['child_exit_code'] == 0 and p['guard'] is None for p in local['profiles'].values())
        and set(local['profiles']) == {'calibration', 'development'}, 'Unfinished/guarded inference')
    recomputed = {}
    for condition in FORMATS:
        selected = [r for r in rows if r['role'] == 'development' and r['condition'] == condition]
        require(len(selected) == 96 and sum(r['label'] for r in selected) == 48, 'Development denominator differs')
        baseline, arms = summary(selected, 'baseline'), {}
        for arm in ARMS:
            candidate = summary(selected, arm)
            b, c = baseline['overall'], candidate['overall']
            requirements = {'complete': True, 'independent_checks': True,
                'at_least_four_more_overlapping_edits': c['overlapping_edits'] >= b['overlapping_edits'] + 4,
                'mean_edit_iou_not_lower': c['mean_edit_iou'] >= b['mean_edit_iou'],
                'genuine_highlights_within_one': c['genuine_highlights'] <= b['genuine_highlights'] + 1,
                'no_family_overlap_loss': all(candidate['families'][f]['overlapping_edits'] >= baseline['families'][f]['overlapping_edits']
                    for f in baseline['families'])}
            arms[arm] = {'statistics': candidate, 'requirements': requirements, 'passed': all(requirements.values())}
        recomputed[condition] = {'baseline': baseline, 'arms': arms,
            'selected_arm': next((arm for arm in ARMS if arms[arm]['passed']), None)}
    require(close(recomputed, local['conditions']), 'Changed aggregate/advancement decision')
    return {'schema': 'chipmunk.patch-drift-portable-review/8', 'passed': True, 'verified_cases': len(rows),
        'genuine_points_refitted': 4, 'development_metric_sets_recomputed': 576, 'native_probe_receipts_bound': probes,
        'conditions': recomputed, 'scope': 'Standard-library scalar/source-binding/count replay; no neural inference or outside human review.'}


def tamper(packet):
    tests = []
    changes = {
        'missing_case': lambda p: p['cases'].pop(),
        'duplicated_case': lambda p: p['cases'].__setitem__(1, copy.deepcopy(p['cases'][0])),
        'changed_source': lambda p: p['cases'][0].__setitem__('source_id', 'changed'),
        'changed_role': lambda p: p['cases'][0].__setitem__('role', 'development'),
        'changed_label': lambda p: p['cases'][0].__setitem__('label', 1),
        'changed_input': lambda p: p['cases'][0].__setitem__('input_sha256', '0' * 64),
        'missing_native_probe': lambda p: next(r for r in p['cases'] if r['native_probe']).__setitem__('native_probe', False),
        'changed_genuine_maximum': lambda p: next(r for r in p['cases'] if r['role'] == 'calibration')['arms']['raw'].__setitem__('eligible_maximum', 1e6),
        'changed_pixel_count': lambda p: next(r for r in p['cases'] if r['role'] == 'development')['arms']['raw']['metrics'].__setitem__('pixels', 1),
        'changed_point_binding': lambda p: p.__setitem__('points_sha256', '0' * 64),
        'changed_retry_provenance': lambda p: p['resource_retry_repeat'].__setitem__('prior_complete_records', 33),
        'false_adoption_claim': lambda p: p.__setitem__('serving_changed', True),
        'false_advancement_claim': lambda p: p['local_audit']['conditions']['native'].__setitem__('selected_arm', 'invented'),
    }
    for name, mutate in changes.items():
        changed = copy.deepcopy(packet)
        mutate(changed)
        try:
            verify(changed)
        except (ValueError, KeyError, TypeError):
            tests.append({'id': name, 'refused': True})
        else:
            raise ValueError('Tamper was accepted: ' + name)
    return {'schema': 'chipmunk.patch-drift-portable-tamper/8', 'passed': True, 'refused': len(tests), 'cases': tests}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review', 'tamper'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    packet = read('public-scalars')
    result = verify(packet) if args.mode == 'review' else tamper(packet)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)
