"""Standard-library tree/point/ranking/outcome review; optional shared-learner refit.

No native video decode, neural forward, source-mask audit or outside human review.
Original source/feature arithmetic is checked by the existing portable reviewer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.review_mouth_context_v10 import key, overlap
from scripts.review_mouth_temporal_v11 import review as review_source

SOURCE_SHA = 'ac7555bf05b0b4c013f93ce07144880a67886916d84c22748fead5cf1e791566'
FREEZE_SHA = 'e0f60e10aac8cce8e4aec5a2df04da5f11d508f951aaf25dfeb39a98cbc0d6cf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin(features, head):
    require(len(features) == 49 and all(type(v) in (float, int) and math.isfinite(v) for v in features), 'Complete finite feature window required')
    result = head['base_margin']
    for tree in head['trees']:
        require(1 <= len(tree) <= 13, 'Complete bounded tree required')
        index, visited = 0, set()
        while True:
            require(type(index) is int and 0 <= index < len(tree) and index not in visited, 'Malformed tree routing')
            visited.add(index)
            node = tree[index]
            require(node['is_leaf'] in (0, 1) and math.isfinite(node['value']), 'Finite explicit node required')
            if node['is_leaf']:
                result += node['value']
                break
            feature, threshold = node['feature_idx'], node['num_threshold']
            require(type(feature) is int and 0 <= feature < 49 and math.isfinite(threshold), 'Invalid numeric split')
            index = node['left'] if features[feature] <= threshold else node['right']
    require(math.isfinite(result), 'Nonfinite raw tree margin')
    return result


def close(a, b):
    require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=0, abs_tol=1e-10), 'Tree/point arithmetic differs')


def summary(rows):
    ai, ordinary = [r for r in rows if r['label']], [r for r in rows if not r['label']]
    return {'planned': len(rows), 'ai_edits': len(ai), 'non_ai_controls': len(ordinary),
        'assessed': sum(r['assessed'] for r in rows), 'localized_ai': sum(r['localized'] for r in ai),
        'ai_without_flag': sum(not r['flagged'] for r in ai), 'non_ai_flags': sum(r['flagged'] for r in ordinary),
        'mean_ai_temporal_iou': statistics.mean(r['temporal_iou'] for r in ai),
        'mean_ai_spatial_iou': statistics.mean(r['spatial_iou'] for r in ai)}


def review(source_path, freeze_path, packet_path, output, refit=False):
    require(digest(source_path) == SOURCE_SHA and digest(freeze_path) == FREEZE_SHA, 'Original source/freeze changed')
    source = json.loads(gzip.decompress(source_path.read_bytes()))
    review_source(source)
    freeze = json.loads(freeze_path.read_text())
    packet = json.loads(gzip.decompress(packet_path.read_bytes()))
    require(packet['source_packet_sha256'] == SOURCE_SHA and packet['public_freeze_sha256'] == FREEZE_SHA, 'Wrong complete packet lineage')
    head, result = packet['head'], packet['summary']
    require(head['spec'] == freeze['lock']['rules']['method'] and len(head['trees']) == 100
            and head['feature_names'] == source['head']['feature_names']
            and head['fit_groups'] == source['head']['fit_groups'] and head['fit_windows'] == 3560
            and head['deployment_eligible'] is False, 'Original complete fitted tree/order/roles required')
    head_sha = hashlib.sha256((json.dumps(head, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    require(result['head_sha256'] == head_sha and result['lock_sha256'] == head['lock_sha256'] == freeze['private_lock_sha256']
            and result['packet_sha256'] == head['packet_sha256'] == SOURCE_SHA, 'Fitting-before-development lineage differs')
    controls = source['panels']['calibration']['cases']
    control_max = max(margin(w['features'], head) for row in controls for w in row['windows'])
    close(head['ordinary_max_margin'], control_max)
    require(head['review_point'] == math.nextafter(max(0., control_max), math.inf), 'Frozen ordinary point rule differs')
    development = source['panels']['development']['cases']
    scored, temporal = result['scored_windows'], result['cases']['temporal']
    require([r['id'] for r in scored] == [r['id'] for r in development] == [r['id'] for r in temporal], 'Complete ordered88 outcomes required')
    for row, recorded, outcome in zip(development, scored, temporal, strict=True):
        windows = sorted([{**w, 'response': margin(w['features'], head)} for w in row['windows']], key=lambda w: (-w['response'], w['start_s'], w['track_id']))
        require(len(recorded['windows']) == len(windows) and [key(w) for w in recorded['windows']] == [key(w) for w in windows], 'Every ranked window required')
        for actual, expected in zip(recorded['windows'], windows, strict=True):
            require({k: v for k, v in actual.items() if k != 'response'} == {k: v for k, v in expected.items() if k != 'response'}, 'Native region/features changed')
            close(actual['response'], expected['response'])
        flagged = windows[0]['response'] >= head['review_point']
        time_iou = overlap(windows[0] if flagged else None, row['edit_time_range_s'])
        require(all(outcome[k] == row[k] for k in ('id', 'group', 'condition', 'label'))
                and outcome['assessed'] is True and outcome['flagged'] == flagged
                and outcome['localized'] == bool(row['label'] and time_iou >= .2), 'Raw response/time/outcome differs')
        close(outcome['temporal_iou'], time_iou)
        require(type(outcome['spatial_iou']) in (int, float) and math.isfinite(outcome['spatial_iou']) and 0 <= outcome['spatial_iou'] <= 1,
                'Finite stored source-mask intersection required')
    reference = source['absolute_reference']
    combined = result['cases']['combined']
    require(len(combined) == len(result['selected_signals']) == 88, 'Complete combined outcomes required')
    for old, a, b, signal, outcome in zip(reference['cases'], reference['outcomes'], temporal, result['selected_signals'], combined, strict=True):
        old_flag = bool(old['probe_windows'] and old['probe_windows'][0]['response'] >= reference['review_point'])
        expected_signal = 'absolute' if old_flag else 'relative' if b['flagged'] else None
        require(signal == {'id': old['id'], 'signal': expected_signal} and outcome == (a if old_flag else b), 'Absolute-first preservation differs')
    for field, rows in [('baseline', reference['outcomes']), ('temporal_only', temporal), ('combined', combined)]:
        calculated = summary(rows)
        for name, value in calculated.items():
            if type(value) is float:
                close(result[field][name], value)
            else:
                require(result[field][name] == value, 'Complete outcome counts differ')
    a, b = reference['outcomes'], combined
    ordinary = {r['condition'] for r in a if not r['label']}
    first = [r for r, s in zip(b, development, strict=True) if s['family'] == 'liveportrait']
    second = [r for r, s in zip(b, development, strict=True) if s['family'] == 'musetalk']
    stats_a, stats_b = summary(a), summary(b)
    gates = {'six_net_additional_locations': stats_b['localized_ai'] >= stats_a['localized_ai'] + 6,
        'first_family_six_locations': sum(r['localized'] for r in first) >= 6,
        'both_first_family_conditions': all(sum(r['localized'] for r in first if r['condition'] == c) >= 2 for c in ('ai-redraw', 'ai-retime')),
        'retained_second_family': sum(r['localized'] for r in second) >= 13,
        'at_most_one_lost_location': sum(x['localized'] and not y['localized'] for x, y in zip(a, b, strict=True)) <= 1,
        'total_ordinary_flags': stats_b['non_ai_flags'] <= stats_a['non_ai_flags'] + 1,
        'condition_ordinary_flags': all(sum(r['flagged'] for r in b if r['condition'] == c) <= sum(r['flagged'] for r in a if r['condition'] == c) + 1 for c in ordinary),
        'spatial_iou_gain': stats_b['mean_ai_spatial_iou'] >= stats_a['mean_ai_spatial_iou'] + .02,
        'retained_temporal_iou': stats_b['mean_ai_temporal_iou'] >= stats_a['mean_ai_temporal_iou'],
        'retained_coverage': all(not x['assessed'] or y['assessed'] for x, y in zip(a, b, strict=True)),
        'complete_replays': all(r['native_input_replay'] and all(h['baseline_replay_parity'] and h['prior_native_and_margin_parity'] for h in r['heads'].values()) for r in development),
        'complete_export_correspondence': result['native_estimator_export_windows'] == 6373 and result['max_native_export_margin_difference'] <= 1e-10}
    require(result['gates'] == gates and result['passes'] is all(gates.values()) and result['deployment_eligible'] is False,
            'Unchanged requirements/admission differ')
    if refit:
        from scripts.mouth_temporal_v12 import fit
        _, rebuilt = fit(source['fitting_windows'])
        require(all(head[k] == v for k, v in rebuilt.items()), 'Complete fitting-only JSON tree refit differs')
    receipt = {'schema': 'chipmunk.mouth-histogram-public-review/12', 'passed': True, 'packet_sha256': digest(packet_path),
        'source_packet_sha256': SOURCE_SHA, 'public_freeze_sha256': FREEZE_SHA, 'reviewer_sha256': digest(Path(__file__)),
        'development_cases': 88, 'trees': 100, 'point_windows': 492, 'development_windows': sum(len(r['windows']) for r in development),
        'fitting_only_exact_refit_performed': refit, 'gates': gates, 'candidate_passes': all(gates.values()),
        'outside_reviewers': 0, 'serving_admitted': False,
        'scope': 'Separate scalar raw-tree traversal, point rule, complete ranking/time/outcomes/counts/requirements; prior source arithmetic shared. Stored spatial IoU and estimator/export audit assertions bound, not independently rerun. Optional exact refit shares declared learner. No native decode/neural forward, fresh source test or outside human review.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'exact_refit': refit, 'candidate_passes': all(gates.values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'freeze', 'packet', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    review(args.source, args.freeze, args.packet, args.output, args.refit)
