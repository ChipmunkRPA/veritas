"""Standard-library review of public final ranges and development decision only.

No raw images, facial feature vectors, model weights or inference dependencies.
Native/vector/selection claims remain separately documented engineering evidence.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def intersection_over_union(predicted, known):
    if predicted is None or known is None:
        return 0.
    total = max(predicted[1], known[1]) - min(predicted[0], known[0])
    return max(0., min(predicted[1], known[1]) - max(predicted[0], known[0])) / total if total > 0 else 0.


def metric(rows, arm):
    positive, negative = [r for r in rows if r['label'] == 1], [r for r in rows if r['label'] == 0]
    controls = [r for r in negative if r['family'] != 'original_videos']
    overlaps = [intersection_over_union(r[arm + '_range_s'], r['known_edit_s']) for r in positive]
    return {'planned': len(rows), 'source_groups': len({r['group'] for r in rows}), 'edits': len(positive),
        'genuine': len(negative), 'localized_edits': sum(x >= .3 for x in overlaps),
        'missed_localizations': sum(x < .3 for x in overlaps),
        'genuine_review_flags': sum(r[arm + '_range_s'] is not None for r in negative),
        'ordinary_edit_flags': sum(r[arm + '_range_s'] is not None for r in controls),
        'mean_edit_iou': sum(overlaps) / len(overlaps) if overlaps else None,
        'status_counts': dict(Counter(r['candidate_status'] for r in rows)),
        'failed_cases': sum(r['candidate_status'] == 'failed' for r in rows)}


def compare(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError('Aggregate field set changed')
    for key, value in actual.items():
        if isinstance(value, float):
            if abs(value - expected[key]) > 1e-12:
                raise ValueError('Independent float64 mean differs')
        elif value != expected[key]:
            raise ValueError('Independent range/status accounting differs')


def review(packet, panel):
    items = {r['id']: r for r in panel['items'] if r['role'] == 'development'}
    cases, summary = packet['cases'], packet['summary']
    if len(cases) != 120 or len({r['id'] for r in cases}) != 120 or set(items) != {r['id'] for r in cases}:
        raise ValueError('Exact120 source-bound development cases required')
    for row in cases:
        item = items[row['id']]
        if any(row[k] != item[k] for k in ('id', 'group', 'family', 'label', 'role')) or row['known_edit_s'] != item['edit_s']:
            raise ValueError('Frozen source labels, groups or intervals changed')
        for arm in ('current', 'candidate'):
            span = row[arm + '_range_s']
            if span is not None and (len(span) != 2 or not 0 <= span[0] <= span[1] <= item['frame_count'] / item['fps'] + 1e-6):
                raise ValueError('Invalid predicted native-time range')
            measured = intersection_over_union(span, row['known_edit_s'])
            if measured != row[arm + '_iou'] or (measured >= .3) != row[arm + '_localized']:
                raise ValueError('Per-case overlap or localization decision changed')
    current, candidate = metric(cases, 'current'), metric(cases, 'candidate')
    compare(current, summary['current'])
    compare(candidate, summary['candidate'])
    checked_fields = len(current) + len(candidate)
    for family, values in summary['families'].items():
        for arm in ('current', 'candidate'):
            measured = metric([r for r in cases if r['family'] == family], arm)
            compare(measured, values[arm])
            checked_fields += len(measured)
    changes = {'new_correct_locations': sum(r['label'] and r['candidate_localized'] and not r['current_localized'] for r in cases),
        'lost_correct_locations': sum(r['label'] and r['current_localized'] and not r['candidate_localized'] for r in cases),
        'new_genuine_flags': sum(not r['label'] and r['current_range_s'] is None and r['candidate_range_s'] is not None for r in cases),
        'removed_genuine_flags': sum(not r['label'] and r['candidate_range_s'] is None and r['current_range_s'] is not None for r in cases)}
    if changes != summary['case_changes']:
        raise ValueError('Paired case changes differ')
    checks = packet['engineering_counts']
    gates = {
        'all_source_cases_accounted': checks['calibration_cases'] == 60 and len(cases) == 120,
        'no_candidate_failure': checks['failed_cases'] == 0,
        'all_original_runs_complete_without_guard': all(p['child_exit_code'] == 0 and p['guard'] is None for p in packet['profiles'].values()),
        'all_current_consecutive_replays_match': checks['current_parity_cases'] == 180,
        'all_actual_native_alignments_inputs_features_match': checks['native_parity_cases'] == 180,
        'more_correct_edit_localizations': candidate['localized_edits'] > current['localized_edits'],
        'genuine_total_flags_at_most_current_plus_one': candidate['genuine_review_flags'] <= current['genuine_review_flags'] + 1,
        'ordinary_edit_flags_at_most_current_plus_one': candidate['ordinary_edit_flags'] <= current['ordinary_edit_flags'] + 1,
        'mean_edit_iou_not_lower': candidate['mean_edit_iou'] >= current['mean_edit_iou'],
        'no_swap_family_localization_loss': all(summary['families'][family]['candidate']['localized_edits'] >=
            summary['families'][family]['current']['localized_edits'] for family in ('inswapper', 'uniface'))}
    if gates != summary['requirements'] or all(gates.values()) != summary['advance_to_separately_frozen_fresh_confirmation']:
        raise ValueError('Frozen advancement decision differs')
    return {'passed': True, 'cases': 120, 'aggregate_fields_verified': checked_fields,
        'per_case_iou_and_location_flags_verified': 240, 'gates_verified': len(gates),
        'native_or_biometric_inference_repeated': False, 'private_vectors_or_images_exported': False,
        'scope': 'Separate standard-library final-range/source-label/IoU/count/decision arithmetic. Engineering totals are recorded claims, not native model replay; bootstrap intervals and operating-point selection are not recomputed here.'}


def tamper(packet, panel):
    review(packet, panel)
    results = []
    for kind in ('missing', 'duplicate', 'unknown_id', 'label', 'truth', 'range', 'overlap', 'status', 'guard', 'advance'):
        altered = copy.deepcopy(packet)
        row = altered['cases'][0]
        if kind == 'missing':
            altered['cases'].pop()
        elif kind == 'duplicate':
            altered['cases'][1] = copy.deepcopy(row)
        elif kind == 'unknown_id':
            row['id'] += '-unexpected'
        elif kind == 'label':
            row['label'] = 1 - row['label']
        elif kind == 'truth':
            row['known_edit_s'] = [-1, 100]
        elif kind == 'range':
            row['candidate_range_s'] = [-1, 100]
        elif kind == 'overlap':
            row['candidate_iou'] += .1
        elif kind == 'status':
            row['candidate_status'] = 'failed'
        elif kind == 'guard':
            altered['profiles']['development']['guard'] = 'incomplete'
        elif kind == 'advance':
            altered['summary']['advance_to_separately_frozen_fresh_confirmation'] = not altered['summary']['advance_to_separately_frozen_fresh_confirmation']
        try:
            review(altered, panel)
        except (ValueError, KeyError, TypeError):
            results.append({'mutation': kind, 'rejected': True})
        else:
            results.append({'mutation': kind, 'rejected': False})
    return {'passed': all(r['rejected'] for r in results), 'mutations': results,
        'scope': 'Actual published range packet deliberately changed; no raw media or face vectors.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review', 'tamper'))
    parser.add_argument('packet', type=Path)
    parser.add_argument('panel', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    value = json.loads(args.packet.read_text())
    if hashlib.sha256(args.panel.read_bytes()).hexdigest() != value['public_panel_sha256']:
        raise ValueError('Exact original public source panel bytes required')
    result = (review if args.mode == 'review' else tamper)(value, json.loads(args.panel.read_text()))
    result['packet_sha256'] = hashlib.sha256(args.packet.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise ValueError('Public range comparison failed')
