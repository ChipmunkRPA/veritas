"""Standard-library audit of published fresh source IDs, ranges and decision.

This is not native inference or verification that the underlying recordings are
unseen people. Engineering replay counts remain explicitly recorded claims.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from scripts.reproduce_sface_result_v16 import compare, intersection_over_union, metric


def review(packet, panel, plan):
    cases, summary = packet['cases'], packet['summary']
    items = {r['id']: r for r in panel['items']}
    sources = plan['items']
    earlier_groups = {r['group'] for r in plan['prior_source_exclusions']}
    earlier_donors = {r['donor_filename'] for r in plan['prior_source_exclusions'] if r['donor_filename']}
    if (len(sources) != 90 or len({r['group'] for r in sources}) != 30
            or {r['group'] for r in sources} & earlier_groups
            or {r['donor_filename'] for r in sources if r['donor_filename']} & earlier_donors):
        raise ValueError('Source-video IDs or donor filenames overlap previous research')
    if (len(cases) != 120 or len(items) != 120 or len({r['id'] for r in cases}) != 120
            or set(items) != {r['id'] for r in cases}
            or {r['group'] for r in cases} != {r['group'] for r in sources}):
        raise ValueError('Exact120 source-bound fresh cases required')
    for row in cases:
        item = items[row['id']]
        if (any(row[k] != item[k] for k in ('id', 'group', 'label', 'family'))
                or row['known_edit_s'] != item['edit_s'] or row['role'] != 'confirmation'):
            raise ValueError('Frozen source labels or temporal truth changed')
        if row['candidate_status'] not in {'ok', 'not_applicable', 'failed'}:
            raise ValueError('Unknown candidate status')
        for arm in ('current', 'candidate'):
            span = row[arm + '_range_s']
            if span is not None and (len(span) != 2 or not all(math.isfinite(v) for v in span)
                    or not 0 <= span[0] <= span[1] <= item['frame_count'] / item['fps'] + 1e-6):
                raise ValueError('Invalid native-time predicted range')
            measured = intersection_over_union(span, row['known_edit_s'])
            if measured != row[arm + '_iou'] or (measured >= .3) != row[arm + '_localized']:
                raise ValueError('Per-case overlap or localization claim changed')
        if row['candidate_status'] != 'ok' and row['candidate_range_s'] != row['current_range_s']:
            raise ValueError('Unavailable/failed candidate must preserve the current range')
    current, candidate = metric(cases, 'current'), metric(cases, 'candidate')
    compare(current, summary['current'])
    compare(candidate, summary['candidate'])
    count = len(current) + len(candidate)
    if set(summary['families']) != {r['family'] for r in cases}:
        raise ValueError('Incomplete family accounting')
    for family, values in summary['families'].items():
        for arm in ('current', 'candidate'):
            measured = metric([r for r in cases if r['family'] == family], arm)
            compare(measured, values[arm])
            count += len(measured)
    changes = {
        'new_correct_locations': sum(r['label'] and r['candidate_localized'] and not r['current_localized'] for r in cases),
        'lost_correct_locations': sum(r['label'] and r['current_localized'] and not r['candidate_localized'] for r in cases),
        'new_genuine_flags': sum(not r['label'] and r['current_range_s'] is None and r['candidate_range_s'] is not None for r in cases),
        'cleared_genuine_flags': sum(not r['label'] and r['candidate_range_s'] is None and r['current_range_s'] is not None for r in cases)}
    usage = {'edited_candidate_ranges': sum(r['label'] and r['candidate_used'] for r in cases),
        'genuine_candidate_ranges': sum(not r['label'] and r['candidate_used'] for r in cases),
        'edited_correct_fallbacks': sum(r['label'] and not r['candidate_used'] and r['candidate_localized'] for r in cases),
        'genuine_shared_fallback_flags': sum(not r['label'] and not r['candidate_used'] and r['candidate_range_s'] is not None for r in cases)}
    if changes != packet['paired_changes'] or usage != packet['candidate_window_usage']:
        raise ValueError('Paired changes or actual candidate/fallback coverage differs')
    engineering = packet['engineering_counts']
    requirements = {
        'all120_cases30_source_groups_accounted': len(cases) == 120 and len({r['group'] for r in cases}) == 30,
        'no_candidate_failure': candidate['failed_cases'] == 0,
        'complete_without_guard': all(p['child_exit_code'] == 0 and p['guard'] is None for p in summary['profiles'].values()),
        'all_current_checks_freshly_recomputed': engineering['fresh_current_recomputed_cases'] == 120,
        'all_native_alignments_inputs_features_match': engineering['native_alignment_passed_cases'] == 120,
        'more_correct_edit_localizations': candidate['localized_edits'] > current['localized_edits'],
        'genuine_total_flags_at_most_current_plus_one': candidate['genuine_review_flags'] <= current['genuine_review_flags'] + 1,
        'ordinary_edit_flags_at_most_current_plus_one': candidate['ordinary_edit_flags'] <= current['ordinary_edit_flags'] + 1,
        'mean_edit_iou_not_lower': candidate['mean_edit_iou'] >= current['mean_edit_iou'],
        'no_swap_family_localization_loss': all(summary['families'][f]['candidate']['localized_edits'] >=
            summary['families'][f]['current']['localized_edits'] for f in ('inswapper', 'uniface'))}
    if (requirements != summary['requirements'] or all(requirements.values()) !=
            summary['qualifies_for_local_integration_testing'] or summary['point']['value'] != .11243900750879454):
        raise ValueError('Fixed method/point/decision changed')
    return {'passed': True, 'source_groups': 30, 'source_metadata_records': 90, 'cases': len(cases),
        'previous_source_groups_excluded': len(earlier_groups), 'previous_donor_files_excluded': len(earlier_donors),
        'aggregate_fields_verified': count, 'per_case_iou_and_location_flags_verified': 240,
        'additional_descriptive_paired_and_usage_counts_verified': len(changes) + len(usage),
        'gates_verified': len(requirements), 'native_or_biometric_inference_repeated': False,
        'private_vectors_or_images_exported': False,
        'scope': 'Separate source-ID/donor exclusions, final ranges, IoU, counts and fixed-decision arithmetic. Native/fresh-recomputation totals remain recorded engineering claims; no biometric inference, bootstrap or unseen-person assertion.'}


def tamper(packet, panel, plan):
    review(packet, panel, plan)
    results = []
    kinds = ('missing', 'duplicate', 'label', 'truth', 'range', 'overlap', 'status', 'guard', 'advance',
             'point', 'prior_source', 'prior_donor')
    for kind in kinds:
        changed, changed_plan = copy.deepcopy(packet), copy.deepcopy(plan)
        row = changed['cases'][0]
        if kind == 'missing':
            changed['cases'].pop()
        elif kind == 'duplicate':
            changed['cases'][1] = copy.deepcopy(row)
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
            changed['summary']['profiles']['candidate']['guard'] = 'incomplete'
        elif kind == 'advance':
            changed['summary']['qualifies_for_local_integration_testing'] = not changed['summary']['qualifies_for_local_integration_testing']
        elif kind == 'point':
            changed['summary']['point']['value'] += .01
        elif kind == 'prior_source':
            changed_plan['items'][0]['group'] = changed_plan['prior_source_exclusions'][0]['group']
        elif kind == 'prior_donor':
            donor = next(r['donor_filename'] for r in changed_plan['prior_source_exclusions'] if r['donor_filename'])
            changed_plan['items'][1]['donor_filename'] = donor
        try:
            review(changed, panel, changed_plan)
        except (ValueError, KeyError, TypeError):
            results.append({'mutation': kind, 'rejected': True})
        else:
            results.append({'mutation': kind, 'rejected': False})
    return {'passed': all(r['rejected'] for r in results), 'mutations': results,
        'scope': 'Actual final-range/source plan deliberately altered; no raw imagery or face representations.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review', 'tamper'))
    parser.add_argument('packet', type=Path)
    parser.add_argument('panel', type=Path)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    for path, key in ((args.panel, 'public_panel_sha256'), (args.plan, 'public_source_plan_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != packet[key]:
            raise ValueError('Published source/preparation bytes changed')
    result = (review if args.mode == 'review' else tamper)(packet, json.loads(args.panel.read_text()), json.loads(args.plan.read_text()))
    result['packet_sha256'] = hashlib.sha256(args.packet.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise ValueError('Public confirmation arithmetic failed')
