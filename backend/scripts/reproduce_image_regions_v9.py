"""Standard-library scalar/certificate/count replay, without source images."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

from scripts.reproduce_patch_drift_v8 import close, pixel_counts, require, summary

PUBLIC = Path(__file__).resolve().parents[2] / 'docs/evaluation-panels'
PREFIX = 'image-region-calibration-v9-'


def read(suffix):
    return json.loads((PUBLIC / (PREFIX + suffix + '.json')).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(packet):
    frozen, points = read('freeze'), read('points')
    original = json.loads((PUBLIC / 'image-v8-patch-drift-development-panel.json').read_text())
    previous = json.loads((PUBLIC / 'image-v8-patch-drift-public-scalars.json').read_text())
    old = {(r['role'], r['id'], r['condition']): r for r in previous['cases']}
    panel = {(r['role'], r['id'], r['condition']): r for r in original['items']}
    freeze_sha = digest(PUBLIC / (PREFIX + 'freeze.json'))
    require(frozen['arm'] == 'raw' and frozen['new_project_critical_levels_before_freeze'] == 0
        and frozen['new_neural_forwards'] == 0 and frozen['planned'] == {'calibration': 120, 'development': 192}, 'Changed prospective role/arm/count contract')
    require(frozen['minimum_additional_overlapping_edits'] == 4 and frozen['maximum_additional_genuine_highlights'] == 1
        and frozen['minimum_mean_iou_delta'] == frozen['minimum_family_overlap_delta'] == 0, 'Changed advancement rules')
    rule = frozen['component_policy']
    require(rule['minimum_region_fraction'] == .001 and rule['connectivity'] == 8 and rule['per_map_display_max_regions'] == 2,
        'Changed region geometry/inspection budget')
    require(frozen['prior_panel_sha256'] == digest(PUBLIC / 'image-v8-patch-drift-development-panel.json'), 'Changed original panel')
    require(packet['schema'] == 'chipmunk.image-region-calibration-scalars/9' and packet['freeze_sha256'] == freeze_sha
        and packet['points_sha256'] == digest(PUBLIC / (PREFIX + 'points.json')), 'Changed experiment/point binding')
    require(packet['serving_changed'] is False and packet['new_neural_forwards'] == 0, 'Unsupported deployment/neural claim')
    require(points['freeze_sha256'] == freeze_sha and points['passed'] and points['edited_inputs_used'] == 0, 'Not the original genuine-only fit')
    rows = packet['cases']
    require(len(rows) == len({(r['role'], r['id'], r['condition']) for r in rows}) == 312, 'Missing/duplicate outcomes')
    require({(r['role'], r['id'], r['condition']) for r in rows} == set(panel), 'Unplanned source/role')
    display_checks = {(r['id'], r['condition']): r for r in points['display_checks']}
    require(len(display_checks) == len(points['display_checks']) == 120, 'Missing genuine display audit')
    for row in rows:
        key = row['role'], row['id'], row['condition']
        source, prior = panel[key], old[key]
        require(row['status'] == 'ok' and row['freeze_sha256'] == freeze_sha, 'Incomplete/wrong-contract outcome')
        for name in ('source_id', 'family', 'label', 'input_sha256', 'input_rgb_sha256', 'geometry'):
            require(row[name] == source[name] == prior[name], 'Changed source/tensor binding: ' + name)
        require(row['arrays_sha256'] == prior['arrays_sha256'] == frozen['cached_archives_sha256'][row['role'] + '/' + prior['arrays']],
            'Changed native response archive')
        h, w = row['geometry'][:2]
        scale = min(1., 1024 / max(h, w))
        expected_shape = [max(1, round(h * scale)), max(1, round(w * scale))]
        require(row['screening_grid_shape'] == expected_shape, 'Changed actual screening grid')
        minimum = max(1, math.ceil(.001 * math.prod(expected_shape)))
        response, boundary = row['response'], row['independent_boundary']
        require(boundary['passed'] is True, 'Failed plateau oracle')
        if response is None:
            require(boundary['available'] is False and 0 <= boundary['largest_reliable_component'] < minimum, 'Unsupported unavailability')
            critical = None
        else:
            critical = response['critical_level']
            require(type(critical) is float and math.isfinite(critical)
                and struct.unpack('<f', struct.pack('<f', critical))[0] == critical, 'Critical level is not an exact native float32 value')
            require(boundary['available'] is True and response['minimum_component_cells'] == boundary['minimum_component_cells'] == minimum
                and 0 <= boundary['strict_largest_component'] < minimum <= boundary['nonstrict_largest_component'] <= math.prod(expected_shape),
                'Plateau does not certify maximum feasible connected region')
        expected_flag = critical is not None and critical > points['points'][row['condition']]
        target = display_checks[row['id'], row['condition']]['metrics'] if row['role'] == 'calibration' else row['candidate']
        pixel_counts(target)
        require(target['pixels'] == h * w and (target['reference_positive_pixels'] > 0) == bool(row['label']), 'Changed source/reference pixel denominator')
        require((target['predicted_positive_pixels'] > 0) == expected_flag, 'Region statistic and actual display disagree')
        if row['role'] == 'calibration':
            require(row['label'] == 0, 'Edited example used in calibration')
        else:
            require(row['baseline'] == source['baseline'] and row['historical_v8_raw'] == prior['arms']['raw']['metrics'], 'Changed current or historical comparator')
            pixel_counts(row['baseline'])
    for condition in ('native', 'jpeg75'):
        chosen = [r for r in rows if r['role'] == 'calibration' and r['condition'] == condition]
        require(len(chosen) == 60, 'Missing genuine denominator')
        levels = sorted((r['response']['critical_level'] for r in chosen if r['response'] is not None), reverse=True)
        require(len(levels) >= 48 and levels[1] > 0 and points['points'][condition] == levels[1], 'Genuine-only point refit differs')
        require(sum(c > levels[1] for c in levels) == points['genuine_displayed_regions'][condition] <= 1, 'Genuine display budget differs')
    audit = packet['local_audit']
    require(audit['freeze_sha256'] == freeze_sha and audit['points_sha256'] == packet['points_sha256']
        and audit['verified_calibration_inputs'] == 120 and audit['verified_development_inputs'] == 192
        and audit['independent_plateau_and_pixel_replays'] == 312, 'Incomplete local geometry/source audit')
    require(set(audit['profiles']) == {'calibration', 'development'} and all(p['child_exit_code'] == 0
        and p['guard'] is None for p in audit['profiles'].values()), 'Incomplete/guarded original computation')
    conditions = {}
    for condition in ('native', 'jpeg75'):
        subset = [r for r in rows if r['role'] == 'development' and r['condition'] == condition]
        require(len(subset) == 96 and sum(r['label'] for r in subset) == 48, 'Missing edited/genuine development denominator')
        translated = [{**r, 'arms': {'region': {'metrics': r['candidate']}}} for r in subset]
        baseline, candidate = summary(translated, 'baseline'), summary(translated, 'region')
        b, c = baseline['overall'], candidate['overall']
        requirements = {'complete': True, 'independent_checks': True,
            'at_least_four_more_overlapping_edits': c['overlapping_edits'] >= b['overlapping_edits'] + 4,
            'mean_edit_iou_not_lower': c['mean_edit_iou'] >= b['mean_edit_iou'],
            'genuine_highlights_within_one': c['genuine_highlights'] <= b['genuine_highlights'] + 1,
            'no_family_overlap_loss': all(candidate['families'][f]['overlapping_edits'] >= baseline['families'][f]['overlapping_edits']
                for f in baseline['families'])}
        conditions[condition] = {'baseline': baseline, 'candidate': candidate, 'requirements': requirements,
            'passed': all(requirements.values())}
    require(close(conditions, audit['conditions']), 'Aggregate/qualification decision differs')
    return {'schema': 'chipmunk.image-regions-portable-review/9', 'passed': True, 'case_bindings': 312,
        'region_boundary_certificates': 312, 'genuine_points_refitted': 2, 'genuine_display_audits': 120,
        'development_metric_sets_recomputed': 384, 'conditions': conditions,
        'scope': 'Scalar/certificate/count consistency; no private-grid reconstruction, neural inference or outside human review.'}


def tamper(packet):
    changes = {
        'missing_case': lambda p: p['cases'].pop(),
        'duplicate_case': lambda p: p['cases'].__setitem__(1, copy.deepcopy(p['cases'][0])),
        'changed_source': lambda p: p['cases'][0].__setitem__('source_id', 'changed'),
        'changed_role': lambda p: p['cases'][0].__setitem__('role', 'development'),
        'changed_label': lambda p: p['cases'][0].__setitem__('label', 1),
        'changed_response_archive': lambda p: p['cases'][0].__setitem__('arrays_sha256', '0' * 64),
        'changed_grid_geometry': lambda p: p['cases'][0].__setitem__('screening_grid_shape', [1, 1]),
        'false_region_certificate': lambda p: p['cases'][0]['independent_boundary'].__setitem__('strict_largest_component', 10000000),
        'false_unavailability': lambda p: p['cases'][0].__setitem__('response', None),
        'changed_point_binding': lambda p: p.__setitem__('points_sha256', '0' * 64),
        'changed_pixel_count': lambda p: next(r for r in p['cases'] if r['role'] == 'development')['candidate'].__setitem__('pixels', 1),
        'false_adoption': lambda p: p.__setitem__('serving_changed', True),
        'false_fresh_inference': lambda p: p.__setitem__('new_neural_forwards', 312),
        'false_qualification': lambda p: p['local_audit']['conditions']['native'].__setitem__('passed', not p['local_audit']['conditions']['native']['passed']),
    }
    refused = []
    for name, mutate in changes.items():
        changed = copy.deepcopy(packet)
        mutate(changed)
        try:
            verify(changed)
        except (ValueError, KeyError, TypeError, OverflowError):
            refused.append(name)
        else:
            raise ValueError('Accepted altered packet: ' + name)
    return {'schema': 'chipmunk.image-regions-portable-tamper/9', 'passed': True, 'refused': len(refused), 'cases': refused}


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
