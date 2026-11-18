# SPDX-License-Identifier: MIT
"""Recount frozen context and token-score comparisons without models or source text."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from scripts import writing_native_context_study as study

SCHEMA = 'chipmunk.writing-native-context-review/1'
ROLES = ('ordinary', 'ordinary_edit', 'edited')


def identity(row):
    return row['group'], row['role'], row['variant'], row['sha256']


def digest(value):
    return study.sha((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def _compatible_geometry(row):
    points = row['scores']
    if not 0 <= len(row['truth']) <= 1 or len(row['truth']) != len(row['compact_truth']):
        return False
    return (len({value['sentence'] for value in points}) == len(points)
            and all(2 <= value['sentence'] < row['sentences'] - 1
                    and all(0 <= a < b for a, b in (value['range'], value['compact'])) for value in points)
            and (row['role'] == 'edited') == bool(row['truth']))


def summarize(panel, point):
    packet = panel['measurements']
    incumbent = panel['incumbent']
    baseline = {identity(row): row['result'] for row in incumbent['rows']}
    raw = packet['observed']
    if (len(baseline) != len(incumbent['rows']) or set(baseline) != {identity(row) for row in raw}
            or len(baseline) != len(raw) or any(not _compatible_geometry(row) for row in raw)):
        raise ValueError('Measurement geometry or case identities differ')
    rows = study.compare(packet, point)
    roles = Counter(row['role'] for row in rows)
    if roles.keys() - set(ROLES):
        raise ValueError('Unknown source role')
    totals = {name: Counter() for name in study.ARMS}
    genuine_positive_groups = {name: set() for name in study.ARMS}
    genuine_groups = {row['group'] for row in rows if row['role'] == 'ordinary'}
    for row in rows:
        if row['role'] in {'ordinary', 'ordinary_edit'}:
            for name, selected in row['selections'].items():
                if selected is not None:
                    genuine_positive_groups[name].add(row['group'])
    current = Counter()
    for row in rows:
        old = baseline[identity(row)]
        role = row['role']
        if not 0 <= old['matched_truth'] <= len(row['compact_truth']) or (old['matched_truth'] and not old['ranges']):
            raise ValueError('Invalid incumbent case reading')
        current.update({f'{role}_flags': bool(old['ranges']), 'locations': old['matched_truth']})
        for name, selected in row['selections'].items():
            matched = bool(row['matched'][name])
            counts = totals[name]
            counts.update({f'{role}_flags': selected is not None, f'{role}_assessed': len(row['scores']) >= study.POLICY['minimum_complete_centers'],
                           f'{role}_added_flags': selected is not None and not old['ranges'], 'locations': matched,
                           'new_locations': matched and not old['matched_truth'],
                           'new_locations_without_same_group_ordinary_alert': matched and not old['matched_truth']
                           and row['group'] in genuine_groups and row['group'] not in genuine_positive_groups[name],
                           'misplaced_edit_flags': role == 'edited' and selected is not None and not matched,
                           'location_is_top_before_cutoff': bool(row['top_before_point'][name]),
                           'reachable_known_edit': bool(row['reachable'][name])})
    return {'roles': dict(roles), 'cases': len(rows), 'source_groups': len({row['group'] for row in rows}),
            'views': packet['unique_inputs'], 'incumbent': dict(current),
            'candidates': {key: dict(value) for key, value in totals.items()}}


def compare_panels(panels, point):
    totals = {}
    for name, panel in panels.items():
        measured = panel['measurements']
        if measured['freeze']['point_sha256'] != digest(point):
            raise ValueError('Assessment responses do not bind to the calibration saved before their inference')
        primary = summarize(panel, point)
        totals[name] = {'primary': primary}
        excluded = {row['sha256'] for row in panel.get('adjudication_exclusions', [])}
        if excluded:
            if not excluded <= {case['sha256'] for case in measured['observed'] if case['role'] == 'edited'}:
                raise ValueError('Adjudications can exclude only labeled rewrite cases')
            filtered = {key: panel[key] | {rows: [case for case in panel[key][rows] if case['sha256'] not in excluded]}
                        for key, rows in (('measurements', 'observed'), ('incumbent', 'rows'))}
            totals[name]['excluding_adjudicated_rewrites'] = summarize(filtered, point)
    return totals


def replay(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    if data['schema'] != SCHEMA or data['point']['producer_sha256'] != study.sha(Path(study.__file__).read_bytes()):
        raise ValueError('Unknown or changed scoring method')
    point = data['point']
    if digest(data['calibration']) != point['measured_sha256'] or study.fit(data['calibration'], point['measured_sha256']) != point:
        raise ValueError('Saved operating points differ from the genuine calibration rows')
    result = compare_panels(data['assessment'], point)
    if result != data['summary']:
        raise ValueError('Recounted writing locations and control alerts differ')
    return {'calibration_groups': len(point['calibration_groups']),
            'complete_calibration_groups': len(point['maxima'][study.ARMS[0]]), 'assessment': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    print(json.dumps(replay(parser.parse_args().receipt), indent=2, sort_keys=True, allow_nan=False))
