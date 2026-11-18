# SPDX-License-Identifier: MIT
"""Recount the short-window Opus comparison without recordings or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

from scripts.review_audio_exact_foreground import PARENT_NAME
from scripts.review_audio_severe_overlap_voice_diagnostic import RATE, ROLES, STEP, WINDOW, readings
from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = 'chipmunk.audio-overlap-window-grid/1'
HOP = 4_000
ARMS = ('existing', 'dense_full', 'dense_intersection', 'closer_intersection')


def positions(values, point, starts, fixed):
    above = {tick: value for tick, value in zip(starts, values, strict=True) if value is not None and value > point}
    paired = [tick for tick in above if tick + STEP in above]
    fixed_windows = {(t, t + WINDOW) for tick in fixed if tick in paired for t in (tick, tick + STEP)}
    dense_windows = {(t, t + WINDOW) for tick in paired for t in (tick, tick + STEP)}
    narrow = [(tick + STEP, tick + WINDOW) for tick in paired]
    close = sorted(((-min(value, above[tick + HOP]), tick + HOP, tick + WINDOW)
                    for tick, value in above.items() if tick + HOP in above))
    distinct = []
    for _, start, stop in close:
        if all(stop <= a or start >= b for a, b in distinct):
            distinct.append((start, stop))
    return tuple(map(sorted, (fixed_windows, dense_windows, narrow, distinct)))


def overlap_counts(ranges, labels):
    return (sum(bool(matches(ranges, [label])) for label in labels), len(matches(ranges, labels)))


def totals_for(points):
    return {tier: {name: Counter() for name in ARMS} for tier in points}


def count_group(totals, points, values_by_role, starts, fixed, labels, *, interior=None):
    for tier, point in points.items():
        all_ranges = [positions(values, point, starts, fixed) for values in values_by_role]
        reference_clear = set()
        for arm, (edited, ordinary, background) in zip(ARMS, zip(*all_ranges, strict=True), strict=True):
            hits = set(matches(edited, labels))
            controls = sorted(set(ordinary) | set(background))
            clear = {found for found in hits if not matches(controls, [labels[found]])}
            if arm == 'existing':
                reference_clear = clear
            totals[tier][arm].update(located=len(hits), clear=len(clear), wrong_time=bool(edited) and not hits,
                ordinary_files=bool(ordinary), background_files=bool(background), new_clear=len(clear - reference_clear),
                groups_with_new=bool(clear - reference_clear), lost_clear=len(reference_clear - clear),
                added_ordinary=bool(ordinary) and not all_ranges[1][0], added_background=bool(background) and not all_ranges[2][0])
            if interior is not None:
                totals[tier][arm].update(interior_located=len(hits & interior), interior_clear=len(clear & interior))


def decision(totals, *, min_new, max_ordinary):
    return {tier: {name: dict(item) | {'passes': item['new_clear'] >= min_new and item['groups_with_new'] >= min_new
                                   and not item['lost_clear'] and item['added_ordinary'] <= max_ordinary and not item['added_background']}
                    for name, item in arms.items()} for tier, arms in totals.items()}


def review(receipt, parent_bytes):
    parent = json.loads(gzip.decompress(parent_bytes))
    require(receipt['schema'] == SCHEMA and receipt['parent_sha256'] == hashlib.sha256(parent_bytes).hexdigest()
            and receipt['clock'] == [RATE, WINDOW, HOP, STEP]
            and receipt['operating_points_hex'] == parent['raw_calibration_hex'],
            'The frozen comparison differs from its parent source or thresholds')
    points = {key: float.fromhex(point) for key, point in parent['raw_calibration_hex'].items()}
    totals = totals_for(points)
    reach = {name: Counter() for name in ARMS}
    measured = maximum = labels_total = 0
    for index, (native, published) in enumerate(zip(receipt['groups'], parent['matched_groups'], strict=True), 1):
        samples, labels, original = published
        fixed = [4 * RATE, 6 * RATE] if index <= 5 else [4 * RATE]
        starts = list(range(0, samples - WINDOW + 1, HOP))
        require(len(native) == len(original) == len(ROLES), 'Missing matched role')
        for name, geometry in zip(ARMS, (
                [(start + offset, start + offset + WINDOW) for start in fixed for offset in (0, STEP)],
                [(start, start + WINDOW) for start in starts],
                [(start + STEP, start + WINDOW) for start in starts if start + STEP <= starts[-1]],
                [(start + HOP, start + WINDOW) for start in starts if start + HOP <= starts[-1]],
                ), strict=True):
            reachable, ceiling = overlap_counts(geometry, labels)
            reach[name].update(individually_reachable=reachable, one_to_one_ceiling=ceiling)
        labels_total += len(labels)
        by_role = []
        for all_readings, (_, history, _) in zip(native, original, strict=True):
            values = tuple(readings(all_readings, starts))
            measured += sum(value is not None for value in values)
            maximum += len(starts)
            earlier = [point for t in fixed for point in (t, t + STEP)]
            for tick, (speech, old_hex), old_value in zip(earlier, history[0], readings(history[0], earlier), strict=True):
                observed = values[tick // HOP]
                require(speech == all_readings[tick // HOP][0]
                        and ((old_value is None and observed is None)
                             or (old_hex is not None and observed is not None and abs(old_value - observed) <= .00001
                                 and all((observed > point) == (old_value > point) for point in points.values()))),
                        'A previously measured window no longer matches')
            by_role.append(values)
        count_group(totals, points, by_role, starts, fixed, labels)
    require((len(receipt['groups']), labels_total, maximum) == (11, 75, 1401), 'Incomplete native evidence panel')
    result = decision(totals, min_new=3, max_ordinary=2)
    require(result == receipt['expected'] and measured == receipt['forwards'], 'Recounted observations differ')
    return {'labeled_edits': labels_total, 'model_calls': measured, 'scheduled_windows': maximum, 'results': result,
            'geometry': reach}


def review_transfer(receipt, points):
    totals = totals_for(points)
    fixed = [4 * RATE, 6 * RATE]
    label_count = interior_count = scored = scheduled = 0
    for length, labels, by_role in receipt['groups']:
        require(type(length) is int and max(fixed) + STEP + WINDOW <= length
                and len(by_role) == len(ROLES) and all(type(a) is int and type(b) is int and 0 <= a < b <= length for a, b in labels)
                and all(left[1] <= right[0] for left, right in pairwise(labels)), 'Invalid transfer grid or labels')
        interior = {i for i, (a, b) in enumerate(labels) if 0 < a < b < length}
        starts = list(range(0, length - WINDOW + 1, HOP))
        values = [tuple(readings(row, starts)) for row in by_role]
        label_count += len(labels)
        interior_count += len(interior)
        scheduled += len(starts) * len(ROLES)
        scored += sum(value is not None for row in values for value in row)
        count_group(totals, points, values, starts, fixed, labels, interior=interior)
    require((len(receipt['groups']), label_count, interior_count, scheduled) == (4, 30, 26, 492), 'Incomplete transfer panel')
    result = decision(totals, min_new=2, max_ordinary=1)
    require(result == receipt['expected'] and scored == receipt['forwards'], 'Recorded transfer outcome differs')
    return {'labeled_edits': label_count, 'interior_labels': interior_count, 'model_calls': scored,
            'scheduled_windows': scheduled, 'results': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('parent', type=Path, nargs='?')
    args = parser.parse_args()
    try:
        receipt = json.loads(gzip.decompress(args.receipt.read_bytes()))
        review_result = {'recurring': review(receipt, (args.parent or args.receipt.with_name(PARENT_NAME)).read_bytes())}
        if 'transfer' in receipt:
            review_result['transfer'] = review_transfer(receipt['transfer'], {tier: float.fromhex(point)
                                                        for tier, point in receipt['operating_points_hex'].items()})
    except (OSError, TypeError, ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    print(json.dumps(review_result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
