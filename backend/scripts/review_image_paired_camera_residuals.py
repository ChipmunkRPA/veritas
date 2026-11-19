# SPDX-License-Identifier: Apache-2.0
"""Recount frozen JPEG region overlaps and alarms from an anonymous receipt."""

import argparse
import json
from collections import Counter
from pathlib import Path

BANK_SOURCES = {'camera': 12, 'scene_text': 14}
ARMS = ('ordinary_whitened', 'paired_process_suppressed')
QUALITIES = (75, 90)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recount_panel(rows, source_count):
    require({row[0] for row in rows} == set(range(1, source_count + 1)), 'Unexpected source roster')
    totals = {str(q): {arm: Counter() for arm in ARMS} for q in QUALITIES}
    useful_sources = {(q, arm): set() for q in QUALITIES for arm in ARMS}
    false_sources = {(q, arm): set() for q in QUALITIES for arm in ARMS}
    for source, q, edit, old_hit, old_flag, truth, *records in rows:
        require(q in QUALITIES and all(type(value) is bool for value in (edit, old_hit, old_flag)), 'Invalid label')
        require(type(truth) is int and truth >= 0 and edit == bool(truth), 'Invalid mask extent')
        for arm, (area, overlap, above) in zip(ARMS, records, strict=True):
            require(type(area) is type(overlap) is int and 0 <= overlap <= min(area, truth), 'Invalid crop overlap')
            require(type(above) is bool, 'Invalid operating-point flag')
            useful = edit and area > 0 and 2 * overlap >= truth and 10 * overlap >= 3 * (area + truth - overlap)
            addition = useful and above and not old_hit
            false_alarm = not edit and above and not old_flag
            totals[str(q)][arm].update(edited=edit, ordinary=not edit, pre_point_correct=useful,
                                       existing_useful=edit and old_hit, existing_false=not edit and old_flag,
                                       additional_useful=addition, additional_false=false_alarm,
                                       misplaced=edit and above and not useful)
            if addition:
                useful_sources[q, arm].add(source)
            if false_alarm:
                false_sources[q, arm].add(source)
    for q in QUALITIES:
        for arm in ARMS:
            totals[str(q)][arm].update(additional_useful_sources=len(useful_sources[q, arm]),
                                       additional_false_sources=len(false_sources[q, arm]))
    return totals


def recount(receipt):
    require(receipt['schema'] == 'chipmunk.jpeg-paired-controls-localization/1', 'Unsupported receipt')
    require(receipt['fit_and_calibration_groups'] == [12, 12], 'Incorrect development roster')
    banks = receipt['panels']
    require(banks.keys() == BANK_SOURCES.keys(), 'Missing panel')
    require(sum(map(len, banks.values())) == receipt['native_deliveries_rehashed'] == 532, 'Incorrect delivery count')
    result = {bank: recount_panel(banks[bank], groups) for bank, groups in BANK_SOURCES.items()}
    require(result == receipt['expected'], 'Recount differs from recorded totals')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    path = parser.parse_args().receipt
    print(json.dumps(recount(json.loads(path.read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
