# SPDX-License-Identifier: Apache-2.0
"""Recount privacy-minimized frozen audio decision states; no media or models.

This checks decision-state accounting and a frozen gate. It does not independently
regenerate scores, sample locations, source grouping, or speech admission.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 'chipmunk.audio-phase-partition-decisions/1'
DEVELOPMENT = 'd2fa678621bb2e9206915c07274821742143b971cb7815ccb0d474cda8f509c2'
ARMS = ('incumbent', 'mixture', 'projected', 'contrast', 'add')
CONDITIONS = ('clean', 'traffic', 'competing_speech')
FAMILIES = ('MeloTTS', 'ChatterboxTTS')
STATES = ('unassessed', 'unflagged', 'flagged', 'located', 'flagged_elsewhere')
TOP = {'schema', 'commitments', 'operating_points', 'case_states', 'reader_outcomes', 'paired_control_outcomes', 'outcome', 'state_sha256', 'parity'}


def require(value, message):
    if not value:
        raise ValueError(message)


def serial(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def number(value, low, high):
    return type(value) is int and low <= value <= high


def state_part(payload):
    return {key: payload[key] for key in ('case_states', 'reader_outcomes', 'paired_control_outcomes')}


def eligible(values):
    severe = values['competing_speech']
    decision = (severe['by_arm']['add']['located'] - severe['by_arm']['incumbent']['located'] >= 3
                and severe['added_readers'] >= 3 and severe['families_with_gain'] == 2
                and severe['same_time_control_collisions'] == 0)
    for entry in values.values():
        current, add = entry['by_arm']['incumbent'], entry['by_arm']['add']
        primary = add['edited'] + add['ordinary']
        missed = add['edited_unassessed'] + add['ordinary_unassessed']
        decision &= (5 * missed <= primary and entry['new_ordinary_groups'] <= 2
                     and add.get('background_flags', 0) <= current.get('background_flags', 0))
    return bool(decision)


def recount(payload, development):
    require(set(payload) == TOP and payload['schema'] == SCHEMA, 'Unexpected public receipt fields')
    commitments = payload['commitments']
    required_hashes = {'development', 'private_freeze', 'private_measurements', 'private_result',
                       'frozen_comparator', 'frozen_pair_comparator', 'public_reviewer'}
    require(set(commitments) == required_hashes and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
            for v in commitments.values()), 'Unexpected commitment')
    require(commitments['development'] == DEVELOPMENT == hashlib.sha256(development.read_bytes()).hexdigest(),
            'Use the exact previously committed development receipt')
    # The exact, committed development artifact already contains verified per-condition points.
    # Its full byte hash pins both the scalar readings and those points.
    prior = read(development)
    expected_points = {arm: max(x[arm]['strict'] for x in prior['points'].values())
                       for arm in ('mixture', 'projected', 'contrast')}
    require(payload['operating_points'] == expected_points, 'Operating points differ from frozen development')
    require(payload['parity'] == {'private_cases_checked': 408, 'exact_interval_predictions_checked': 2040,
                                  'decision_states_checked': 2040, 'paired_control_comparisons_checked': 528,
                                  'condition_reader_summaries_checked': 24}, 'Private parity scope differs')
    require(hashlib.sha256(serial(state_part(payload))).hexdigest() == payload['state_sha256'],
            'Decision-state commitment differs')
    totals = {condition: {arm: Counter() for arm in ARMS} for condition in CONDITIONS}
    cohort, edited_states, unique = Counter(), Counter(), set()
    for entry in payload['case_states']:
        require(set(entry) == {'condition', 'role', 'family', 'arm', 'state', 'count'}, 'Unexpected marginal decision fields')
        condition, role, family, arm, state, count = (entry[k] for k in ('condition', 'role', 'family', 'arm', 'state', 'count'))
        require(condition in CONDITIONS and role in ('edited', 'ordinary_resampled', 'untouched', 'background')
                and arm in ARMS and number(count, 1, 408), 'Invalid counted case decision')
        require((family in FAMILIES) if role in ('edited', 'ordinary_resampled') else family is None,
                'Invalid case family')
        require(role != 'background' or condition != 'clean', 'Unexpected clean background')
        allowed = {'unassessed', 'unflagged', 'located', 'flagged_elsewhere'} if role == 'edited' else {'unassessed', 'unflagged', 'flagged'}
        require(type(state) is str and state in allowed, 'Invalid categorical decision')
        signature = serial({k: v for k, v in entry.items() if k != 'count'})
        require(signature not in unique, 'Duplicate marginal decision bin')
        unique.add(signature)
        if role == 'edited':
            edited_states[condition, family, arm, state] += count
        cohort[condition, role, family, arm] += count
        normalized = 'ordinary' if role in ('untouched', 'ordinary_resampled') else role
        stat = totals[condition][arm]
        stat.update({normalized: count, f'{normalized}_flags': count * (state in ('flagged', 'located', 'flagged_elsewhere')),
                     f'{normalized}_unassessed': count * (state == 'unassessed')})
        if role == 'edited':
            stat.update(labels=count, located=count * (state == 'located'),
                        edited_wrong_location=count * (state == 'flagged_elsewhere'))
    expected_cohort = Counter()
    for condition in CONDITIONS:
        for arm in ARMS:
            expected_cohort[condition, 'untouched', None, arm] = 24
            for family in FAMILIES:
                for role in ('edited', 'ordinary_resampled'):
                    expected_cohort[condition, role, family, arm] = 24
            if condition != 'clean':
                expected_cohort[condition, 'background', None, arm] = 24
        # Marginal checks do not claim to reconstruct the private cross-case correspondence.
        # They still enforce elementary union and assessment bounds.
        for role in ('edited', 'ordinary', 'background'):
            if role == 'background' and condition == 'clean':
                continue
            by = totals[condition]
            flags = [by[a][role + '_flags'] for a in ('incumbent', 'contrast', 'add')]
            require(max(flags[:2]) <= flags[2] <= sum(flags[:2]), 'Union flag counts are inconsistent')
            checked = [by[a][role] - by[a][role + '_unassessed'] for a in ('incumbent', 'projected', 'add')]
            require(max(checked[:2]) <= checked[2] <= min(by['add'][role], sum(checked[:2])), 'Union assessment counts are inconsistent')
            require(by['contrast'][role + '_unassessed'] == by['projected'][role + '_unassessed'], 'Contrast assessment counts are inconsistent')
    require(cohort == expected_cohort and sum(cohort.values()) == 2040, 'Study cases or family denominators differ')
    reader_counts, added_readers, new_ordinary = Counter(), Counter(), Counter()
    ordinary_groups = {condition: Counter() for condition in CONDITIONS}
    unique.clear()
    for entry in payload['reader_outcomes']:
        fields = {'condition', 'gained', 'incumbent_ordinary_flagged', 'add_ordinary_flagged', 'count'}
        require(set(entry) == fields and entry['condition'] in CONDITIONS and number(entry['count'], 1, 8), 'Invalid reader summary bin')
        require(all(type(entry[k]) is bool for k in fields - {'condition', 'count'}), 'Non-boolean reader summary')
        signature = serial({k: v for k, v in entry.items() if k != 'count'})
        require(signature not in unique, 'Duplicate reader-summary bin')
        unique.add(signature)
        condition, count = entry['condition'], entry['count']
        require(not entry['incumbent_ordinary_flagged'] or entry['add_ordinary_flagged'], 'Union lost an ordinary reader flag')
        reader_counts[condition] += count
        added_readers[condition] += count * entry['gained']
        new_ordinary[condition] += count * (entry['add_ordinary_flagged'] and not entry['incumbent_ordinary_flagged'])
        for arm in ('incumbent', 'add'):
            ordinary_groups[condition][arm] += count * entry[arm + '_ordinary_flagged']
    require(reader_counts == dict.fromkeys(CONDITIONS, 8), 'Each condition must summarize eight unlinked reader groups')
    paired_counts, paired_states, gains, gain_families, collisions = Counter(), Counter(), Counter(), defaultdict(set), Counter()
    unique.clear()
    for entry in payload['paired_control_outcomes']:
        fields = {'condition', 'family', 'incumbent_located', 'add_located', 'ordinary_at_edit', 'background_at_edit', 'count'}
        require(set(entry) == fields and entry['condition'] in CONDITIONS and entry['family'] in FAMILIES
                and number(entry['count'], 1, 24), 'Invalid paired-control bin')
        require(all(type(entry[k]) is bool for k in ('incumbent_located', 'add_located'))
                and number(entry['ordinary_at_edit'], 0, 3) and number(entry['background_at_edit'], 0, 1), 'Invalid paired-control state')
        condition, family, count = entry['condition'], entry['family'], entry['count']
        require(condition != 'clean' or entry['background_at_edit'] == 0, 'No clean background control exists')
        require(not entry['incumbent_located'] or entry['add_located'], 'Union lost a located edit')
        signature = serial({k: v for k, v in entry.items() if k != 'count'})
        require(signature not in unique, 'Duplicate paired-control bin')
        unique.add(signature)
        paired_counts[condition, family] += count
        paired_states[condition, family, entry['incumbent_located'], entry['add_located']] += count
        if entry['add_located'] and not entry['incumbent_located']:
            gains[condition] += count
            gain_families[condition].add(family)
            collisions[condition] += count * (entry['ordinary_at_edit'] + entry['background_at_edit'])
    require(paired_counts == {(c, f): 24 for c in CONDITIONS for f in FAMILIES}, 'Paired-control denominators differ')
    for condition in CONDITIONS:
        for family in FAMILIES:
            for arm, slot in (('incumbent', 2), ('add', 3)):
                located = sum(n for signature, n in paired_states.items()
                              if signature[:2] == (condition, family) and signature[slot])
                require(located == edited_states[condition, family, arm, 'located'],
                        'Paired-control outcomes disagree with marginal decisions')
    conditions = {}
    for condition in CONDITIONS:
        require(added_readers[condition] <= gains[condition] <= 6 * added_readers[condition], 'Reader gains contradict case gains')
        for arm in ('incumbent', 'add'):
            require(ordinary_groups[condition][arm] <= totals[condition][arm]['ordinary_flags'] <= 9 * ordinary_groups[condition][arm], 'Reader false flags contradict case flags')
        conditions[condition] = {'by_arm': {arm: dict(counts) for arm, counts in totals[condition].items()},
                                 'added_readers': added_readers[condition],
                                 'families_with_gain': len(gain_families[condition]),
                                 'new_ordinary_groups': new_ordinary[condition],
                                 'same_time_control_collisions': collisions[condition]}
    outcome = {'displays_eligible_for_new_app_validation': eligible(conditions), 'condition_counts': conditions}
    require(outcome == payload['outcome'], 'Public decision-state recount differs from recorded result')
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('development', type=Path)
    args = parser.parse_args()
    payload = read(args.receipt)
    require(payload['commitments']['public_reviewer'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Public reviewer commitment differs')
    print(json.dumps(recount(payload, args.development), indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
