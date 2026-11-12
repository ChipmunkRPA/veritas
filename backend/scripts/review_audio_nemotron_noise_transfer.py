# SPDX-License-Identifier: Apache-2.0
"""Verify matched loud-background speech decisions without either model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 'chipmunk.nemotron-noise-transfer/1'
HEADS = ('old_enhancer', 'new_enhancer')
DIRECTIONS = {'ai', 'original', 'inconclusive', 'unassessed'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def points_for(rows):
    selected = [row for row in rows if row['label'] in {'bonafide', 'isolated'}]
    require(len(rows) == 56 and len(selected) == 32, 'Wrong independent operating-point roster')
    points = {}
    for arm in HEADS:
        negatives = [row for row in selected if row['candidates'][arm]['admitted']]
        require(len({row['group'] for row in negatives if row['label'] == 'bonafide'}) >= 6,
                'Not enough assessed genuine groups to set the point')
        points[arm] = max(row['candidates'][arm]['logit'] for row in negatives)
    return points


def valid_rows(rows):
    return all(type(row['id']) is int and row['incumbent'] in DIRECTIONS and row['label'] in {'bonafide', 'spoof', 'isolated'}
               and set(row['candidates']) == set(HEADS) and len(row['pcm_sha256']) == 64
               and all(type(item['admitted']) is bool and math.isfinite(item['logit'])
                       and math.isfinite(item['suppression_db']) and len(item['pcm_sha256']) == 64
                       for item in row['candidates'].values()) for row in rows)


def result(lock, selection, held):
    require(lock['schema'] == SCHEMA and valid_rows(selection + held), 'Invalid response ledger')
    require([row['id'] for row in selection + held] == list(range(130)), 'Frozen view order changed')
    points = points_for(selection)
    scenes = set(lock['backgrounds'])
    held_groups = defaultdict(list)
    for row in held:
        if row['label'] != 'isolated':
            held_groups[row['group']].append(row)
    require(len(held_groups) == 22 and not set(held_groups).intersection(row['group'] for row in selection)
            and Counter(rows[0]['label'] for rows in held_groups.values()) == {'bonafide': 10, 'spoof': 12}
            and len({row['family'] for row in held if row['label'] == 'spoof'}) == 6, 'Held source classes or roles differ')
    require(all({row['scene'] for row in rows} == {'gain_control', *scenes} and len(rows) == 3
                and len({(row['label'], row['family'], row['origin']['wav_sha256']) for row in rows}) == 1
                for rows in held_groups.values()), 'Matched held views differ in source identity')
    arms = {}
    rule = lock['decision_rule']
    for arm in HEADS:
        counts = defaultdict(Counter)
        gained, false, families, gain_scenes = set(), set(), set(), set()
        clean_or_isolated = 0
        for row in held:
            source, scene, label = row['group'], row['scene'], row['label']
            measure = row['candidates'][arm]
            baseline = row['incumbent']
            add = (measure['admitted'] and measure['logit'] > points[arm] and baseline in {'inconclusive', 'unassessed'})
            cell = counts[scene, label]
            cell['planned'] += 1
            cell['original_assessed'] += int(baseline != 'unassessed')
            cell['enhanced_assessed'] += int(measure['admitted'])
            cell['incumbent_ai'] += int(baseline == 'ai')
            cell['combined_ai'] += int(baseline == 'ai' or add)
            if add and label == 'spoof' and scene in scenes:
                gained.add(source)
                families.add(row['family'])
                gain_scenes.add(scene)
            if add and label == 'bonafide' and scene in scenes:
                false.add(source)
            if add and (label == 'isolated' or label == 'bonafide' and scene == 'gain_control'):
                clean_or_isolated += 1
        arms[arm] = {'cells': {f'{scene}/{label}': dict(cell) for (scene, label), cell in sorted(counts.items())},
                     'new_noisy_synthetic_groups': len(gained), 'generator_families': sorted(families),
                     'noisy_scenes_with_gains': sorted(gain_scenes), 'new_noisy_genuine_groups': len(false),
                     'new_clean_or_isolated_false_flags': clean_or_isolated,
                     'advances': len(gained) >= rule['min_noisy_synthetic_groups'] and len(families) >= rule['min_families']
                     and len(gain_scenes) >= rule['min_scenes'] and len(false) <= rule['max_noisy_genuine_groups']
                     and clean_or_isolated <= rule['max_clean_or_isolated_flags']}
    return {'points': points, 'held_inputs': len(held), 'selection_inputs': len(selection), 'arms': arms}


def export(run):
    lock, expanded, point, sel, held = [read(run / f'{name}.json') for name in
                                       ('freeze', 'enhancements', 'points', 'selection', 'held')]
    roster = read(run / 'roster.json')
    require(sha(run / 'roster.json') == lock['roster_sha256']
            and sha(Path(__file__)) == lock['reviewer_sha256']
            and all(row['freeze_sha256'] == sha(run / 'freeze.json') for row in (sel, held, expanded))
            and all(row['enhancements_sha256'] == sha(run / 'enhancements.json') for row in (sel, held))
            and held['points_sha256'] == sha(run / 'points.json') and point['selection_sha256'] == sha(run / 'selection.json'),
            'A phase no longer matches its frozen predecessor')
    checked = 0
    for row, score, measures in zip(roster, sel['rows'] + held['rows'], expanded['rows'], strict=True):
        require(all(score[key] == value for key, value in row.items()) and measures['id'] == row['id'], 'A result changed views')
        for arm in HEADS:
            require(all(score['candidates'][arm][key] == value for key, value in measures['enhancers'][arm].items())
                    and sha(run / 'enhanced' / f"{row['id']:03}-{arm}.pcm") == measures['enhancers'][arm]['pcm_sha256'],
                    'Enhancement bytes or measurements changed')
            checked += 1
        require(sha(run / 'pcm' / f"{row['id']:03}.pcm") == row['pcm_sha256'], 'Original PCM changed')
        checked += 1
    summary = result(lock, sel['rows'], held['rows'])
    require(summary['points'] == point['points'] and 0 <= expanded['repeat_max_abs'] <= 1e-5, 'Numerical repeat or point differs')
    return {'freeze': lock, 'freeze_sha256': sha(run / 'freeze.json'), 'selection': sel['rows'], 'held': held['rows'],
            'result': summary, 'model_free_pcm_checks': checked, 'repeat_max_abs': expanded['repeat_max_abs']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'review'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode == 'export':
        receipt = export(args.path)
        with args.out.open('x') as destination:
            json.dump(receipt, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write('\n')
    else:
        receipt = read(args.path)
        computed = result(receipt['freeze'], receipt['selection'], receipt['held'])
        require(computed == receipt['result'], 'Public results differ from the replayed outcomes')
    print(json.dumps(receipt['result'], sort_keys=True))


if __name__ == '__main__':
    main()
