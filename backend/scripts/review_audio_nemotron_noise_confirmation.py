# SPDX-License-Identifier: Apache-2.0
"""Replay the second noise transfer using its parent's fixed points."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from review_audio_nemotron_noise_transfer import HEADS, read, require, sha, valid_rows

SCHEMA = 'chipmunk.nemotron-noise-confirmation/1'
PARENT = Path(__file__).resolve().parents[2] / 'docs/evaluation-panels/audio-nemotron-noise-transfer-2026-11-12.json'


def calculate(freeze, scored):
    require(freeze['schema'] == SCHEMA and freeze['parent_sha256'] == sha(PARENT)
            and freeze['decision_points'] == read(PARENT)['result']['points'], 'Frozen parent decision points differ')
    scenes = set(freeze['backgrounds'])
    require(valid_rows(scored) and [row['id'] for row in scored] == list(range(92)), 'Incomplete held outcomes')
    require(Counter((row['scene'], row['label']) for row in scored) == {
        **{(scene, label): 14 for scene in (*scenes, 'gain_control') for label in ('bonafide', 'spoof')},
        **{(scene, 'isolated'): 4 for scene in scenes}}, 'Unbalanced held roster')
    grouped = defaultdict(list)
    for row in scored:
        if row['label'] != 'isolated':
            grouped[row['group']].append(row)
    require(len(grouped) == 28 and all({r['scene'] for r in rows} == scenes | {'gain_control'} and len(rows) == 3
            and len({r['origin']['sha256'] for r in rows}) == len({r['label'] for r in rows}) == 1
            for rows in grouped.values()), 'Matched source delivery differs')
    summary = {}
    for arm in HEADS:
        point = freeze['decision_points'][arm]
        cells, added_synthetic, added_genuine, scenes_with_gains = defaultdict(Counter), set(), set(), set()
        clean_or_isolated = 0
        for row in scored:
            label, scene, group = (row[key] for key in ('label', 'scene', 'group'))
            cell = cells[scene, label]
            old = row['incumbent']
            candidate = row['candidates'][arm]
            passed = candidate['admitted'] and candidate['logit'] > point
            additional = passed and old in {'inconclusive', 'unassessed'}
            cell['planned'] += 1
            cell['original_ai'] += int(old == 'ai')
            cell['enhanced_ai'] += int(passed)
            cell['combined_ai'] += int(old == 'ai' or additional)
            cell['original_assessed'] += int(old != 'unassessed')
            cell['enhanced_assessed'] += int(candidate['admitted'])
            if additional and scene in scenes and label == 'spoof':
                added_synthetic.add(group)
                scenes_with_gains.add(scene)
            if additional and scene in scenes and label == 'bonafide':
                added_genuine.add(group)
            if additional and (label == 'isolated' or label == 'bonafide' and scene == 'gain_control'):
                clean_or_isolated += 1
        rule = freeze['confirmation_rule']
        summary[arm] = {'counts': {f'{scene}/{label}': dict(cell) for (scene, label), cell in sorted(cells.items())},
                        'added_synthetic_sources': len(added_synthetic), 'added_genuine_sources': len(added_genuine),
                        'scenes_with_gains': sorted(scenes_with_gains), 'new_clean_or_isolated': clean_or_isolated,
                        'passes': len(added_synthetic) >= rule['minimum_added_noisy_synthetic_sources']
                        and len(added_genuine) <= rule['maximum_added_noisy_genuine_sources']
                        and len(scenes_with_gains) >= rule['minimum_scenes_with_gains']
                        and clean_or_isolated <= rule['maximum_added_clean_or_isolated']}
    return summary


def export(run):
    freeze, roster, selected, points, enhanced, held = [read(run / f'{name}.json') for name in
        ('freeze', 'roster', 'selection', 'points', 'enhancements', 'held')]
    require(sha(Path(__file__)) == freeze['reviewer_sha256'] and sha(run / 'roster.json') == freeze['roster_sha256']
            and selected['from_parent_sha256'] == freeze['parent_sha256'] and selected['rows'] == []
            and points['points'] == freeze['decision_points'] and sha(run / 'selection.json') == points['selection_sha256']
            and held['points_sha256'] == sha(run / 'points.json') and held['enhancements_sha256'] == sha(run / 'enhancements.json')
            and sha(run / 'freeze.json') == held['freeze_sha256'] == enhanced['freeze_sha256'], 'Frozen phase binding differs')
    count = 0
    for source, row, extra in zip(roster, held['rows'], enhanced['rows'], strict=True):
        require(source['id'] == row['id'] == extra['id'] and all(row[key] == value for key, value in source.items()),
                'A model response differs from its planned input')
        require(sha(run / 'pcm' / f"{source['id']:03}.pcm") == row['pcm_sha256'], 'Native input changed')
        count += 1
        for arm in HEADS:
            pcm = run / 'enhanced' / f"{row['id']:03}-{arm}.pcm"
            require(all(row['candidates'][arm][key] == value for key, value in extra['enhancers'][arm].items())
                    and sha(pcm) == row['candidates'][arm]['pcm_sha256'], 'Enhanced input changed')
            count += 1
    return {'freeze': freeze, 'freeze_sha256': sha(run / 'freeze.json'), 'held': held['rows'],
            'model_free_pcm_checks': count, 'repeat_max_abs': enhanced['repeat_max_abs'], 'result': calculate(freeze, held['rows'])}


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
        lock = (json.dumps(receipt['freeze'], indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        require(hashlib.sha256(lock).hexdigest() == receipt['freeze_sha256'], 'Public freeze hash differs')
        require(calculate(receipt['freeze'], receipt['held']) == receipt['result'], 'Published results differ')
    print(json.dumps(receipt['result'], sort_keys=True))


if __name__ == '__main__':
    main()
