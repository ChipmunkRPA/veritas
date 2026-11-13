# SPDX-License-Identifier: Apache-2.0
"""Recount the PNG layout ablation with the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POINT = ROOT / 'docs/evaluation-panels/image-v7-smaug-genuine-operating-point.json'
SCOPE = ROOT / 'docs/evaluation-panels/image-whole-png-scope-probe-2026-11-12.json'
DEV = ROOT / 'docs/evaluation-panels/image-whole-png-transfer-2026-11-12.json'
JPEG = ROOT / 'docs/evaluation-panels/image-whole-fresh-sources-2026-11-12.json'
RUNNER = Path(__file__).with_name('image_png_layout_guard_transfer.py')
CAPTURE = Path(__file__).with_name('capture_image_png_layout_controls.cjs')
EXPECTED_PANELS = {
    ('development', 'synthetic'): 16, ('development', 'ordinary'): 12, ('development', 'screen'): 8,
    ('retained-layout', 'ordinary'): 12, ('retained-layout', 'screen'): 12,
    ('new-layout', 'ordinary'): 12, ('new-layout', 'screen'): 12,
    ('new-native', 'synthetic'): 8, ('png-from-jpeg', 'synthetic'): 18,
}
ARMS = ('unguarded', 'flat_floor', 'layout')
SCHEMA = 'chipmunk.image-png-layout-guard/1'
GATE = {'maximum_new_photo_groups': 1, 'maximum_native_png_loss': 1, 'maximum_loss_per_jpeg_family': 1,
        'maximum_unexcluded_new_screen_groups': 1, 'maximum_genuine_flags': 1,
        'minimum_generated_indications_added': 8, 'minimum_generated_families_added': 2}
BASELINE_COMMIT = 'c8ff179fd9a3e9289fe06b772e1c4e351bf6778c'
DIRECTIONS = {'ai', 'record', 'original', 'inconclusive', 'unavailable'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(path.read_bytes())


def hashes():
    return {path.name: digest(path.read_bytes()) for path in (DEV, SCOPE, JPEG, POINT, RUNNER)}


def verify_score(measured, frozen, known):
    status = measured['status']
    mean = measured['mean']
    require(type(mean) in (int, float) and math.isfinite(mean), 'Non-finite response')
    if frozen['panel'] == 'development':
        require(status == 'reused' and mean == frozen['prior_score'] == known[frozen['id']]['mean_logit']
                and frozen['sha256'] == known[frozen['id']]['sha256'], 'Known response changed')
        return
    values, crops = measured['crops'], measured['crop_rgb_sha256']
    require(status == 'measured' and len(values) == len(crops) == 3
            and all(type(value) in (int, float) and math.isfinite(value) for value in values)
            and all(re.fullmatch('[a-f0-9]{64}', crop) for crop in crops)
            and abs(math.fsum(values) / 3 - mean) <= .0001, 'Invalid measured three-view response')


def audit(payload):
    frozen = payload['freeze']
    require(digest(encoded(frozen)) == payload['freeze_sha256'] and frozen['schema'] == SCHEMA
            and frozen['pins'] == hashes() and frozen['gate'] == GATE
            and frozen['capture_runner_sha256'] == digest(CAPTURE.read_bytes())
            and frozen['point'] == read(POINT)['gate']['raw_mean_logit'], 'Frozen commitments differ')
    counts = Counter((row['panel'], row['label']) for row in frozen['rows'])
    require(counts == EXPECTED_PANELS and len({row['id'] for row in frozen['rows']}) == sum(counts.values()),
            'Wrong cohort size or duplicate source ID')
    groups = defaultdict(list)
    counts = defaultdict(Counter)
    method = frozen['method']
    known = {row['id']: row for row in read(DEV)['cases']}
    for row, measured in zip(frozen['rows'], payload['outcomes'], strict=True):
        require(all(measured[key] == row[key] for key in ('id', 'sha256', 'size')), 'Metric source or geometry changed')
        w, h = measured['size']
        flat = measured['flat_blocks']
        ink = measured['ink_words']
        horizontal, vertical = measured['edges']
        blocks = (w // 8) * (h // 8)
        require(all(type(value) is int for value in (w, h, flat, ink, horizontal, vertical))
                and 768 <= min(w, h) <= max(w, h) <= 4920 and 0 <= flat <= blocks and 0 <= ink <= 10000
                and 0 <= horizontal <= w and 0 <= vertical <= h, 'Impossible native layout measurement')
        flag_flat = flat / blocks > method['flat_block_floor']
        flag_layout = (flat / blocks > method['flat_block_limit'] or ink >= method['minimum_same_ink_words']
                       or horizontal >= max(method['minimum_edge_pixels'], w*method['minimum_edge_share'])
                       or vertical >= max(method['minimum_edge_pixels'], h*method['minimum_edge_share']))
        require(measured['guards'] == {'flat_floor': flag_flat, 'layout': flag_layout}, 'Layout rule changed')
        verify_score(measured['candidate'], row, known)
        elevated = measured['candidate']['mean'] > frozen['point']
        key = (row['panel'], row['label'], row['family'])
        cell = counts[key]
        cell['n'] += 1
        for arm, excluded in zip(ARMS, (False, flag_flat, flag_layout), strict=True):
            cell[arm+'_elevated'] += elevated and not excluded
            cell[arm+'_excluded'] += excluded
        groups[row['panel'], row['label'], row['group']].append((flag_layout, elevated))
    new_screen_open = [g for (p, label, g), vals in groups.items() if p == 'new-layout' and label == 'screen'
                       and not all(excluded for excluded, _ in vals)]
    new_photo_closed = [g for (p, label, g), vals in groups.items() if p == 'new-layout' and label == 'ordinary'
                        and any(excluded for excluded, _ in vals)]
    genuine_groups = [g for (p, label, g), vals in groups.items() if p != 'development' and label != 'synthetic'
                      and any(elevated and not excluded for excluded, elevated in vals)]
    native = counts['new-native', 'synthetic', 'gemini']
    generated = [cell for (p, label, _), cell in counts.items() if p != 'development' and label == 'synthetic']
    jpeg = [cell for (p, _, _), cell in counts.items() if p == 'png-from-jpeg']
    # A separate Full comparison still has to show incremental end-user findings.
    candidate_gate = {
        'new_photo_groups': len(new_photo_closed) <= GATE['maximum_new_photo_groups'],
        'new_native_loss': native['unguarded_elevated'] - native['layout_elevated'] <= GATE['maximum_native_png_loss'],
        'png_from_jpeg_family_loss': all(
            cell['unguarded_elevated'] - cell['layout_elevated'] <= GATE['maximum_loss_per_jpeg_family'] for cell in jpeg),
        'new_screen_groups': len(new_screen_open) <= GATE['maximum_unexcluded_new_screen_groups'],
        'genuine_flags': len(set(genuine_groups)) <= GATE['maximum_genuine_flags'],
        'additional_generated_responses': sum(cell['layout_elevated'] for cell in generated) >= GATE['minimum_generated_indications_added'],
        'additional_generated_families': sum(bool(cell['layout_elevated']) for cell in generated) >= GATE['minimum_generated_families_added'],
    }
    result = {'cells': {'/'.join(key): dict(val) for key, val in sorted(counts.items())}, 'candidate_gate': candidate_gate,
            'candidate_passed': all(candidate_gate.values()), 'full_comparison': False,
            'new_open_screen_groups': sorted(new_screen_open), 'new_excluded_photo_groups': sorted(new_photo_closed),
            'genuine_groups_flagged': sorted(set(genuine_groups))}
    if 'baseline' in payload:
        result['full_comparison'] = full_comparison(payload, frozen, result['candidate_passed'])
    return result


def full_comparison(payload, frozen, candidate_passed):
    baseline = payload['baseline']
    lock = baseline['lock_public']
    require(lock['input_freeze_sha256'] == payload['freeze_sha256']
            and lock['source_sha256']['commit'] == BASELINE_COMMIT and lock['baseline']['profile'] == 'full',
            'Full comparison does not match the pinned application and inputs')
    require(digest(encoded(lock)) == baseline['lock_public_sha256']
            and re.fullmatch('[a-f0-9]{64}', baseline['lock_sha256']), 'Full comparison lock changed')
    rows = [row for row in frozen['rows'] if row['panel'] != 'development']
    measured = {row['id']: row for row in payload['outcomes']}
    table = defaultdict(Counter)
    added_families, added_ordinary, lost = set(), set(), set()
    for row, outcome in zip(rows, baseline['outcomes'], strict=True):
        name, label, family = row['id'], row['label'], row['family']
        direction = outcome['direction']
        require(outcome['id'] == name and outcome['sha256'] == row['sha256'] and outcome['status'] == 'done'
                and direction in DIRECTIONS and re.fullmatch('[a-f0-9]{64}', outcome['result_sha256'])
                and re.fullmatch('[a-f0-9]{64}', outcome['packet_sha256']), 'Full source or response is incomplete')
        cell = table[row['panel'], label, family if label == 'synthetic' else label]
        entry = measured[name]
        cue = not entry['guards']['layout'] and entry['candidate']['mean'] > frozen['point']
        added = cue and direction not in {'ai', 'record'}
        cell['files'] += 1
        cell[f'full_{direction}'] += 1
        cell['supplementary_indications'] += cue
        cell['added_indications'] += added
        cell['added_disagreements'] += added and direction == 'original'
        if added:
            (added_families if label == 'synthetic' else added_ordinary).add(family if label == 'synthetic' else row['group'])
        if entry['guards']['layout'] and entry['candidate']['mean'] > frozen['point']:
            lost.add(name)
    added_generated = sum(cell['added_indications'] for key, cell in table.items() if key[1] == 'synthetic')
    return {'files': len(rows), 'baseline_commit': BASELINE_COMMIT,
            'cells': {'/'.join(key): dict(cell) for key, cell in sorted(table.items())},
            'added_generated_indications': added_generated, 'added_generated_families': sorted(added_families),
            'added_ordinary_groups': sorted(added_ordinary), 'suppressed_supplementary_indications': len(lost),
            'passed': candidate_passed and added_generated >= GATE['minimum_generated_indications_added']
                      and len(added_families) >= GATE['minimum_generated_families_added']
                      and len(added_ordinary) <= GATE['maximum_genuine_flags']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    payload = read(args.receipt)
    result = audit(payload)
    require(result == payload['summary'], 'Recorded outcome summary differs')
    print(encoded(result).decode(), end='')
