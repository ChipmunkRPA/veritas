# SPDX-License-Identifier: Apache-2.0
"""Recount the published severe-overlap comparison without model weights or NumPy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

RATE = 16000
RULE = {
    'assignment': 'Fix one permutation per file by both reference stems and separation scores outside all true edits.',
    'minimum_assignment_samples': RATE // 2,
    'severity_db': [-6, -10],
    'foreground_gain_floor_db': -3,
    'assignment_margin_db': 3,
    'views': ['direct', 'mixture_phase'],
    'clock': 'Same exact samples throughout. No alignment shifts or gain normalization of separator outputs.',
}


def summary(rows):
    result = {'rows': len(rows), 'roles': dict(Counter(row['role'] for row in rows)),
              'assessed': sum(row['status'] == 'scored' for row in rows), 'views': {}}
    for name in RULE['views']:
        edited, ordinary, backgrounds = [], [], []
        for row in rows:
            review = row.get('views', {}).get(name)
            if review is None:
                continue
            if row['role'] == 'background':
                backgrounds.append(review['second_output_ratio_db'])
            else:
                lane = edited if row['role'] == 'edited' else ordinary
                lane.extend(review['scored_ranges'])

        def counts(values):
            usable = [row for row in values if row['scale_invariant_gain_db'] is not None and row['target_gain_db'] is not None]
            return {'readable': len(usable),
                    'median_si_sdr_improvement_db': median(row['scale_invariant_gain_db'] for row in usable) if usable else None,
                    'median_target_level_change_db': median(row['target_gain_db'] for row in usable) if usable else None,
                    'improved_without_over_3db_target_loss': sum(row['scale_invariant_gain_db'] > 0 and row['target_gain_db'] >= -3 for row in usable),
                    'target_stronger_than_residual': sum(row['target_vs_residual_db'] > 0 for row in usable),
                    'strong_local_assignment_disagreements': sum(row['assignment_disagrees'] and row['local_assignment_margin_db'] >= 3 for row in usable)}

        result['views'][name] = {'all_edits': counts(edited),
                                 'edits_at_or_below_minus_6_db': counts([row for row in edited if row['input_ratio_db'] <= -6]),
                                 'edits_at_or_below_minus_10_db': counts([row for row in edited if row['input_ratio_db'] <= -10]),
                                 'ordinary': counts(ordinary),
                                 'background_second_output_median_db': median(backgrounds) if backgrounds else None}
    return result


PANELS = {'ami_matched': (11, 75), 'two_voice': (6, 6), 'two_voice_attenuated': (6, 6), 'ami_transfer': (4, 30),
          'sepformer_two_voice_attenuated': (6, 6)}


def _file_sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _validate_panel(panel, groups, edits):
    rows = panel['rows']
    roles = {'edited', 'ordinary', 'background'}
    if (panel['upstream']['total'] != 3 * groups or len(rows) != 3 * groups
            or {(row['group'], row['role']) for row in rows} != {(g, role) for g in range(1, groups + 1) for role in roles}
            or sum(row['expected_edits'] for row in rows) != edits):
        raise ValueError('The receipt has an incomplete set of exact-clock pairs and controls')
    if any(row['expected_edits'] < 0 or (row['role'] == 'edited') != (row['expected_edits'] > 0) for row in rows):
        raise ValueError('Only edited files can carry truth intervals')
    for row in rows:
        if 'views' not in row:
            if row['status'] not in {'no_active_profiles', 'silent_input'}:
                raise ValueError('Scored separator outputs require view measurements')
            continue
        if (row['status'] != 'scored' or set(row['views']) != set(RULE['views'])
                or not math.isfinite(row['relative_reconstruction_error']) or row['relative_reconstruction_error'] < 0):
            raise ValueError('Invalid measured tracks')
        for view in row['views'].values():
            if row['role'] == 'background':
                if set(view) != {'second_output_ratio_db'} or not math.isfinite(view['second_output_ratio_db']):
                    raise ValueError('Background-only controls have no target reference')
                continue
            ranges = view['scored_ranges']
            expected = row['expected_edits'] or 1
            if (len(ranges) != (0 if view['assignment_margin_db'] is None else expected)
                    or any(not 0 <= interval['start_sample'] < interval['end_sample'] for interval in ranges)):
                raise ValueError('An edited interval differs from its fixed global assignment scope')
        if row['role'] == 'edited':
            clocks = [[(item['start_sample'], item['end_sample']) for item in view['scored_ranges']]
                      for view in row['views'].values() if view['assignment_margin_db'] is not None]
            if clocks and any(clock != clocks[0] for clock in clocks):
                raise ValueError('Alternative views must preserve the same edit clock')


def replay(path):
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        receipt = json.load(file)
    if (receipt['schema'] != 'chipmunk.severe-overlap-separation-review/1' or receipt['rule'] != RULE
            or set(receipt['panels']) != set(PANELS)):
        raise ValueError('Unexpected severe overlap receipt or measurement procedure')
    sources = Path(__file__).resolve().parent
    result = {}
    for key, panel in receipt['panels'].items():
        _validate_panel(panel, *PANELS[key])
        expected_method = 'sepformer' if key == 'sepformer_two_voice_attenuated' else 'kezek'
        if (panel['upstream']['method'] != expected_method or panel['method'] != RULE
                or panel['reader_sha256'] != _file_sha(sources / 'audio_severe_overlap_measure.py')
                or panel['upstream']['code_sha256'] != _file_sha(sources / 'audio_severe_overlap_separation.py')):
            raise ValueError('The numeric receipt differs from the pinned local source code')
        reproduced = summary(panel['rows'])
        if reproduced != panel['summary']:
            raise ValueError('The reported severe overlap counts differ from the measurement rows')
        result[key] = reproduced
    if (receipt['panels']['sepformer_two_voice_attenuated']['upstream']['panel_sha256']
            != receipt['panels']['two_voice_attenuated']['upstream']['panel_sha256']):
        raise ValueError('The two separators must be measured on the same attenuated inputs')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    print(json.dumps(replay(parser.parse_args().receipt), indent=2, sort_keys=True, allow_nan=False))
