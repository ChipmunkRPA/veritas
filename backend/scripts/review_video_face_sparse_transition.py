"""Recount the published sparse-background face transition study without models."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from scripts import video_face_sparse_transition as study


def replay(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        receipt = json.load(source)
    if receipt['schema'] != 'chipmunk.video-face-sparse-transition-evaluation/1':
        raise ValueError('Unexpected evaluation format')
    plan, calibration = receipt['freeze'], receipt['calibration']
    if (plan['reviewer_sha256'] != study.sha(Path(study.__file__))
            or plan['admission'] != study.ADMISSION or plan['calibration'] != study.CALIBRATION):
        raise ValueError('The published scoring code differs from the frozen procedure')
    development, held = receipt['development'], receipt['held']
    groups = {split: {row['group'] for row in rows} for split, rows in (('development', development), ('held', held))}
    if (groups['development'] & groups['held'] or any(sorted(group) != plan['groups'][split] for split, group in groups.items())
            or len({row['sha256'] for row in development + held}) != len(development) + len(held)):
        raise ValueError('Evaluation file bindings or split memberships differ from the frozen procedure')
    maxima = {}
    for row in development:
        measure = row['measured']['off_face_motion']
        if row['expected_frame'] is None and study.is_motion_candidate(measure):
            group = row['group']
            maxima[group] = max(maxima.get(group, -2), measure['score'])
    rule = plan['calibration']
    cut = (max(rule['floor'], sorted(maxima.values(), reverse=True)[rule['allowed_development_groups']])
           if len(maxima) >= rule['minimum_ordinary_groups'] else None)
    if (maxima != calibration['group_maxima'] or cut != calibration['point']
            or calibration['source_groups'] != sorted(groups['development'])):
        raise ValueError('The development rows do not reproduce the recorded calibration')
    by_profile = defaultdict(list)
    for row in held:
        by_profile[row['profile']].append(row)
    result = {'rows': len(held), 'calibrated_point': cut, 'outcomes': study.summarize(held, plan['point'], cut),
              'by_profile': {profile: study.summarize(rows, plan['point'], cut) for profile, rows in by_profile.items()}}
    if result != receipt['summary']:
        raise ValueError('The reported result does not match the recorded scores')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    print(json.dumps(replay(parser.parse_args().receipt), indent=2, sort_keys=True, allow_nan=False))
