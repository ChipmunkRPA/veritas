"""Freeze fixed local supports from already measured constant native clocks.

The rational frame-index axis is a coordinate calculation, not an invented
native PTS table. A future scorer must re-probe and match the committed actual
table. No face model, detector score or calibration chooses these positions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts import inventory_video_specialists_v18 as inventory
from scripts import materialize_video_specialists_v18 as media
from scripts import video_specialist_units_v18 as previous

require = media.require


def plan_case(row):
    geometry = row['source_geometry']
    rate, count = Fraction(geometry['native_frame_rate']), geometry['frame_count']
    require(rate in (24, 25) and type(count) is int and 2 <= count <= 300
        and geometry['width'] == 640 and geometry['height'] == 480
        and geometry['duration_ns'] == round(count / rate * 10**9),
        'The prepared zero-origin constant native frame clock is required')
    # Reparameterize the verified constant source clock by frame index. The
    # native clock commitment remains geometry['pts_table_sha256'] throughout.
    axis = {'pts': list(range(count)), 'time_base': str(1 / rate)}
    centers = previous.scheduled_centers(axis, geometry)
    edit = row['known_edit']
    require(type(row['label']) is int and row['label'] in (0, 1) and (edit is None) == (row['label'] == 0), 'Explicit clip truth required')
    if edit:
        a, b = edit['frames']
        require(rate == 24 and 0 <= a < b <= count
            and edit['time_ns'] == [round(Fraction(i * 10**9, 24)) for i in (a, b)], 'Verified edit/frame clock changed')
        changed = set(range(a, b))
    else:
        changed = set()
    units = []

    def append(identity, view, target, frames, context=()):
        units.append({'id': identity, 'view': view, 'target': target,
            'location': previous.location(axis, geometry, frames, context),
            'label': int(bool(set(frames) & changed)), 'status': 'planned',
            'truth_scope': 'Manipulated source-frame presence in sampled support; no identified-face/pixel truth.'})

    for center in centers:
        append(f'crop.{center}', 'face-crop', 'video.sampled_frame_manipulation_presence', [center])
    triplets = [i for i in centers if 0 < i < count - 1]
    for start in range(2, len(triplets) - 4):
        middle = triplets[start:start + 3]
        flanks = triplets[start - 2:start] + triplets[start + 3:start + 5]
        frames = sorted({i + k for i in middle for k in (-1, 0, 1)})
        context = sorted({i + k for i in flanks for k in (-1, 0, 1)})
        for prefix, view in (('face-appearance', 'face-appearance'), ('features', 'face-feature-change')):
            append(f'{prefix}.{middle[0]}.{middle[-1]}', view,
                'video.sampled_middle_frame_manipulation_presence', frames, context)
            units[-1]['middle_centers'] = middle
            units[-1]['before_centers'] = triplets[start - 2:start]
            units[-1]['after_centers'] = triplets[start + 3:start + 5]
    observed = {i for unit in units for i in unit['location']['sampled_frames']}
    return {'units': units, 'centers': centers, 'planned_scope_label': int(any(u['label'] for u in units)),
        'whole_clip_label': row['label'], 'unobserved_known_edit_frames': len(changed - observed),
        'whole_clip_originality_covered': False}


def prepare(main, trial, output):
    _, grouped = media.inputs(main, trial)
    files, counts = [], Counter()
    for group in sorted(grouped, key=inventory.stable):
        for row in sorted(grouped[group], key=lambda r: r['id']):
            current = plan_case(row)
            require(current['planned_scope_label'] == row['label'],
                'A known edit falls entirely outside planned support; retain and handle the missing scope explicitly')
            files.append({**row, **current})
            counts[row['role']] += 1
    result = {'schema': 'chipmunk.video-specialist-scoring-input-plan/18',
        'preparation_code_sha256': inventory.sha(Path(__file__)),
        'prepared_input_sha256': inventory.sha(trial / 'input-preparation.json'),
        'source_inventory_sha256': inventory.sha(trial / 'source-inventory.json'),
        'coordinate_method': 'Frame indices divided by measured constant native rate; source preparation verified zero origin. Actual native PTS must match again before model inference.',
        'semantic_note': 'The earlier metadata prototype called the feature-change view photometric. The new view is explicitly a learned patch-feature transition; it is not a color/light statistic.',
        'files': files, 'source_groups': len(grouped), 'role_files': dict(counts),
        'scoring_protocol_complete': False, 'new_neural_forwards': 0, 'serving_admitted': False}
    inventory.write(output, result)
    print(json.dumps({'planned_clips': len(files), 'planned_units': sum(len(r['units']) for r in files),
        'source_groups': len(grouped), 'new_neural_forwards': 0, 'scoring_protocol_complete': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'trial', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.main.resolve(), args.trial.resolve(), args.output.resolve())
