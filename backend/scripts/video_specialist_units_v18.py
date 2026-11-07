"""Source-clock plans for video specialist calibration; no model or serving code.

Truth is presence of a publisher-labeled manipulated source frame among the
specific frames supporting a unit. It is not face identity or pixel authorship.
Unobserved edit frames and context-only edits are explicitly distinguished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts import inventory_video_specialists_v18 as inventory
from scripts import prepare_video_specialists_v18 as preparation

require = preparation.require


def scheduled_centers(table, geometry):
    base = Fraction(table['time_base'])
    times = [value * base for value in table['pts']]
    duration = Fraction(geometry['duration_ns'], 10**9)
    count = (duration * 2).__ceil__()
    return sorted({min(range(len(times)), key=lambda i: (abs(times[i] - Fraction(k, 2)), i))
        for k in range(count) if Fraction(k, 2) < duration})


def location(table, geometry, frames, context=()):
    frames, context = sorted(set(frames)), sorted(set(context))
    count = geometry['frame_count']
    require(frames and all(type(i) is int and 0 <= i < count for i in [*frames, *context]),
        'Every sampled/context frame must exist in the native source')
    pts, base = table['pts'], Fraction(table['time_base'])
    first, end = frames[0], frames[-1] + 1
    return {'kind': 'time', 'unit': 'nanoseconds', 'start': round(pts[first] * base * 10**9),
        'end': round(pts[end] * base * 10**9) if end < count else geometry['duration_ns'],
        'extent': geometry['duration_ns'], 'frames': {'start': first, 'end': end, 'extent': count},
        'sampled_frames': frames, 'context_frames': context,
        'pts_table_sha256': geometry['pts_table_sha256']}


def planned(row, table):
    geometry = preparation.frame_clock(table, row['source_geometry']['frame_count'], expected_fps=None)
    require(geometry == row['source_geometry'], 'Unit source clock/geometry differs from prepared input')
    edit = row['known_edit']
    require(type(row['label']) is int and row['label'] in (0, 1)
        and (edit is None) == (row['label'] == 0), 'Explicit full-clip truth required')
    if edit:
        a, b = edit['frames']
        require(0 <= a < b <= geometry['frame_count'] and geometry['native_frame_rate'] == '24'
            and edit['time_ns'] == [round(Fraction(i * 10**9, 24)) for i in (a, b)],
            'Known swap interval must remain its exact native24fps source frames')
        edited = set(range(a, b))
    else:
        edited = set()
    centers, units = scheduled_centers(table, geometry), []

    def append(name, view, target, support, context=()):
        units.append({'id': name, 'view': view, 'target': target,
            'location': location(table, geometry, support, context),
            'label': int(bool(set(support) & edited)), 'status': 'planned',
            'truth_scope': 'Manipulated frame presence among sampled support; no face/pixel authorship label.'})

    for center in centers:
        append(f'crop.{center}', 'face-crop', 'video.sampled_frame_manipulation_presence', [center])
    triplets = [i for i in centers if 0 < i < geometry['frame_count'] - 1]
    for start in range(2, len(triplets) - 4):
        middle = triplets[start:start + 3]
        flanks = triplets[start - 2:start] + triplets[start + 3:start + 5]
        support = sorted({i + k for i in middle for k in (-1, 0, 1)})
        context = sorted({i + k for i in flanks for k in (-1, 0, 1)})
        for view in ('face-appearance', 'photometric'):
            append(f'{view}.{middle[0]}.{middle[-1]}', view,
                'video.sampled_middle_frame_manipulation_presence', support, context)
    observed = {i for unit in units for i in unit['location']['sampled_frames']}
    return {'units': units, 'whole_clip_label': row['label'],
        'planned_scope_label': int(any(unit['label'] for unit in units)),
        'sampled_support_frames': len(observed), 'native_frames': geometry['frame_count'],
        'unobserved_known_edit_frames': len(edited - observed),
        'unsupported_views': sorted({'face-crop', 'face-appearance', 'photometric'} - {u['view'] for u in units}),
        'whole_clip_originality_covered': False,
        'limits': 'Sparse source-frame plan only. Appearance/photometric context is not target truth. No specific face, pixel, lip-sync, audio or general-synthesis conclusion.'}


def attach(plan, responses):
    """Keep every planned local unit; a missing native readout is not zero."""
    expected = {unit['id'] for unit in plan['units']}
    require(set(responses) <= expected, 'Unplanned native observation cannot change calibration scope')
    result = []
    for unit in plan['units']:
        response = responses.get(unit['id'])
        if response is None:
            result.append({**unit, 'status': 'unavailable', 'raw': None,
                'reason': 'Planned native face/window readout unavailable; no originality verdict.'})
        else:
            require(set(response) <= {'location', 'status', 'raw', 'reason', 'measurement_sha256'},
                'Native response must not overwrite planned truth, target or scope')
            require(response['location'] == unit['location'], 'Native response does not match its exact sampled source frames')
            require(response['status'] in ('ok', 'failed', 'skipped', 'not_applicable', 'unavailable'), 'Explicit native status required')
            raw = response['raw']
            require((type(raw) in (int, float) and math.isfinite(raw)) if response['status'] == 'ok'
                else raw is None and isinstance(response.get('reason'), str) and response['reason'],
                'Unavailable/nonfinite native response cannot become a numeric negative')
            result.append({**unit, **response})
    return result


def audit(main, trial, output):
    inventory_path = trial / 'source-inventory.json'
    source_plan = json.loads(inventory_path.read_text())
    expected = {(Path(r['interval_manifest']).parts[-3], r['id']): r for r in source_plan['intervals']}
    journals = {'video-v4': ('development.jsonl', 'test.jsonl'),
        'video-v5': ('current-development.jsonl', 'current-test.jsonl'),
        'video-v6': ('current-test.jsonl',), 'video-v17-sface': ('baseline.jsonl',)}
    counts, sources, units, seen, proofs = Counter(), {}, 0, set(), []
    for folder, names in journals.items():
        for name in names:
            path = main / 'backend/data/independent-evaluation' / folder / name
            sources[str(path.relative_to(main))] = inventory.sha(path)
            header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
            manifest = main / 'backend/data/independent-evaluation' / folder / 'intervals/manifest.json'
            require(header['manifest_sha256'] == inventory.sha(manifest), 'Native table differs from exact source manifest')
            for row in rows:
                key = folder, row['id']
                require(key in expected and key not in seen and all(row[k] == v for k, v in expected[key].items()
                    if k not in ('role', 'component', 'interval_manifest')), 'Changed/duplicate prepared source binding')
                seen.add(key)
                geometry = preparation.frame_clock(row['table'], row['frame_count'], expected_fps=None)
                plan = planned({'label': row['label'], 'source_geometry': geometry,
                    'known_edit': preparation.truth(row)}, row['table'])
                counts['whole_ai_clips' if row['label'] else 'genuine_clips'] += 1
                counts['known_edit_frames_not_directly_sampled'] += plan['unobserved_known_edit_frames']
                counts['whole_ai_clips_with_no_positive_planned_unit'] += row['label'] == 1 and plan['planned_scope_label'] == 0
                counts['genuine_local_units_in_ai_clips'] += row['label'] * sum(u['label'] == 0 for u in plan['units'])
                counts['native_rate_' + geometry['native_frame_rate']] += 1
                units += len(plan['units'])
                proofs.append({'input_sha256': row['sha256'], 'role': expected[key]['role'],
                    'pts_table_sha256': geometry['pts_table_sha256'], 'plan': plan})
    require(len(seen) == len(expected) == 432, 'Every retained native source table required')
    receipt = {'schema': 'chipmunk.video-specialist-unit-clock-audit/18',
        'implementation_sha256': inventory.sha(Path(__file__)), 'source_inventory_sha256': inventory.sha(inventory_path),
        'cached_native_journal_sha256': sources, 'retained_clips': len(seen), 'planned_units': units, **counts,
        'private_unit_plan_sha256': hashlib.sha256(json.dumps(proofs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'new_decodes': 0, 'new_neural_forwards': 0, 'serving_admitted': False,
        'limits': 'Read retained native clocks/source boundaries, not detector scores. Only432 retained clips; full648 preparation and native scoring/calibration remain separate. Sparse frame presence is not pixel/identity truth or a whole-file originality verdict.'}
    inventory.write(output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'trial', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    audit(args.main, args.trial, args.output)
