"""Rebuild prepared private video groups exactly before future native scoring.

No network, model, fitting or serving change. Successful temporary renderings
are removed after the caller records its evidence. Failed attempts remain for
inspection. Retained research media are read-only and are never deleted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from scripts import inventory_video_specialists_v18 as inventory
from scripts import prepare_video_specialists_v18 as preparation

ROOT = Path(__file__).resolve().parents[2]
require = preparation.require


def inputs(main, trial):
    inventory_path = trial / 'source-inventory.json'
    receipt = json.loads((trial / 'input-preparation.json').read_text())
    policy_path = trial / 'input-preparation-policy.json'
    policy = json.loads(policy_path.read_text())
    require(receipt['clips'] == 648 and receipt['all_native_frame_clocks_verified']
        and receipt['policy_sha256'] == inventory.sha(policy_path)
        and receipt['input_plan_sha256'] == inventory.sha(trial / 'input-plan.jsonl')
        and policy['source_inventory_sha256'] == inventory.sha(inventory_path), 'Complete exact prepared input commitment required')
    for name, expected in policy['implementation_sha256'].items():
        require(inventory.sha(ROOT / name) == expected, 'Original source preparation/clock implementation changed')
    original = json.loads(inventory_path.read_text())
    for name, expected in original['parent_manifest_sha256'].items():
        require(inventory.sha(main / name) == expected, 'Prepared source acquisition manifest changed')
    header, *rows = [json.loads(line) for line in (trial / 'input-plan.jsonl').read_text().splitlines()]
    require(header['policy_sha256'] == inventory.sha(policy_path) and len(rows) == 648
        and len({r['id'] for r in rows}) == 648, 'All prepared input identities required')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group']].append(row)
    require(len(grouped) == 162 and all(len(part) == 4 and sum(r['label'] for r in part) == 2
        and len({r['role'] for r in part}) == 1 for part in grouped.values()), 'Every connected four-case source group required')
    require(all(row['role'] == original['assignments'][row['group']]['role']
        and row['component'] == original['assignments'][row['group']]['component'] for row in rows),
        'Input role/component differs from its connected source assignment')
    return original, dict(grouped)


def verify_case(expected, actual, path, table):
    require(actual['group'] == expected['group'] and actual['id'] == expected['source_id']
        and actual['family'] == expected['family'] and actual['label'] == expected['label'], 'Rebuilt source identity/truth changed')
    require(path.stat().st_size == expected['bytes'] and inventory.sha(path) == expected['sha256'],
        'Rebuilt video is not byte-identical to prepared input')
    geometry = preparation.frame_clock(table, actual['frame_count'], expected_fps=None)
    require(not expected['label'] or geometry['native_frame_rate'] == '24', 'Inserted source positives must keep their exact24fps clock')
    require(geometry == expected['source_geometry'] and preparation.truth(actual) == expected['known_edit'],
        'Rebuilt native frame clock, geometry or exact edit boundary changed')


@contextmanager
def group_files(main, trial, original, expected):
    """Yield all four checked paths/native tables; keep a failed temporary group."""
    from app.detectors.video.decode import probe_frames

    require(len(expected) == 4 and len({r['group'] for r in expected}) == 1, 'Exact connected source group required')
    group = expected[0]['group']
    folder = Path(tempfile.mkdtemp(prefix='video-v18-rematerialize-', dir=trial))
    try:
        cases = []
        for row in original['intervals']:
            if row['group'] == group:
                path = main / Path(row['interval_manifest']).parent / row['path']
                require(inventory.sha(path) == row['sha256'], 'Retained original interval changed')
                cases.append((row, path))
        if not cases:
            parents = sorted((r for r in original['parents'] if r['group'] == group), key=lambda r: (r['label'], r['family']))
            require(len(parents) == 3 and len({r['acquisition_manifest'] for r in parents}) == 1,
                'Virtual source group has changed/missing parents')
            source_root = (main / parents[0]['acquisition_manifest']).parent
            (folder / 'normalized').mkdir()
            generated = preparation.helper().prepare(parents, source_root, folder, development=False, hard_controls=False)
            require(len(generated) == 3 and all(r['status'] == 'ready' for r in generated),
                'Failed virtual source retained; no silent replacement')
            original_clip = next(r for r in generated if r['family'] == 'original_videos')
            generated.append(preparation.ordinary(original_clip, folder / original_clip['path'],
                folder / (original_clip['id'] + '-ordinary.mp4')))
            cases = [(row, folder / row['path']) for row in generated]
        elif len(cases) == 3:
            row, path = next(case for case in cases if case[0]['family'] == 'original_videos')
            ordinary = preparation.ordinary(row, path, folder / (row['id'] + '-ordinary.mp4'))
            cases.append((ordinary, folder / ordinary['path']))
        require(len(cases) == 4 and len({r['family'] for r, _ in cases}) == 4, 'Complete source cases required')
        by_family = {row['family']: (row, path) for row, path in cases}
        checked = []
        for row in sorted(expected, key=lambda r: r['id']):
            actual, path = by_family[row['family']]
            table = probe_frames(path, timeout_s=30).to_json()
            verify_case(row, actual, path, table)
            checked.append({'planned': row, 'path': path, 'table': table})
        yield checked
    except BaseException as error:
        inventory.write(folder / 'failed-attempt.json', {'schema': 'chipmunk.video-rematerialization-failure/18',
            'error_type': type(error).__name__, 'group_commitment': inventory.stable(group),
            'original_media_unchanged': True, 'new_neural_forwards': 0})
        raise
    else:
        shutil.rmtree(folder)


def pilot(main, trial, output):
    require(not output.exists(), 'Preserve the prior native pilot; no implicit overwrite or retry')
    original, grouped = inputs(main, trial)
    eligible = [rows for rows in grouped.values() if all(r['materialization']['kind'] == 'virtual-interval-group' for r in rows)]
    require(bool(eligible), 'A fully virtual consumed group is required for the byte/clock pilot')
    selected = min(eligible, key=lambda rows: inventory.stable(rows[0]['group']))
    started = time.monotonic()
    with group_files(main, trial, original, selected) as cases:
        receipt = {'schema': 'chipmunk.video-rematerialization-native-pilot/18',
            'implementation_sha256': inventory.sha(Path(__file__)), 'input_preparation_sha256': inventory.sha(trial / 'input-preparation.json'),
            'selection': 'First hash-ordered fully virtual consumed group; no detector scores used',
            'group_commitment': inventory.stable(selected[0]['group']), 'role': selected[0]['role'],
            'source_clips': len(cases), 'all_input_bytes_and_native_pts_exact': True,
            'input_sha256': [case['planned']['sha256'] for case in cases],
            'pts_table_sha256': [case['planned']['source_geometry']['pts_table_sha256'] for case in cases],
            'duration_s': time.monotonic() - started, 'new_neural_forwards': 0, 'serving_admitted': False,
            'limits': 'Source reconstruction only. No detection result, calibration, fresh source or outside human review.'}
        inventory.write(output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'trial', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    pilot(args.main.resolve(), args.trial.resolve(), args.output.resolve())
