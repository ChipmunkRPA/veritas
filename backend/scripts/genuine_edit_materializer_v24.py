"""Exact RGB source copying for private ordinary different-source edits.

Successful temporary clips are removed after evidence is recorded; failures
remain. Publisher-original labels are not independent proof of source origin.
This module loads no neural model and never identifies a person.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from scripts import materialize_video_specialists_v18 as previous
from scripts import prepare_video_scoring_units_v18 as scheduling
from scripts import prepare_video_specialists_v18 as clock

require = previous.require


def digest(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def compose(target, donor, interval):
    a, b = interval
    require(target.dtype == donor.dtype == np.uint8 and target.ndim == donor.ndim == 4
        and target.shape[1:] == donor.shape[1:] == (480, 640, 3)
        and type(a) is type(b) is int and 0 < a < b < len(target) and b <= len(donor),
        'Exact native RGB parents and an interior source-frame interval required')
    result = target.copy()
    result[a:b] = donor[a:b]
    return result


def encode(path, pixels):
    require(not path.exists() and pixels.dtype == np.uint8 and pixels.shape[1:] == (480, 640, 3), 'New lossless RGB target required')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', '640x480', '-framerate', '24', '-i', 'pipe:0', '-map_metadata', '-1',
        '-an', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-threads', '2',
        '-g', '48', '-fflags', '+bitexact', '-flags:v', '+bitexact', str(path)],
        input=np.ascontiguousarray(pixels).tobytes(), check=True, timeout=90)


def decode(path, table):
    from app.detectors.video.decode import FrameTable, decode_frames

    native = FrameTable.from_json(table)
    frames = list(decode_frames(path, native, list(range(len(native))), timeout_s=120))
    require([f.index for f in frames] == list(range(len(native))), 'Incomplete/reordered source decoding')
    return np.stack([f.rgb for f in frames])


def checked_clip(path, identity, component):
    from app.detectors.video.decode import probe_frames

    table = probe_frames(path, timeout_s=30).to_json()
    geometry = clock.frame_clock(table, len(table['pts']), expected_fps=24)
    row = {'id': identity, 'group': component, 'sha256': previous.inventory.sha(path),
        'source_geometry': geometry, 'label': 0, 'known_edit': None}
    return {'path': path, 'table': table, 'planned': row}, {**row, **scheduling.plan_case(row)}


@contextmanager
def pair_files(main, trial, pair, original, grouped, original_rgb):
    """Reconstruct both originals and both edit directions; retain failed input."""
    from app.detectors.video.decode import probe_frames

    folder = Path(tempfile.mkdtemp(prefix='ordinary-v24-pair-', dir=trial))
    try:
        with ExitStack() as contexts:
            parents, decoded = [], []
            for parent in pair['parents']:
                cases = contexts.enter_context(previous.group_files(main,
                    main / 'backend/data/independent-evaluation/video-specialists-v18', original, grouped[parent['group']]))
                case = next(c for c in cases if c['planned']['id'] == parent['id'])
                require(case['planned']['family'] == 'original_videos' and case['planned']['label'] == 0
                    and case['planned']['sha256'] == parent['sha256']
                    and case['planned']['source_geometry'] == parent['source_geometry'], 'Planned original parent changed')
                pixels = decode(case['path'], case['table'])
                require([digest(frame) for frame in pixels] == original_rgb[parent['id']], 'Original RGB differs from completed source audit')
                copied = folder / ('original-' + str(len(parents)) + '.mp4')
                shutil.copy2(case['path'], copied)
                require(previous.inventory.sha(copied) == parent['sha256'], 'Temporary original copy changed bytes')
                parents.append({**case, 'path': copied})
                decoded.append(pixels)
            composites = []
            for direction, (target, donor) in enumerate(pair['directions']):
                require(pair['insert_source_frames'] == pair['target_frames'] == [48, 96], 'Frozen ordinary copy interval changed')
                pixels = compose(decoded[target], decoded[donor], pair['target_frames'])
                path = folder / f'direction-{direction}.mp4'
                encode(path, pixels)
                case, planned = checked_clip(path, pair['component'] + '.' + str(direction), pair['component'])
                actual = decode(path, case['table'])
                require(np.array_equal(actual, pixels), 'Encoded ordinary clip changed copied source RGB')
                require(planned['source_geometry']['frame_count'] == len(decoded[target]), 'Ordinary clip frame extent changed')
                hashes = [digest(frame) for frame in actual]
                composites.append({'case': case, 'plan': planned, 'rgb_sha256': hashes,
                    'anchors': {str(i): actual[i].copy() for i in pair['before_during_after_frames']},
                    'target_parent': pair['parents'][target]['id'], 'donor_parent': pair['parents'][donor]['id']})
                del pixels, actual
            # Probe tables remain native, not a synthetic frame_index/fps table.
            require(all(probe_frames(r['case']['path'], timeout_s=30).to_json() == r['case']['table'] for r in composites), 'Native clock changed after RGB verification')
        # Source-only rematerialization contexts close before any caller may
        # run a model. Their failure receipts therefore never misreport a later
        # inference failure as a zero-inference preparation failure.
        del decoded
        yield parents, composites
    except BaseException as error:
        previous.inventory.write(folder / 'failed-attempt.json', {'schema': 'chipmunk.ordinary-edit-materialization-failure/24',
            'component': pair['component'], 'error_type': type(error).__name__, 'original_media_unchanged': True})
        raise
    else:
        shutil.rmtree(folder)


def source_inputs(main, trial):
    source_trial = main / 'backend/data/independent-evaluation/video-specialists-v18'
    plan = json.loads((trial / 'source-plan.json').read_text())
    require(plan['source_plan_sha256'] == previous.inventory.sha(source_trial / 'scoring-input-plan-v2.json')
        and len(plan['pairs']) == 20 and plan['new_neural_forwards'] == 0, 'Exact prospective twenty-pair source plan required')
    original, grouped = previous.inputs(main, source_trial)
    source_rgb = {r['id']: r['rgb_sha256'] for r in [json.loads(s) for s in
        (source_trial / 'source-frame-audit-v1.jsonl').read_text().splitlines()][1:]}
    return plan, original, grouped, source_rgb
