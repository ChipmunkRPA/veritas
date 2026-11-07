"""Supplemental source-renderer replay of four original failed video cases.

Explicit BT.601 limited-range reproduction of the old OpenCV input, not a
metadata-faithful correction or revised native/model quality requirement.
No neural models, weights, changed scores or overwritten original receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import traceback
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from scripts import video_demamba_clock_v13 as original_clock
from scripts.diagnose_video_color_v14 import CANDIDATE_SHA, IDENTITIES, file_sha

FREEZE_SHA = 'fe8aa9a67bcca54259b0d2e9bf7966b9723fc73ac1ebb4d891601559e3898d61'
CLOCK_TOLERANCE = Fraction(11, 10_000_000)


def parse_clock(stderr, count):
    configs = re.findall(r'config in time_base:\s*(\d+/\d+)', stderr)
    rows = re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', stderr)
    if len(configs) != 1 or [int(n) for n, _ in rows] != list(range(count)):
        raise ValueError('Complete ordered integer presentation timestamps required')
    base = Fraction(configs[0])
    if base <= 0:
        raise ValueError('Positive filter time base required')
    return base, [int(pts) for _, pts in rows]


def checked_time(pts, base, source_frame, source_base, recorded_seconds):
    if pts * base != int(source_frame['best_effort_timestamp']) * source_base:
        raise ValueError('Separate integer source/filter timestamps differ')
    difference = abs(pts * base - Fraction(str(recorded_seconds)))
    if difference > CLOCK_TOLERANCE:
        raise ValueError('Original1.1microsecond clock tolerance exceeded')
    return float(difference)


def frames(path, support):
    width, height = support['frames'][0]['native_size']
    indices = [r['native_index'] for r in support['frames']]
    if (len(indices) != 8 or indices != sorted(set(indices)) or indices != support['plan']['selected_indices']
            or any(r['native_size'] != [width, height] for r in support['frames'])):
        raise ValueError('All eight ordered original-resolution frames required')
    expression = '+'.join(f'eq(n\\,{index})' for index in indices)
    filters = ('select=' + expression + ',scale=in_color_matrix=bt601:in_range=tv:out_range=pc,'
               'format=rgb24,showinfo')
    command = ['ffmpeg', '-v', 'info', '-nostdin', '-copyts', '-threads', '1', '-i', str(path),
        '-map', '0:v:0', '-vf', filters, '-filter_threads', '1', '-frames:v', '8',
        '-fps_mode', 'passthrough', '-threads', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True, timeout=120)
    base, pts = parse_clock(result.stderr.decode(errors='replace'), 8)
    if len(result.stdout) != width * height * 3 * 8:
        raise ValueError('Complete independent native RGB byte count required')
    return np.frombuffer(result.stdout, np.uint8).reshape(8, height, width, 3), base, pts


def replay(freeze, journal, source_root, output):
    if output.exists():
        raise ValueError('New supplemental receipt required; never overwrite original audit')
    if file_sha(freeze) != FREEZE_SHA or file_sha(journal) != CANDIDATE_SHA:
        raise ValueError('Exact original public freeze and candidate/support journal required')
    public = json.loads(freeze.read_text())
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    if lines[0]['lock_sha256'] != public['private_lock_sha256'] or len(lines) != 289:
        raise ValueError('Complete original288-case candidate lineage required')
    code = {r['path']: r['sha256'] for r in public['lock']['implementation_sha256']}
    expected = code['backend/scripts/video_demamba_clock_v13.py']
    if file_sha(Path(original_clock.__file__)) != expected:
        raise ValueError('Original crop/resize/normalization implementation changed')
    records = []
    for row in lines[1:]:
        if (row['id'], row['variant']) not in IDENTITIES:
            continue
        record = {'id': row['id'], 'variant': row['variant'], 'source_sha256': row['sha256'],
                  'source_renderer_correspondence': False, 'frames': []}
        try:
            path = source_root / row['relative_path']
            if file_sha(path) != row['sha256'] or row['status'] != 'ok':
                raise ValueError('Original assessed source and file hash required')
            probe = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
                '-show_frames', '-show_streams', '-show_entries',
                'stream=time_base,color_space,color_range:frame=best_effort_timestamp,color_space,color_range',
                '-of', 'json', str(path)], capture_output=True, check=True, timeout=120)
            source = json.loads(probe.stdout)
            stream = source['streams'][0]
            source_base = Fraction(stream['time_base'])
            support = row['support']
            if len(source['frames']) != support['source_frame_count']:
                raise ValueError('Complete original source frame clock required')
            decoded, base, pts = frames(path, support)
            tensors = []
            for rgb, timestamp, trace in zip(decoded, pts, support['frames'], strict=True):
                source_frame = source['frames'][trace['native_index']]
                if (stream['color_space'] != 'bt709' or stream['color_range'] != 'tv'
                        or source_frame['color_space'] != 'bt709' or source_frame['color_range'] != 'tv'):
                    raise ValueError('This fixed diagnostic requires declared BT709 limited-range source frames')
                error = checked_time(timestamp, base, source_frame, source_base, trace['pts_s'])
                tensor, again = original_clock.preprocess(np.ascontiguousarray(rgb[:, :, ::-1]))
                if any(again[k] != trace[k] for k in again):
                    raise ValueError('Exact original RGB/crop/resize/tensor correspondence failed')
                tensors.append(tensor)
                record['frames'].append({'native_index': trace['native_index'], 'filter_pts': timestamp,
                    'filter_time_base': str(base), 'source_pts': int(source_frame['best_effort_timestamp']),
                    'source_time_base': str(source_base), 'original_recorded_seconds': trace['pts_s'],
                    'clock_difference_s': error, 'frame_color_space': source_frame['color_space'],
                    'frame_color_range': source_frame['color_range'], 'rgb_sha256': again['native_rgb_sha256'],
                    'crop_box_xyxy': again['crop_box_xyxy'], 'resized_rgb_sha256': again['resized_rgb_sha256'],
                    'normalized_tensor_sha256': again['normalized_tensor_sha256'], 'passed': True})
            tensor = np.ascontiguousarray(np.stack(tensors)[None], dtype=np.float32)
            digest = hashlib.sha256(tensor.astype('<f4').tobytes()).hexdigest()
            if list(tensor.shape) != support['tensor_shape'] or digest != support['tensor_sha256']:
                raise ValueError('Complete original eight-frame tensor commitment differs')
            record.update(source_renderer_correspondence=True, tensor_sha256=digest)
        except Exception as error:
            traceback.print_exc()
            record.update(error_type=type(error).__name__)
        records.append(record)
    if {(r['id'], r['variant']) for r in records} != IDENTITIES:
        raise ValueError('All four original failed case identities required')
    result = {'schema': 'chipmunk.video-source-renderer-supplement/14', 'code_sha256': file_sha(Path(__file__)),
        'public_freeze_sha256': FREEZE_SHA, 'candidate_journal_sha256': CANDIDATE_SHA,
        'original_preprocess_sha256': expected, 'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
        'renderer': 'Explicit BT601 limited-range YUV to RGB reproducing original OpenCV input despite declared BT709; not metadata-faithful rendering.',
        'clock_tolerance_s': float(CLOCK_TOLERANCE), 'cases': records,
        'all_source_renderer_correspondence': all(r['source_renderer_correspondence'] for r in records),
        'neural_forwards': 0, 'neural_scores_replayed': False, 'original_audit_rewritten': False,
        'original_native_gate_changed': False, 'quality_gate_changed': False, 'serving_changed': False,
        'scope': 'Supplemental exact source-renderer/crop/tensor and integer-clock replay only. Original four native failures and candidate quality rejection remain intact. No model response, new predictive performance, intended source colours or global OpenCV behaviour is proved.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'cases': len(records), 'passed': sum(r['source_renderer_correspondence'] for r in records),
                      'source_frames': sum(len(r['frames']) for r in records), 'neural_forwards': 0}))
    if not result['all_source_renderer_correspondence']:
        raise ValueError('Supplement failed; preserve receipt without changing original conclusions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'journal', 'source_root', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    replay(args.freeze, args.journal, args.source_root, args.output)
