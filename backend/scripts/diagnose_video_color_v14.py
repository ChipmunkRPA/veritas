"""Post-hoc first-frame colour conversion diagnosis; no neural inference.

Preserve original failed audit and rejected candidate. Compare metadata default,
explicit BT.601 and explicit BT.709 in one decoded FFmpeg filter graph. No input
is selected by a neural response, no tolerance is relaxed, and no score changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

CANDIDATE_SHA = '1f2970c6a31ae37f76e31c1b5d4f3fbda396d1a58c1c1ffef2891a5dbeec60b3'
IDENTITIES = {('deepaction-010-Pexels', 'original'), ('deepaction-010-Pexels', 'matched'),
              ('deepaction-074-Pexels', 'original'), ('deepaction-074-Pexels', 'matched')}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cv_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or capture.getBackendName() != 'FFMPEG':
            raise ValueError('Original OpenCV FFmpeg backend required')
        for _ in range(index + 1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Selected original decoded ordinal unavailable')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def ffmpeg_branches(path, index, width, height):
    filters = (f'[0:v]select=eq(n\\,{index}),split=3[a][b][c];'
        '[a]format=rgb24[d];'
        '[b]scale=in_color_matrix=bt601:in_range=tv:out_range=pc,format=rgb24[e];'
        '[c]scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24[f];'
        '[d][e][f]hstack=inputs=3[out]')
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(path),
        '-filter_complex_threads', '1', '-filter_complex', filters, '-map', '[out]', '-frames:v', '1',
        '-fps_mode', 'passthrough', '-threads', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    raw = subprocess.run(command, capture_output=True, timeout=120, check=True).stdout
    if len(raw) != width * 3 * height * 3:
        raise ValueError('All three complete original-resolution RGB outputs required')
    joined = np.frombuffer(raw, np.uint8).reshape(height, width * 3, 3)
    return [np.ascontiguousarray(joined[:, i * width:(i + 1) * width]) for i in range(3)]


def diagnose(root, prior_diagnostic, output):
    if output.exists():
        raise ValueError('New diagnostic receipt required; preserve original failures')
    journal = root / 'candidate.jsonl'
    if file_sha(journal) != CANDIDATE_SHA:
        raise ValueError('Unchanged original candidate/support journal required')
    rows = [json.loads(line) for line in journal.read_text().splitlines()[1:]]
    lock = json.loads((root / 'lock.json').read_text())
    old = json.loads(prior_diagnostic.read_text())
    prior = {(r['id'], r['variant']): r for r in old['cases']}
    if set(prior) != IDENTITIES:
        raise ValueError('Only four already-failed cases in original diagnosis required')
    records = []
    for row in rows:
        identity = row['id'], row['variant']
        if identity not in IDENTITIES:
            continue
        trace = row['support']['frames'][0]
        previous = prior[identity]
        path = Path(lock['base']) / row['relative_path']
        if file_sha(path) != row['sha256'] or trace['native_index'] != previous['first_audited_frame']:
            raise ValueError('Original source bytes or first selected ordinal changed')
        width, height = trace['native_size']
        rgb = cv_frame(path, trace['native_index'])
        if rgb.shape != (height, width, 3) or sha(rgb.tobytes()) != trace['native_rgb_sha256']:
            raise ValueError('Original OpenCV source-frame hash no longer reproduces')
        current = ffmpeg_branches(path, trace['native_index'], width, height)
        if sha(current[0].tobytes()) != previous['replays'][0]['rgb_sha256']:
            raise ValueError('Diagnostic split graph changed original default conversion')
        comparisons = []
        for name, image in zip(('metadata_default', 'explicit_bt601', 'explicit_bt709'), current, strict=True):
            delta = image.astype(np.int16) - rgb.astype(np.int16)
            comparisons.append({'conversion': name, 'rgb_sha256': sha(image.tobytes()),
                'equals_original_opencv': bool(np.array_equal(image, rgb)),
                'changed_channel_values': int(np.count_nonzero(delta)), 'total_channel_values': int(delta.size),
                'max_absolute_channel_difference': int(np.abs(delta).max()),
                'mean_absolute_channel_difference': float(np.abs(delta).mean())})
        records.append({'id': row['id'], 'variant': row['variant'], 'source_sha256': row['sha256'],
            'native_index': trace['native_index'], 'native_size': trace['native_size'],
            'opencv_rgb_sha256': trace['native_rgb_sha256'], 'prior_stream_metadata': previous['stream'],
            'original_default_rgb_reproduced': True, 'comparisons': comparisons})
    if {(r['id'], r['variant']) for r in records} != IDENTITIES:
        raise ValueError('Complete fixed diagnostic case set required')
    result = {'schema': 'chipmunk.video-colour-conversion-diagnostic/14',
        'diagnostic_code_sha256': file_sha(Path(__file__)), 'candidate_journal_sha256': CANDIDATE_SHA,
        'prior_diagnostic_sha256': file_sha(prior_diagnostic), 'opencv_version': cv2.__version__,
        'numpy_version': np.__version__, 'ffmpeg_binary_sha256': file_sha(Path(shutil.which('ffmpeg'))),
        'cases': records, 'neural_forwards': 0, 'original_audit_changed': False,
        'original_rejection_changed': False, 'serving_changed': False,
        'scope': 'Post-hoc first selected frame of the four failed cases, shared single FFmpeg decode and three colour-conversion branches. Original OpenCV/default RGB hashes must reproduce. No complete-clip/model replay, changed pixels in original journal, relaxed tolerance or detection-quality claim.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'cases': len(records), 'bt601_matches': sum(r['comparisons'][1]['equals_original_opencv'] for r in records),
                      'bt709_matches': sum(r['comparisons'][2]['equals_original_opencv'] for r in records), 'neural_forwards': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prior_diagnostic', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    diagnose(args.root, args.prior_diagnostic, args.output)
