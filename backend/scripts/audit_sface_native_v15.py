"""Synthetic-only native SFace alignment/channel/graph preflight, sequential."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from scripts.sface_native_v15 import TARGET, cpp_alignment, face_row, manual_blob

WEIGHT_SHA = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
CPP_SHA = '92bb8b201f2f84c19e5be93495a0c6069f9b32deb78d7c8c3ba24e53e0b3e53e'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(main, output):
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and any(x in command for x in
            ('scripts.run_image_smaug_v7', 'scripts.run_audio_separation_v14', 'scripts.repeat_audio_separation_v14',
             'scripts.run_writing_modernbert_v4', 'scripts.run_video_demamba_v13')):
            raise ValueError('Heavy native/runtime studies run sequentially')
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((root / 'docs/evaluation-panels/video-v15-sface-native-freeze.json').read_text())
    if any(digest(root / name) != expected for name, expected in frozen['implementation_sha256'].items()):
        raise ValueError('Prospectively frozen source or attribution changed before native runtime')
    output.mkdir(exist_ok=False)
    weight = main / 'models/research/video-v15-sface/sface.onnx'
    source = main / 'backend/data/independent-evaluation/video-v15-sface-source/opencv-contract/face_recognize.cpp'
    if digest(weight) != WEIGHT_SHA or digest(source) != CPP_SHA:
        raise ValueError('Exact reviewed weights and upstream contract required')
    cv2.setNumThreads(1)
    recognizer = cv2.FaceRecognizerSF.create(str(weight), '', cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    net = cv2.dnn.readNetFromONNX(str(weight))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    records = []
    for seed in range(8):
        rng = np.random.default_rng(20261107 + seed)
        rgb = rng.integers(0, 256, size=(180, 240, 3), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        for scale, theta, tx, ty in ((1., 0., 0., 0.), (1.2, .1, 16., 8.), (.8, -.2, 35., 18.)):
            rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points = (TARGET.astype(np.float64) @ rotation.T * scale + [tx, ty]).astype(np.float32)
            actual = recognizer.alignCrop(bgr, face_row(points))
            matrix = cpp_alignment(points)
            separate = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
            blob = manual_blob(actual)
            native_blob = cv2.dnn.blobFromImage(actual, 1., (112, 112), (0., 0., 0.), True, False)
            feature = recognizer.feature(actual).copy()
            repeated = recognizer.feature(actual).copy()
            net.setInput(blob)
            graph = net.forward().copy()
            if feature.shape != graph.shape or feature.ndim != 2 or feature.shape[0] != 1:
                raise ValueError('One same-shaped native/direct feature vector required')
            norm = float(np.linalg.norm(feature))
            maximum = float(np.max(np.abs(feature - graph)))
            relative = maximum / max(float(np.max(np.abs(feature))), 1e-8)
            record = {'fixture_seed': seed, 'transform': [scale, theta, tx, ty],
                'source_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                'landmarks_float32_hex': [[float(x).hex() for x in row] for row in points],
                'matrix_float64_hex': [[float(x).hex() for x in row] for row in matrix],
                'native_crop_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                'separate_crop_sha256': hashlib.sha256(separate.tobytes()).hexdigest(),
                'crop_exact': bool(np.array_equal(actual, separate)),
                'crop_differing_channel_values': int(np.count_nonzero(actual != separate)),
                'blob_exact': bool(np.array_equal(blob, native_blob)), 'feature_shape': list(feature.shape),
                'input_blob_sha256': hashlib.sha256(blob.tobytes()).hexdigest(),
                'feature_finite_nonzero': bool(np.isfinite(feature).all() and norm > 0),
                'native_repeat_exact': bool(np.array_equal(feature, repeated)),
                'max_graph_difference': maximum, 'relative_graph_difference': relative,
                'feature_sha256': hashlib.sha256(feature.tobytes()).hexdigest()}
            records.append(record)
            print(len(records), '/24', record['crop_exact'], record['blob_exact'], maximum, flush=True)
    result = {'schema': 'chipmunk.sface-synthetic-runtime/15', 'weight_sha256': WEIGHT_SHA, 'upstream_cpp_sha256': CPP_SHA,
        'opencv': cv2.__version__, 'cases': records, 'project_frames_scored': 0,
        'fixed_requirements': 'Every crop and BGR/RGB blob byte exact; finite nonzero same-shape features; repeat exact; graph max absolute<=1e-4 and relative<=1e-5.',
        'passed': all(r['crop_exact'] and r['blob_exact'] and r['feature_finite_nonzero'] and r['native_repeat_exact']
            and r['max_graph_difference'] <= 1e-4 and r['relative_graph_difference'] <= 1e-5 for r in records),
        'scope': '24 self-authored random-color/landmark fixtures. Separate outer alignment/blob transcription versus native OpenCV FaceRecognizerSF, then same OpenCV DNN graph; not an alternate inference engine, a face-swap benchmark, identity verification, calibrated probability or outside human review.',
        'serving_changed': False}
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if not result['passed']:
        raise ValueError('Original native synthetic requirements failed; preserve full adverse result')


def profile(main, output):
    profile_path = output.with_name(output.name + '-profile')
    profile_path.mkdir(exist_ok=False)
    begin, maximum, guard, errors, tree = time.monotonic(), 0, None, 0, []
    with (profile_path / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.audit_sface_native_v15', str(main), str(output)],
                                 stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                tree = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                errors += 1
                rss = 0
            if rss > 2 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2 or time.monotonic() - begin > 180:
                guard = 'memory_or_timeout'
                for p in reversed(tree):
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        errors += 1
                break
            time.sleep(.1)
        try:
            code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
    record = {'child_exit_code': code, 'guard': guard, 'sample_errors': errors,
              'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - begin}
    with (profile_path / 'profile.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record), flush=True)
    if code or guard:
        raise RuntimeError('Original synthetic runtime failure retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.profile:
        profile(args.main, args.output)
    else:
        audit(args.main, args.output)
