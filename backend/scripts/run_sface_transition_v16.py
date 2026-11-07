"""Freeze/run consumed source-bound face-transition comparison; no customer inference."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from scripts import sface_transition_v16 as method

ROOT = Path(__file__).resolve().parents[2]
WEIGHT_SHA = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
CODE = ('backend/scripts/sface_transition_v16.py', 'backend/scripts/run_sface_transition_v16.py',
    'backend/scripts/review_sface_transition_v16.py', 'backend/scripts/sface_native_v15.py',
    'backend/scripts/run_image_smaug_v7.py', 'backend/tests/test_sface_transition_v16.py',
    'docs/video-v16-sface-development-protocol.md', 'backend/app/detectors/video/face_sequence.py',
    'backend/app/detectors/video/adapters/face_sequence.py', 'backend/app/detectors/video/decode.py',
    'backend/app/detectors/video/tracking.py', 'backend/app/detectors/image/faces.py',
    'backend/app/detectors/image/gend.py', 'backend/app/evidence/face_intervals.py',
    'backend/app/engine/models.py', 'backend/app/detectors/video/adapters/loaders.py',
    'backend/app/detectors/image/adapters/loaders.py', 'backend/app/engine/registry.py',
    'backend/app/core/config.py', 'backend/app/models.yaml',
    'docs/evaluation-panels/video-v15-sface-acquisition.json',
    'docs/evaluation-panels/video-v15-sface-synthetic-runtime.json',
    'docs/evaluation-panels/video-v15-sface-runtime-fingerprint.json')


def code():
    return {name: method.digest(ROOT / name) for name in CODE}


def configure(main, trial):
    from scripts.run_image_smaug_v7 import configure as base_configure

    return base_configure(main, trial / 'isolated-state')


def rows(path):
    with path.open() as stream:
        header = json.loads(next(stream))
        return header, [json.loads(line) for line in stream]


def prepare(main, trial):
    trial.mkdir(parents=True, exist_ok=False)
    configure(main, trial)
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.face_sequence import REVIEW_POINT

    items, manifests, receipts, group_sets = [], {}, {}, {}
    for version, role in (('video-v5', 'calibration'), ('video-v6', 'development')):
        folder = main / 'backend/data/independent-evaluation' / version
        manifest = folder / 'intervals/manifest.json'
        source_manifest = folder / 'sources/manifest.json'
        manifests[str(manifest.relative_to(main))] = method.digest(manifest)
        manifests[str(source_manifest.relative_to(main))] = method.digest(source_manifest)
        expected = {r['id']: r for r in json.loads(manifest.read_text())['items']}
        names = ('current-development.jsonl', 'current-test.jsonl') if role == 'calibration' else ('current-test.jsonl',)
        reference_names = ('photometric-development-reviewed-first.jsonl', 'photometric-development-reviewed-rest.jsonl') if role == 'calibration' else ('photometric-confirmation.jsonl',)
        references = {}
        for name in reference_names:
            source = main / 'backend/data/independent-evaluation/video-v6' / name
            receipts[str(source.relative_to(main))] = method.digest(source)
            _, part = rows(source)
            for row in part:
                if row['id'] in references:
                    raise ValueError('Duplicate historical current-feature reference')
                references[row['id']] = row
        for name in names:
            source = folder / name
            receipts[str(source.relative_to(main))] = method.digest(source)
            header, part = rows(source)
            if header['manifest_sha256'] != method.digest(manifest):
                raise ValueError('Legacy receipt differs from its exact source manifest')
            for row in part:
                item = expected[row['id']]
                if any(row.get(k) != value for k, value in item.items()):
                    raise ValueError('Historical legacy row changed source metadata')
                if role == 'calibration' and item['label'] != 0:
                    continue
                reference = references[item['id']]
                path = manifest.parent / item['path']
                if item['status'] != 'ready' or method.digest(path) != item['sha256'] or reference['source_sha256'] != item['sha256']:
                    raise ValueError('Exact already-consumed source bytes required')
                table = probe_frames(path, timeout_s=30).to_json()
                if table != row['table'] or table['truncated'] or table['interpolated']:
                    raise ValueError('Fresh native frame-clock table differs or has inferred/truncated timestamps')
                items.append({**item, 'role': role, 'source_path': str(path.resolve()),
                    'legacy': row, 'legacy_range_s': method.legacy_range(row),
                    'historical_current_windows': reference.get('raw_windows', []),
                    'source_manifest': str(manifest.relative_to(main))})
                group_sets.setdefault(role, set()).add(item['group'])
    if len(items) != 180 or len({r['id'] for r in items}) != 180 or group_sets['calibration'] & group_sets['development']:
        raise ValueError('Expected 60 genuine calibration plus120 source-disjoint development cases')
    old_lock = main / 'backend/data/independent-evaluation/video-v6/photometric-frozen-v2.json'
    if json.loads(old_lock.read_text())['thresholds']['raw'] != REVIEW_POINT:
        raise ValueError('Current production review point differs from validated reference')
    method.write(trial / 'panel.json', {'schema': 'chipmunk.sface-transition-panel/16',
        'source_manifests': manifests, 'historical_receipts': receipts, 'items': items,
        'role_groups': {k: sorted(v) for k, v in group_sets.items()},
        'fresh_native_clock_tables_verified': len(items),
        'freshness': 'Previously consumed v5/v6 source groups, not fresh confirmation. Source-video IDs are disjoint across roles; common subjects/pretraining overlap remain unknown.',
        'legacy_scope': 'Frozen reference frame-classifier scores/track exhibits are shared fallback. Current consecutive representation is recomputed and checked against the historical reference. This is not a fresh whole-pipeline or tracker/classifier reevaluation.'})
    print('prepared180; fresh native clocks exact; no project SFace forward', flush=True)


def freeze(main, trial):
    configure(main, trial)
    from app.core.config import settings
    from app.engine.build_info import config_sha256
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    from scripts.run_image_smaug_v7 import runtime_files

    panel = json.loads((trial / 'panel.json').read_text())
    registry = get_registry()
    models = {}
    for mid in ('yunet.2023mar', 'dinov2.small'):
        entry = registry.manifest.model(mid)
        if entry.licence.class_ != 'commercial':
            raise ValueError('Baseline/input weights require commercial permission')
        models[mid] = {'licence': entry.licence.model_dump(mode='json'),
            'files': {str(p.resolve().relative_to(main)): method.digest(p) for p in ensure_verified(entry)}}
    weight = main / 'models/research/video-v15-sface/sface.onnx'
    native = ROOT / 'docs/evaluation-panels/video-v15-sface-synthetic-runtime.json'
    native_record = json.loads(native.read_text())
    acquisition = json.loads((ROOT / 'docs/evaluation-panels/video-v15-sface-acquisition.json').read_text())
    if (method.digest(weight) != WEIGHT_SHA or native_record['weight_sha256'] != WEIGHT_SHA
            or acquisition['weight_sha256'] != WEIGHT_SHA or 'Apache' not in acquisition['license']
            or not native_record['passed'] or len(native_record['cases']) != 24):
        raise ValueError('Reviewed exact weight and completed native contract required')
    tools = {}
    for name in ('ffmpeg', 'ffprobe'):
        path = Path(shutil.which(name))
        tools[name] = {'binary_sha256': method.digest(path), 'version': subprocess.run(
            [str(path), '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]}
    lock = {'schema': 'chipmunk.sface-transition-lock/16', 'panel_sha256': method.digest(trial / 'panel.json'),
        'source_sha256': code(), 'model_assets': models, 'sface_weight_sha256': WEIGHT_SHA,
        'native_proof_sha256': method.digest(native), 'config_sha256': config_sha256(),
        'settings': {'face_min_size': settings.face_min_size, 'model_memory_budget_mb': settings.model_memory_budget_mb},
        'installed_source_sha256': runtime_files(main), 'tools': tools,
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'opencv-python-headless')},
        'planned_calibration': sum(r['role'] == 'calibration' for r in panel['items']),
        'planned_development': sum(r['role'] == 'development' for r in panel['items']),
        'project_sface_forwards_before_freeze': 0, 'fresh_confirmation_scored': False, 'serving_changed': False}
    method.write(trial / 'lock.json', lock)
    print('frozen', method.digest(trial / 'lock.json'), flush=True)


def validate(main, trial):
    lock = json.loads((trial / 'lock.json').read_text())
    from scripts.run_image_smaug_v7 import runtime_files

    if code() != lock['source_sha256'] or method.digest(trial / 'panel.json') != lock['panel_sha256']:
        raise ValueError('Frozen implementation/input packet changed')
    if runtime_files(main) != lock['installed_source_sha256']:
        raise ValueError('Frozen installed implementation changed')
    return lock, json.loads((trial / 'panel.json').read_text())


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def parity(actual, expected, point):
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected, strict=True):
        if any(a[k] != b[k] for k in ('track_id', 'start_s', 'end_s', 'centers')):
            return False
        if any(abs(a[k] - b[k]) > 2e-6 for k in ('response', 'inner_change', 'outer_change')):
            return False
    return method.selected(actual, point) == method.selected(expected, point)


def score(item, yunet, dino, recognizer, graph, min_face):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.image.gend import align_transform, warp
    from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
    from app.detectors.video.face_sequence import REVIEW_POINT, SIZE, patch_features, sequence_windows
    from app.evidence.face_intervals import bind_tracks
    from scripts.sface_native_v15 import cpp_alignment, face_row, manual_blob

    path, row = Path(item['source_path']), item['legacy']
    if method.digest(path) != item['sha256'] or probe_frames(path, timeout_s=30).to_json() != row['table']:
        raise ValueError('Source bytes/native clock changed')
    table = FrameTable.from_json(row['table'])
    centers = {p['frame_index'] for p in row['points'] if 0 < p['frame_index'] < len(table) - 1}
    scheduled = {i + k for i in centers for k in (-1, 0, 1)}
    crops, features, faces, bindings, decoded = {}, {}, {}, {}, set()
    for frame in decode_frames(path, table, sorted(scheduled), timeout_s=120):
        i, rgb = frame.index, frame.rgb
        decoded.add(i)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detected = detect_faces(bgr, detector=yunet, min_size=min_face)
        if not detected:
            continue
        face = detected[0]
        faces[i] = face
        record = {'frame_index': i, 'pts': frame.pts, 'time_base': table.time_base,
            'source_rgb_sha256': __import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
            'bbox': [face.x, face.y, face.w, face.h], 'landmarks': face.landmarks}
        transform = align_transform(face.landmarks)
        if transform is not None:
            crops[i] = cv2.resize(warp(rgb, *transform), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            record['current_crop_rgb_sha256'] = __import__('hashlib').sha256(crops[i].tobytes()).hexdigest()
            record['current_alignment_hex'] = [[float(x).hex() for x in row] for row in transform[0]]
            record['current_alignment_side'] = int(transform[1])
        try:
            detection_row = face_row(face.landmarks)
            detection_row[0, :4] = [face.x, face.y, face.w, face.h]
            detection_row[0, 14] = face.score
            native_crop = recognizer.alignCrop(bgr, detection_row)
            matrix = cpp_alignment(face.landmarks)
        except ValueError:
            record['sface_status'] = 'not_applicable_degenerate_landmarks'
            bindings[i] = record
            continue
        separate_crop = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
        blob = manual_blob(native_crop)
        native_blob = cv2.dnn.blobFromImage(native_crop, 1., (112, 112), (0., 0., 0.), True, False)
        vector = recognizer.feature(native_crop).copy()
        repeated = recognizer.feature(native_crop).copy()
        graph.setInput(blob)
        direct = graph.forward().copy()
        if vector.shape != (1, 128) or direct.shape != vector.shape or not np.isfinite(vector).all():
            raise ValueError('Invalid native face-feature shape/values')
        method.unit(vector)
        error = float(np.max(np.abs(vector - direct)))
        relative = error / max(float(np.max(np.abs(vector))), 1e-8)
        native_ok = (np.array_equal(native_crop, separate_crop) and np.array_equal(blob, native_blob)
            and np.array_equal(vector, repeated) and error <= 1e-4 and relative <= 1e-5)
        features[i] = vector.reshape(-1).copy()
        record.update(sface_status='ok', native_alignment_exact=bool(np.array_equal(native_crop, separate_crop)),
            native_input_blob_exact=bool(np.array_equal(blob, native_blob)), native_replay_passed=bool(native_ok),
            native_repeat_exact=bool(np.array_equal(vector, repeated)), native_graph_error=error,
            native_graph_relative_error=relative, sface_alignment_hex=[[float(x).hex() for x in row] for row in matrix],
            sface_crop_bgr_sha256=__import__('hashlib').sha256(native_crop.tobytes()).hexdigest(),
            sface_blob_sha256=__import__('hashlib').sha256(blob.tobytes()).hexdigest(),
            sface_feature_sha256=__import__('hashlib').sha256(vector.tobytes()).hexdigest(),
            private_sface_feature=features[i].tolist())
        bindings[i] = record
    if decoded != scheduled:
        raise ValueError('A scheduled native source frame is missing')
    observed = {'kind': 'regions', 'created_by': 'vid.faces.yunet', 'regions': [
        {'region': {'kind': 'bbox', 'frame_pts': round(table.seconds(i) * 1000),
                    'bbox': {k: getattr(face, k) for k in ('x', 'y', 'w', 'h')}}} for i, face in faces.items()]}
    exhibits = [observed, *row['exhibits']]
    current_tracks = bind_tracks([{'frame_index': i, 'score': 0.} for i in crops], exhibits, table)
    candidate_tracks = bind_tracks([{'frame_index': i, 'score': 0.} for i in features], exhibits, table)
    current = sequence_windows(centers, current_tracks, patch_features(crops, dino), table)
    candidate = method.windows(centers, candidate_tracks, features, table)
    return {'current_windows': current, 'candidate_windows': candidate,
        'baseline_parity_passed': parity(current, item['historical_current_windows'], REVIEW_POINT),
        'native_alignment_all_exact': all(r.get('native_replay_passed', True) for r in bindings.values()),
        'scheduled_frames': len(scheduled), 'decoded_frames': len(decoded), 'current_face_frames': len(crops),
        'candidate_face_frames': len(features), 'frame_inputs': [bindings[i] for i in sorted(bindings)],
        'status': 'ok' if candidate else 'not_applicable'}


def run(main, trial, role):
    settings = configure(main, trial)
    lock, panel = validate(main, trial)
    import cv2
    import torch

    from app.detectors.video.face_sequence import REVIEW_POINT
    from app.engine.build_info import config_sha256
    from app.engine.models import get_model_manager

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if not torch.backends.mps.is_available() or config_sha256() != lock['config_sha256']:
        raise ValueError('Frozen local MPS baseline/configuration required')
    for entry in lock['model_assets'].values():
        if any(method.digest(main / name) != expected for name, expected in entry['files'].items()):
            raise ValueError('Pinned baseline/input weights changed')
    weight = main / 'models/research/video-v15-sface/sface.onnx'
    if method.digest(weight) != lock['sface_weight_sha256']:
        raise ValueError('Reviewed candidate weights changed')
    if role == 'development':
        prior = [json.loads(x) for x in (trial / 'calibration.jsonl').read_text().splitlines()]
        if method.operating_point(panel['items'], prior, REVIEW_POINT) != json.loads((trial / 'operating-point.json').read_text()):
            raise ValueError('Genuine-only operating point changed or calibration incomplete')
    output = trial / (role + '.jsonl')
    if output.exists():
        raise ValueError('Preserve original attempt; no overwrite/resume')
    recognizer = cv2.FaceRecognizerSF.create(str(weight), '', cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    graph = cv2.dnn.readNetFromONNX(str(weight))
    graph.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    graph.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    manager = get_model_manager()
    selected_items = [r for r in panel['items'] if r['role'] == role]
    features_folder = trial / ('private-features-' + role)
    features_folder.mkdir(exist_ok=False)
    with manager.lease(('yunet.2023mar',), prefer='cpu') as cpu, manager.lease(('dinov2.small',), prefer='mps') as accelerated:
        for index, item in enumerate(selected_items):
            begin = time.monotonic()
            record = {'id': item['id'], 'source_sha256': item['sha256']}
            try:
                value = score(item, cpu.get('yunet.2023mar').obj, accelerated.get('dinov2.small').obj,
                              recognizer, graph, settings.face_min_size)
                frame_inputs = value.pop('frame_inputs')
                packet = features_folder / (item['id'] + '.json')
                method.write(packet, {'source_sha256': item['sha256'], 'frame_inputs': frame_inputs,
                    'scope': 'Private real-face features; not approved for public distribution.'})
                record.update(value, private_feature_packet_sha256=method.digest(packet))
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__, reason=str(error))
                traceback.print_exc()
            record['seconds'] = time.monotonic() - begin
            append(output, record)
            gc.collect()
            torch.mps.empty_cache()
            print(index + 1, '/', len(selected_items), item['id'], record['status'], round(record['seconds'], 2), flush=True)
    validate(main, trial)
    if role == 'calibration':
        records = [json.loads(x) for x in output.read_text().splitlines()]
        if not all(r.get('baseline_parity_passed') and r.get('native_alignment_all_exact') for r in records):
            raise ValueError('Calibration native/current parity incomplete; no development admission')
        method.write(trial / 'operating-point.json', method.operating_point(panel['items'], records, REVIEW_POINT))
    print(role, 'completed', flush=True)


def profile(main, trial, role):
    import psutil

    modules = {'scripts.run_sface_transition_v16', 'scripts.run_image_smaug_v7', 'scripts.run_audio_separation_v14',
        'scripts.repeat_audio_separation_v14', 'scripts.run_writing_modernbert_v4', 'scripts.run_video_demamba_v13'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and any(m in command for m in modules):
            raise ValueError('Heavy evaluations must run sequentially')
    folder = trial / ('profile-' + role)
    folder.mkdir(exist_ok=False)
    begin, maximum, guard, errors, tree = time.monotonic(), 0, None, 0, []
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.run_sface_transition_v16', role, str(main), str(trial)],
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
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - begin > 7200):
                guard = 'memory_disk_or_timeout'
                for p in reversed(tree):
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        errors += 1
                break
            time.sleep(.25)
        try:
            status = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            status = child.wait()
    result = {'role': role, 'child_exit_code': status, 'guard': guard, 'sample_errors': errors,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - begin}
    method.write(folder / 'profile.json', result)
    print(json.dumps(result), flush=True)
    if status or guard:
        raise RuntimeError('Original incomplete/failed trial retained; do not restart silently')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'calibration', 'development', 'profile'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--role', choices=('calibration', 'development'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.main, args.trial)
    elif args.mode == 'freeze':
        freeze(args.main, args.trial)
    elif args.mode == 'profile':
        if args.role is None:
            parser.error('--role required')
        profile(args.main, args.trial, args.role)
    else:
        run(args.main, args.trial, args.mode)
