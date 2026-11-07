"""Prospective, consumed-source development of a regional patch-drift cue.

The synthetic contract and project-source contract are separate. Only originals
set operating points; their receipt must be recorded before edited-image forwards.
Nothing here is an AI probability or a customer-facing service change.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts import audit_image_patch_drift_v8 as native
from scripts import image_patch_drift_v8 as method
from scripts.localization_benchmark import policy, sha, source

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('firefly', 'sd2', 'flux-fill')
FORMATS = ('native', 'jpeg75')
EXTRA_CODE = ('backend/scripts/run_image_patch_drift_v8.py',
    'backend/scripts/review_image_patch_drift_v8.py', 'backend/tests/test_patch_drift_development_v8.py',
    'docs/image-patch-drift-v8-development-protocol.md',
    'backend/scripts/run_image_smaug_v7.py', 'backend/scripts/image_smaug_comparison_v7.py',
    'backend/scripts/localization_benchmark.py', 'backend/scripts/score_splice_localizers.py',
    'backend/scripts/audit_fused_production_grid.py', 'backend/app/core/config.py',
    'backend/app/engine/registry.py', 'backend/app/engine/manifest.py',
    'backend/app/detectors/image/localiser.py', 'backend/app/detectors/image/jpeg_bitstream.py',
    'backend/app/detectors/image/splice.py', 'backend/app/evidence/image_packets.py',
    'backend/app/models.yaml')


def implementations():
    names = set(native.CODE + EXTRA_CODE)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'backend/app').rglob('*')
        if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.npy', '.j2', '.css'})
    return {name: method.digest(ROOT / name) for name in sorted(names)}


def paths(main):
    return {'calibration': main / 'backend/data/localization-eval/tgif-pairs-v2',
        'development': main / 'backend/data/independent-evaluation/image-fused-crop-v6/confirmation',
        'baseline': main / 'backend/data/independent-evaluation/image-fused-crop-v6/confirmation-results'}


def variant(rgb, original, condition):
    if condition == 'native':
        return original, rgb
    if condition != 'jpeg75':
        raise ValueError('Only native PNG and whole-image JPEG75 are frozen')
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='JPEG', quality=75)
    data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        return data, np.asarray(image.convert('RGB')).copy()


def choose(originals, development):
    chosen = []
    for family in FAMILIES:
        rows = [r for r in originals if r['label'] == 0 and r['family'] == family]
        rows.sort(key=lambda r: sha(('patch-drift-v8-calibration:' + r['source_id']).encode()))
        if len(rows) != 24:
            raise ValueError('Expected the complete consumed 72-original panel')
        chosen.extend(rows[:20])
    if len({r['source_id'] for r in chosen}) != 60:
        raise ValueError('Genuine calibration sources must be globally unique')
    if {r['source_id'] for r in chosen} & {r['source_id'] for r in development}:
        raise ValueError('Calibration/development source overlap')
    if Counter((r['family'], r['label']) for r in development) != Counter(
            {(f, label): 16 for f in FAMILIES for label in (0, 1)}):
        raise ValueError('Complete consumed 48-pair development panel required')
    if len({(r['source_id'], r['label']) for r in development}) != 96:
        raise ValueError('Development pair identities must be unique')
    return chosen


def probe_keys(items):
    result = []
    for family in FAMILIES:
        for label in (0, 1):
            item = min((r for r in items if r['family'] == family and r['label'] == label),
                key=lambda r: sha(('patch-drift-v8-native:' + r['id']).encode()))
            result.extend((item['id'], condition) for condition in FORMATS)
    return set(result)


def baseline(main, item, condition, data, rgb, truth, old):
    from app.detectors.image import splice
    from scripts.score_splice_localizers import score_map

    if old['status'] != 'ok' or old['input_sha256'] != sha(data):
        raise ValueError('Old current-check input bytes/status differ')
    for key in ('id', 'source_id', 'family', 'label', 'image_sha256', 'mask_sha256'):
        if old[key] != item[key]:
            raise ValueError('Old current-check source identity differs')
    path = paths(main)['baseline'] / old['arrays']
    if method.digest(path) != old['arrays_sha256']:
        raise ValueError('Old native scalar response changed')
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays['baseline'].copy()
    if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Current scalar response is invalid')
    if condition == 'native':
        eligible = int((values > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
        point = splice.PIXEL_THRESHOLD
    else:
        eligible, point = old['baseline_image_score'] >= .5, .5
    measured = score_map(values, eligible, point, data, rgb, truth)['metrics']
    if measured != old['baseline']:
        raise ValueError('Replayed current map/reliability/display pixels differ')
    return measured


def prepare(main, trial):
    native.configure(main, trial)
    lock = json.loads((trial / 'native-freeze.json').read_text())
    passed = json.loads((trial / 'synthetic-native.json').read_text())
    if (passed.get('passed') is not True or len(passed['cases']) != 6
            or passed['freeze_sha256'] != method.digest(trial / 'native-freeze.json')
            or lock['source_sha256'] != native.source_code()):
        raise ValueError('Unchanged, successful synthetic contract required before project scoring')
    roots = paths(main)
    original = json.loads((roots['calibration'] / 'manifest.json').read_text())['items']
    development = json.loads((roots['development'] / 'manifest.json').read_text())['items']
    calibration = choose(original, development)
    old_path = roots['baseline'] / 'predictions.jsonl'
    old_rows = [json.loads(line) for line in old_path.read_text().splitlines()]
    old = {(r['id'], r['condition']): r for r in old_rows}
    if len(old) != 192 or len(old_rows) != 192:
        raise ValueError('Complete original 192-input baseline receipt required')
    items = []
    probes = probe_keys(development)
    for role, rows in (('calibration', calibration), ('development', development)):
        for item in rows:
            original_bytes, rgb, truth = source(roots[role], item)
            for condition in FORMATS:
                data, pixels = variant(rgb, original_bytes, condition)
                entry = {**item, 'role': role, 'condition': condition, 'input_sha256': sha(data),
                    'input_rgb_sha256': sha(pixels.tobytes()), 'geometry': list(pixels.shape),
                    'native_probe': role == 'development' and (item['id'], condition) in probes}
                if role == 'development':
                    receipt = old[item['id'], condition]
                    entry['baseline'] = baseline(main, item, condition, data, pixels, truth, receipt)
                    entry['baseline_arrays'] = receipt['arrays']
                    entry['baseline_arrays_sha256'] = receipt['arrays_sha256']
                items.append(entry)
    native.write(trial / 'development-panel.json', {'schema': 'chipmunk.patch-drift-panel/8', 'items': items})
    native.write(trial / 'development-freeze.json', {'schema': 'chipmunk.patch-drift-development-freeze/8',
        'implementation_sha256': implementations(), 'native_freeze_sha256': method.digest(trial / 'native-freeze.json'),
        'synthetic_native_sha256': method.digest(trial / 'synthetic-native.json'),
        'panel_sha256': method.digest(trial / 'development-panel.json'),
        'baseline_predictions_sha256': method.digest(old_path),
        'source_manifest_sha256': {role: method.digest(roots[role] / 'manifest.json')
            for role in ('calibration', 'development')},
        'planned_counts': dict(Counter(r['role'] + ':' + r['condition'] for r in items)),
        'native_probe_count': len(probes), 'component_policy': policy(),
        'operating_point_rule': 'strict greater than second-largest eligible genuine per-image maximum; positive point; at least48/60 eligible each format',
        'arm_priority': list(method.ARMS), 'minimum_added_overlapping_edits': 4,
        'maximum_added_genuine_highlights': 1, 'minimum_family_overlap_delta': 0,
        'minimum_mean_edit_iou_delta': 0, 'project_response_count_before_freeze': 0,
        'serving_changed': False, 'overall_confidence_weight': 0})
    print(json.dumps({'freeze_sha256': method.digest(trial / 'development-freeze.json'),
        'cases': len(items), 'baseline_component_replays_passed': 192,
        'calibration_groups': len(calibration), 'development_groups': len(development) // 2}), flush=True)


def check(main, trial, expected):
    from scripts.run_image_smaug_v7 import runtime_files

    native.configure(main, trial)
    if method.digest(trial / 'development-freeze.json') != expected:
        raise ValueError('Independently recorded development freeze hash required')
    lock = json.loads((trial / 'development-freeze.json').read_text())
    if implementations() != lock['implementation_sha256']:
        raise ValueError('Frozen development source changed')
    if method.digest(trial / 'development-panel.json') != lock['panel_sha256']:
        raise ValueError('Frozen source panel changed')
    if method.digest(trial / 'native-freeze.json') != lock['native_freeze_sha256']:
        raise ValueError('Native contract changed')
    original_lock = json.loads((trial / 'native-freeze.json').read_text())
    if runtime_files(main) != original_lock['installed_source_sha256']:
        raise ValueError('Installed runtime changed')
    for asset in original_lock['assets'].values():
        if any(method.digest(main / n) != value for n, value in asset['files'].items()):
            raise ValueError('Native architecture/checkpoint changed')
    if method.digest(trial / 'synthetic-native.json') != lock['synthetic_native_sha256']:
        raise ValueError('Synthetic native result changed')
    for role, value in lock['source_manifest_sha256'].items():
        if method.digest(paths(main)[role] / 'manifest.json') != value:
            raise ValueError('Original source manifest changed')
    return lock, json.loads((trial / 'development-panel.json').read_text())['items']


def operating_point(maxima):
    values = [float(v) for v in maxima if v is not None]
    if len(values) < 48 or len(maxima) != 60 or not all(np.isfinite(values)):
        raise ValueError('At least48/60 complete eligible genuine maxima required')
    point = sorted(values, reverse=True)[1]
    if point <= 0:
        raise ValueError('A positive genuine-only operating point is required')
    return point


def run(main, trial, expected, role, points_sha=None):
    import torch

    lock, panel = check(main, trial, expected)
    points = None
    if role == 'development':
        if not points_sha or method.digest(trial / 'operating-points.json') != points_sha:
            raise ValueError('Record the genuine-only points before any edited-image forward')
        points = json.loads((trial / 'operating-points.json').read_text())
        if points['freeze_sha256'] != expected or not points['passed']:
            raise ValueError('Passed genuine-only operating-point receipt required')
        if points['calibration_predictions_sha256'] != method.digest(trial / 'calibration/predictions.jsonl'):
            raise ValueError('Calibration records changed')
    folder = trial / role
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    model = method.load(main)
    rows = [r for r in panel if r['role'] == role]
    with (folder / 'predictions.jsonl').open('x') as log:
        for index, item in enumerate(rows):
            started = time.monotonic()
            original, rgb, truth = source(paths(main)[role], item)
            data, pixels = variant(rgb, original, item['condition'])
            if sha(data) != item['input_sha256'] or sha(pixels.tobytes()) != item['input_rgb_sha256']:
                raise ValueError('Frozen project input bytes changed')
            row = {k: item[k] for k in ('id', 'source_id', 'family', 'label', 'role', 'condition',
                'input_sha256', 'input_rgb_sha256', 'geometry', 'native_probe')}
            row.update(status='failed', freeze_sha256=expected)
            try:
                arrays, proof = method.infer(model, pixels, native_reference=item['native_probe'],
                    retain_features=item['native_probe'])
                row.update(native=proof, arms={})
                for arm in method.ARMS:
                    floating, reliable = method.screening(method.scores(arrays['filtered'], arm), pixels, data)
                    arrays[arm + '_screening'] = floating
                    arrays['reliable'] = reliable
                    result = {'eligible_maximum': float(floating[reliable].max()) if reliable.any() else None}
                    if points is not None:
                        point = points['points'][item['condition']][arm]
                        metrics, grids, facts = method.predict_regions(floating, reliable, point,
                            pixels.shape[1], pixels.shape[0], truth)
                        arrays.update({arm + '_' + name: value for name, value in grids.items() if name != 'reliable'})
                        result.update(metrics=metrics, component_facts=facts)
                    row['arms'][arm] = result
                if role == 'development':
                    row['baseline'] = item['baseline']
                output = folder / f'arrays-{index:03d}.npz'
                np.savez_compressed(output, **arrays)
                row.update(status='ok', arrays=output.name, arrays_sha256=method.digest(output))
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error)[:600])
            row['duration_s'] = time.monotonic() - started
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps({'role': role, 'completed': index + 1, 'planned': len(rows), 'status': row['status']}), flush=True)
    if implementations() != lock['implementation_sha256']:
        raise ValueError('Frozen source changed during project inference')


def fit(main, trial, expected):
    _, panel = check(main, trial, expected)
    rows = [json.loads(line) for line in (trial / 'calibration/predictions.jsonl').read_text().splitlines()]
    planned = {(r['id'], r['condition']) for r in panel if r['role'] == 'calibration'}
    if len(rows) != 120 or {(r['id'], r['condition']) for r in rows} != planned or any(
            r['status'] != 'ok' or r['label'] != 0 or r['freeze_sha256'] != expected for r in rows):
        raise ValueError('All120 frozen genuine-only forwards must complete before fitting')
    points, counts = {}, {}
    for condition in FORMATS:
        selected = [r for r in rows if r['condition'] == condition]
        points[condition] = {arm: operating_point([r['arms'][arm]['eligible_maximum'] for r in selected])
            for arm in method.ARMS}
        counts[condition] = {arm: sum(r['arms'][arm]['eligible_maximum'] is not None
            and r['arms'][arm]['eligible_maximum'] > points[condition][arm] for r in selected) for arm in method.ARMS}
    native.write(trial / 'operating-points.json', {'schema': 'chipmunk.patch-drift-operating-points/8',
        'freeze_sha256': expected, 'calibration_predictions_sha256': method.digest(trial / 'calibration/predictions.jsonl'),
        'points': points, 'genuine_maxima_above_point': counts, 'passed': True,
        'edited_images_used': 0, 'not_ai_probabilities': True})
    print(json.dumps({'points_sha256': method.digest(trial / 'operating-points.json'), 'points': points,
        'genuine_maxima_above_point': counts}), flush=True)


def profile(main, trial, expected, role, points_sha=None):
    import psutil

    folder = trial / (role + '-profile')
    folder.mkdir(exist_ok=False)
    start, maximum, guard, errors = time.monotonic(), 0, None, 0
    command = [sys.executable, '-m', 'scripts.run_image_patch_drift_v8', role, str(main), str(trial),
        '--freeze-sha256', expected]
    if points_sha:
        command.extend(['--points-sha256', points_sha])
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            tree, rss = [process], 0
            try:
                tree += process.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                errors += 1
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - start > 1800):
                guard = 'memory_disk_or_time'
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
    receipt = {'role': role, 'child_exit_code': status, 'guard': guard, 'sample_errors': errors,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - start}
    native.write(folder / 'profile.json', receipt)
    print(json.dumps(receipt), flush=True)
    if status or guard:
        raise RuntimeError('Retain original failed/guarded study; no silent restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'calibration', 'fit', 'development',
        'profile-calibration', 'profile-development'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--points-sha256')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.main, args.trial)
    elif not args.freeze_sha256:
        parser.error('Independently recorded development freeze SHA256 required')
    elif args.stage == 'fit':
        fit(args.main, args.trial, args.freeze_sha256)
    elif args.stage.startswith('profile-'):
        profile(args.main, args.trial, args.freeze_sha256, args.stage.removeprefix('profile-'), args.points_sha256)
    else:
        run(args.main, args.trial, args.freeze_sha256, args.stage, args.points_sha256)
