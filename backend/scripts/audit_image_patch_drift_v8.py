"""Freeze and inspect native patch-drift behavior on authored images only."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts import image_patch_drift_v8 as method

ROOT = Path(__file__).resolve().parents[2]
CODE = ('backend/scripts/image_patch_drift_v8.py', 'backend/scripts/audit_image_patch_drift_v8.py',
    'backend/tests/test_image_patch_drift_v8.py', 'docs/image-patch-drift-v8-native-protocol.md',
    'docs/third-party/TRAIL-v8-MIT.txt', 'backend/app/detectors/image/radar.py', 'backend/app/engine/fetchers.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def configure(main, trial):
    from scripts.run_image_smaug_v7 import configure as isolated

    return isolated(main, trial / 'isolated-state')


def source_code():
    return {n: method.digest(ROOT / n) for n in CODE}


def freeze(main, trial):
    configure(main, trial)
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    from scripts.run_image_smaug_v7 import runtime_files

    reg, assets = get_registry(), {}
    for mid in (method.MODEL_ID, method.ARCHITECTURE_ID):
        model = reg.manifest.model(mid)
        if model.licence.class_ != 'commercial':
            raise ValueError('Exact architecture and weights require commercial admission')
        assets[mid] = {'license': model.licence.model_dump(mode='json'),
            'files': {str(p.relative_to(main)): method.digest(p) for p in ensure_verified(model)}}
    reference = main / 'backend/data/independent-evaluation/image-v8-patch-drift-source'
    objects = json.loads((reference / 'verified-source-objects.json').read_text())
    if objects['revision'] != '0fce0f1de2ac0d219baf91592ee2bca95ec0a219':
        raise ValueError('Previously reviewed primary source required')
    for row in objects['files']:
        if method.digest(reference / row['path']) != row['sha256']:
            raise ValueError('Reviewed native reference changed')
    write(trial / 'native-freeze.json', {'schema': 'chipmunk.patch-drift-native-freeze/8',
        'source_sha256': source_code(), 'reference': objects, 'assets': assets,
        'installed_source_sha256': runtime_files(main),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'Pillow')},
        'native_inputs': 6, 'cpu_normalization_absolute_limit': 5e-7, 'native_hook_feature_limit': 0.,
        'project_images_scored': 0, 'serving_changed': False})
    print('native freeze', method.digest(trial / 'native-freeze.json'), flush=True)


def inputs():
    y, x = np.mgrid[:448, :448]
    rng = np.random.default_rng(20261107)
    return [np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8),
        np.repeat((((x // 8 + y // 8) % 2) * 255)[:, :, None], 3, axis=2).astype(np.uint8),
        np.stack([((x // 32) % 2) * 255, ((y // 32) % 2) * 255, np.full_like(x, 127)], axis=-1).astype(np.uint8),
        rng.integers(0, 256, (320, 512, 3), dtype=np.uint8),
        rng.integers(0, 256, (517, 391, 3), dtype=np.uint8),
        np.broadcast_to(np.asarray([52, 120, 208], np.uint8), (448, 448, 3)).copy()]


def run(main, trial):
    configure(main, trial)
    import torch

    from scripts.run_image_smaug_v7 import runtime_files

    lock = json.loads((trial / 'native-freeze.json').read_text())
    if lock['source_sha256'] != source_code() or lock['installed_source_sha256'] != runtime_files(main):
        raise ValueError('Synthetic native contract changed after freeze')
    for asset in lock['assets'].values():
        if any(method.digest(main / n) != digest for n, digest in asset['files'].items()):
            raise ValueError('Pinned native asset changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise ValueError('Frozen local MPS device required')
    ref = main / 'backend/data/independent-evaluation/image-v8-patch-drift-source/src/trail/core.py'
    expected = next(r['sha256'] for r in lock['reference']['files'] if r['path'] == 'src/trail/core.py')
    if method.digest(ref) != expected:
        raise ValueError('Pure numerical reference changed')
    spec = importlib.util.spec_from_file_location('reviewed_trail_core_v8', ref)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = method.load(main)
    folder = trial / 'synthetic-arrays'
    folder.mkdir(exist_ok=False)
    cases = []
    for index, image in enumerate(inputs()):
        clean, changed, normalized = method.views(image)
        np.testing.assert_array_equal(changed, reference.haar_attenuate(clean, alpha=.2, levels=2))
        independent = ((torch.from_numpy(np.stack([clean, changed])) - torch.tensor([.485, .456, .406])) /
            torch.tensor([.229, .224, .225])).permute(0, 3, 1, 2).contiguous().numpy()
        error = float(np.max(np.abs(normalized - independent)))
        if error > lock['cpu_normalization_absolute_limit']:
            raise ValueError('Separate native input normalization differs')
        arrays, proof = method.infer(model, image, native_reference=True, retain_features=True)
        features = arrays['private_features']
        np.testing.assert_array_equal(arrays['raw'], reference.token_drift(features[0], features[1]).reshape(32, 32))
        np.testing.assert_array_equal(arrays['filtered'], reference.median3(arrays['raw']))
        path = folder / f'authored-{index:02}.npz'
        np.savez_compressed(path, **arrays)
        cases.append({'id': f'authored-{index:02}', 'passed': True, 'source_geometry': list(image.shape),
            'source_rgb_sha256': __import__('hashlib').sha256(image.tobytes()).hexdigest(),
            'normalization_max_difference': error, 'native': proof,
            'private_synthetic_arrays_sha256': method.digest(path)})
        print(index + 1, '/6 authored inputs passed', flush=True)
    if lock['source_sha256'] != source_code():
        raise ValueError('Frozen implementation changed during native check')
    result = {'schema': 'chipmunk.patch-drift-synthetic-native/8', 'passed': len(cases) == 6,
        'freeze_sha256': method.digest(trial / 'native-freeze.json'), 'cases': cases,
        'max_normalization_difference': max(r['normalization_max_difference'] for r in cases),
        'max_feature_difference': max(r['native']['native_hook_feature_max_difference'] for r in cases),
        'project_images_scored': 0, 'serving_changed': False, 'new_accuracy_result': False,
        'scope': 'Authored synthetic preprocessing/native-readout/reference arithmetic only; no detection quality or outside human review.'}
    write(trial / 'synthetic-native.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)


def profile(main, trial):
    import psutil

    folder = trial / 'native-profile'
    folder.mkdir(exist_ok=False)
    start, maximum, guard, errors = time.monotonic(), 0, None, 0
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.audit_image_patch_drift_v8', 'run', str(main), str(trial)],
            stdout=log, stderr=subprocess.STDOUT)
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
    receipt = {'child_exit_code': status, 'guard': guard, 'sample_errors': errors,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - start}
    write(folder / 'profile.json', receipt)
    print(json.dumps(receipt), flush=True)
    if status or guard:
        raise RuntimeError('Original failed/incomplete native attempt retained; no silent restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run', 'profile'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.main, args.trial)
