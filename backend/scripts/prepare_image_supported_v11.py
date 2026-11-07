"""Prepare actual PNG/JPEG detection branches with exact local source truth.

No model, new source acquisition, fitting or inference. The previous image-v10
panel is consumed development data. A complete scoring/calibration protocol must
still be frozen before new neural evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('specialist_cal', 'fusion_fit', 'fusion_cal', 'confirmation')
BRANCHES = {'png': {'model': 'tidal.splice_v30_1', 'device': 'cpu', 'raw_side': 384},
    'jpeg': {'model': 'fused.opensdi', 'device': 'mps', 'raw_side': 512}}
ASSETS = {'models/weights/splice/tidal-v30.1.pt': 'ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061',
    'models/weights/fused/fused_opensdi.pth': '7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c',
    'models/weights/fused/convnext.safetensors': '1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def route(data):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    raise ValueError('No supported byte-signature branch; never substitute a detector')


def encoded(pixels, original, branch):
    if branch == 'png':
        require(route(original) == 'png', 'Original PNG branch requires actual PNG bytes')
        return original, pixels
    require(branch == 'jpeg', 'Unknown supported branch')
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='JPEG', quality=75)
    data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    require(route(data) == branch and rgb.shape == pixels.shape, 'Encoded input changed source coordinate geometry')
    return data, rgb


def units(mask, boxes):
    require(mask.ndim == 2 and mask.dtype == bool, 'Exact source-local Boolean truth required')
    height, width = mask.shape
    whole = [0, 0, width, height]
    rows = [{'id': name, 'target': name, 'box': whole, 'label': int(mask.any())}
        for name in ('primary', 'integrity')]
    for index, box in enumerate(boxes):
        x, y, w, h = box
        require(all(type(v) is int for v in box) and min(w, h) > 0
            and 0 <= x < x + w <= width and 0 <= y < y + h <= height, 'Local source box outside original pixels')
        rows.append({'id': f'quadrant.{index}', 'target': 'region', 'box': list(box),
            'label': int(mask[y:y + h, x:x + w].any())})
    return rows


def bind_role(roles, key, role):
    require(role in ROLES and roles.setdefault(key, role) == role, 'Original/input ancestry crosses source roles')


def source_item(entry):
    from scripts.localization_benchmark import source

    original, pixels, mask = source(Path(entry['root']), entry['item'])
    with Image.open(io.BytesIO(original)) as image:
        require(image.getexif().get(274, 1) == 1, 'Unsupported EXIF rotation; do not alter source coordinates')
        if image.mode == 'RGBA':
            require(np.asarray(image.getchannel('A')).min() == 255, 'Transparent source requires a separately specified compositing contract')
    return original, pixels, mask


def native_fused_references(main):
    data = main / 'backend/data/localization-eval'
    source_manifest, results_path = data / 'manifest.json', data / 'fused-research-200-f32/results.json'
    reference = json.loads(results_path.read_text())
    require(reference['manifest_sha256'] == sha(source_manifest)
        and reference['device'] == 'mps' and reference['dtype'] == 'float32', 'Retained FUSED native reference contract changed')
    expected = {Path(name).name: value for name, value in ASSETS.items() if '/fused/' in name}
    require(reference['weights_sha256'] == expected, 'Retained FUSED reference uses other weights')
    by_id = {row['id']: row for row in reference['rows']}
    original = json.loads(source_manifest.read_text())['items']
    cases = []
    for label in (0, 1):
        selected = None
        for item in sorted((row for row in original if row['label'] == label), key=lambda r: r['image_sha256']):
            path = data / item['image']
            require(sha(path) == item['image_sha256'], 'Retained FUSED source bytes changed')
            if route(path.read_bytes()) == 'jpeg':
                selected = item
                break
        require(selected is not None and selected['id'] in by_id, 'Consumed genuine/AI JPEG native reference required')
        row = by_id[selected['id']]
        projected = results_path.parent / (selected['id'] + '.probabilities.npy')
        values = np.load(projected, allow_pickle=False, mmap_mode='r')
        with Image.open(data / selected['image']) as image:
            require(values.dtype == np.float32 and values.shape == (image.height, image.width), 'Retained source-projected map geometry changed')
        cases.append({'source': selected, 'source_path': str((data / selected['image']).resolve()),
            'projected_map_path': str(projected.resolve()), 'projected_map_sha256': sha(projected),
            'score': row['score'], 'reference_kind': 'source-projected float32 map and original whole-image response, not a retained512native map'})
    return {'manifest_sha256': sha(source_manifest), 'results_sha256': sha(results_path),
        'selection': 'First hash-ordered genuine and AI consumed source JPEG; no score-based selection', 'cases': cases}


def prepare(main, trial):
    parent = main / 'backend/data/independent-evaluation/image-specialists-v10'
    plan_path = parent / 'plan.json'
    previous, lock = json.loads(plan_path.read_text()), json.loads((parent / 'freeze.json').read_text())
    require(sha(plan_path) == lock['plan_sha256'] and len(previous['groups']) == 240
        and len(previous['files']) == 960, 'Complete byte-pinned consumed image panel required')
    by_id = {row['id']: row for row in previous['files']}
    groups, files, lineage, verified_pixels = [], [], {}, 0
    for group in previous['groups']:
        entries = sorted(group['items'], key=lambda e: e['item']['label'])
        require([e['item']['label'] for e in entries] == [0, 1], 'Original/edit source pairing changed')
        loaded = [source_item(entry) for entry in entries]
        require(loaded[0][1].shape == loaded[1][1].shape
            and np.array_equal(np.any(loaded[0][1] != loaded[1][1], axis=-1), loaded[1][2]),
            'Declared AI truth no longer equals the retained exact source RGB change mask')
        height, width = loaded[0][1].shape[:2]
        require((width, height) == (group['width'], group['height']), 'Stored source geometry changed')
        verified_pixels += width * height
        bind_role(lineage, 'original:' + group['original_sha256'], group['role'])
        groups.append(group)
        for entry, (original, pixels, mask) in zip(entries, loaded, strict=True):
            local = units(mask, group['regions'])
            for branch, old_condition in (('png', 'native'), ('jpeg', 'jpeg75')):
                data, rgb = encoded(pixels, original, branch)
                original_id = entry['item']['id'] + '-' + old_condition
                expected = by_id[original_id]
                input_sha, rgb_sha = hashlib.sha256(data).hexdigest(), hashlib.sha256(rgb.tobytes()).hexdigest()
                require(input_sha == expected['input_sha256'] and rgb_sha == expected['rgb_sha256']
                    and route(data) == branch, 'Original exact PNG/JPEG input contract changed')
                require([u['label'] for u in local[2:]] == [u['label'] for u in expected['units'][1:]],
                    'Local labels were copied from file truth or moved between coordinates')
                bind_role(lineage, 'input:' + input_sha, group['role'])
                files.append({**{k: expected[k] for k in ('id', 'group', 'role', 'label', 'source_sha256',
                    'input_sha256', 'rgb_sha256', 'source_geometry', 'label_provenance_sha256', 'lineage')},
                    'format': branch, 'condition': old_condition, 'item_id': entry['item']['id'], 'units': local})
    cache_root = main / 'backend/data/localization-eval/tgif-pairs-v2-tidal'
    cache_path = cache_root / 'predictions.jsonl'
    native = {row['item_id']: row for row in files if row['format'] == 'png'}
    caches = {}
    for line in cache_path.read_text().splitlines():
        row = json.loads(line)
        require(row['id'] in native and row['image_sha256'] == native[row['id']]['input_sha256'], 'PNG native cache belongs to other source bytes')
        path = (cache_root / row['raw_map']).resolve()
        require(path.is_relative_to(cache_root.resolve()) and sha(path) == row['raw_map_sha256'], 'Retained native PNG map changed')
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        require(array.dtype == np.float32 and array.shape == (384, 384), 'Retained native PNG shape/dtype changed')
        require(row['id'] not in caches, 'Repeated cached PNG cannot inflate source support')
        caches[row['id']] = {'path': str(path), 'file_sha256': row['raw_map_sha256'], 'input_sha256': row['image_sha256']}
    require(len(caches) == 144, 'All144 retained consumed PNG maps must be accounted for')
    registry_path = main / 'backend/app/models.yaml'
    registry = yaml.safe_load(registry_path.read_text())
    models = {m['id']: m for m in registry['models'] if m['id'] in ('tidal.splice_v30_1', 'fused.opensdi', 'fused.convnext_xxlarge')}
    require(len(models) == 3 and all(m['licence']['class'] == 'commercial' for m in models.values()), 'Both branch weights and dependency need commercial records')
    references = native_fused_references(main)
    trial.mkdir(exist_ok=False)
    document = {'schema': 'chipmunk.image-supported-input-plan/11', 'parent_plan_sha256': sha(plan_path),
        'preparation_code_sha256': sha(Path(__file__)), 'source_data': 'Consumed240 TGIF source pairs; publisher CC-BY-SA-4.0; external base-training overlap unknown.',
        'groups': groups, 'files': files, 'branches': BRANCHES, 'assets': ASSETS,
        'commercial_models': models, 'commercial_license_record_sha256': sha(registry_path),
        'retained_png_cache_journal_sha256': sha(cache_path), 'png_caches': caches, 'fused_native_references': references,
        'exact_rgb_truth_pixels_verified': verified_pixels, 'source_roles': dict(Counter(g['role'] for g in groups)),
        'minimum_groups_per_class': 25, 'new_neural_forwards': 0, 'scoring_protocol_complete': False, 'serving_admitted': False,
        'limits': 'Input preparation only. FUSED is evaluated on JPEG and TIDAL on PNG. JPEG compression changes are ordinary processing, not new AI-mask truth. Fixed quadrants are local-presence targets, not pixel-authorship masks. Scoring/fitting/gates must be implemented and frozen before new neural evaluation.'}
    write(trial / 'source-plan.json', document)
    receipt = {'schema': 'chipmunk.image-supported-input-audit/11', 'source_plan_sha256': sha(trial / 'source-plan.json'),
        'preparation_code_sha256': sha(Path(__file__)), 'groups': len(groups), 'inputs': len(files),
        'files_by_format': dict(Counter(f['format'] for f in files)), 'source_roles': document['source_roles'],
        'local_units': sum(len(row['units']) for row in files), 'retained_png_maps': len(caches),
        'consumed_fused_native_references': len(references['cases']), 'exact_rgb_truth_pixels_verified': verified_pixels,
        'new_neural_forwards': 0, 'scoring_protocol_complete': False, 'serving_admitted': False}
    write(trial / 'input-audit.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    prepare(args.main, args.trial)
