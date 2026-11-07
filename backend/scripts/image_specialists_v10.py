"""Source-bound image/region/compression calibration development, never serving."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path

from app.bench import specialists as engine
from scripts import specialist_calibration as fitting

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ('tgif-pairs-v2', 'tgif-confirmation-v3', 'tgif-confirmation-v4', 'tgif-confirmation-v5')
ARMS = {'global-image': (False, False), 'condition-image': (True, False),
    'global-regions': (False, True), 'condition-regions': (True, True)}
METHOD = {'schema': 'chipmunk.image-specialists-method/10', 'source_groups': 240,
    'roles': list(engine.ROLES), 'groups_per_role': 60, 'minimum_groups_per_class': 25,
    'selection': 'All240 consumed paired sources from the four declared manifests; no fresh-confirmation claim. Hash-order within three edit families, round-robin roles; exact source or original-byte duplicates across roles refused.',
    'units': 'Whole-image95th percentile and mean native response in four deterministic source-coordinate quadrants. Labels are actual retained edit-mask intersection, not copied file truth.',
    'projection': 'Nearest-neighbor native384-square float32 map to stored-source width/height; no EXIF orientation or resampling of truth.',
    'contexts': 'Actual PNG or deterministic Pillow JPEGquality75 bytes, established before detection; compression changes are not additional AI-edit truth.',
    'combination': 'Same fixed model and dependence family for whole/region readouts; positive calibrated maximum, not five independent votes. Separate fitted and calibrated file combination.',
    'device': 'cpu', 'dtype': 'float32', 'threads': 2, 'l2': .001,
    'reference_prior': .5, 'diagnostic_point': .5, 'tie_tolerance': 1e-12,
    'native_audit': 'First original/edit input per role/format repeated bitwise; compare every available original PNG response to retained144-map TIDAL journal.',
    'resource_guard': {'tree_rss_gib': 6, 'minimum_available_mib': 1536, 'minimum_disk_mib': 1536, 'maximum_seconds': 1800},
    'serving_admitted': False,
    'limits': 'Consumed COCO-derived inpainting development; external base-training overlap unknown. JPEG TIDAL is experimental: serving JPEG continues to use its current check. Regions are spatial associations, not proof of pixel origin. No full-generation, face-specific, customer confidence or court claim.'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def boxes(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < 64:
        raise ValueError('Supported stored-source geometry required')
    return [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ((0, height // 2), (height // 2, height))
        for x0, x1 in ((0, width // 2), (width // 2, width))]


def code():
    paths = {'adapter': Path(__file__), 'engine': Path(engine.__file__), 'fitting': Path(fitting.__file__),
        'model': ROOT / 'backend/app/detectors/image/splice.py',
        'source': ROOT / 'backend/scripts/localization_benchmark.py',
        'input_bytes': ROOT / 'backend/scripts/evaluate_iml_vit.py'}
    return {k: sha(v) for k, v in paths.items()}


def runtime():
    return {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'Pillow', 'opencv-python-headless', 'segmentation-models-pytorch')}


def planned_row(group, entry, condition, original, pixels, truth):
    from scripts.evaluate_iml_vit import input_bytes

    item = entry['item']
    data, rgb = input_bytes(pixels, original, condition)
    width, height = group['width'], group['height']
    units = [{'id': 'whole', 'target': 'whole', 'box': [0, 0, width, height], 'label': int(truth.any())}]
    for index, (x, y, w, h) in enumerate(group['regions']):
        units.append({'id': f'quadrant.{index}', 'target': 'region', 'box': [x, y, w, h],
            'label': int(truth[y:y + h, x:x + w].any())})
    return {'id': item['id'] + '-' + condition, 'group': group['id'], 'role': group['role'],
        'label': item['label'], 'condition': condition, 'source_sha256': item['image_sha256'],
        'input_sha256': hashlib.sha256(data).hexdigest(), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'source_geometry': {'width': width, 'height': height},
        'label_provenance_sha256': item['mask_sha256'] or group['original_sha256'],
        'lineage': [group['original_sha256']], 'units': units}


def prepare(main, trial):
    import numpy as np

    from scripts.localization_benchmark import source

    by_source, manifest_hashes = defaultdict(list), {}
    for name in MANIFESTS:
        root = main / 'backend/data/localization-eval' / name
        path = root / 'manifest.json'
        manifest_hashes[name] = sha(path)
        for item in json.loads(path.read_text())['items']:
            by_source[item['source_id']].append({'root': str(root.resolve()), 'item': item})
    if len(by_source) != METHOD['source_groups'] or any(len(v) != 2 for v in by_source.values()):
        raise ValueError('All240 exact paired sources required; duplicates cannot inflate support')
    families = defaultdict(list)
    for group, items in by_source.items():
        if {v['item']['label'] for v in items} != {0, 1} or len({v['item']['family'] for v in items}) != 1:
            raise ValueError('Original/edit source pairing required')
        families[items[0]['item']['family']].append(group)
    groups, planned, original_roles, pixel_count = [], [], {}, 0
    for family, keys in sorted(families.items()):
        keys.sort(key=lambda s: hashlib.sha256(('image-specialists-v10:' + s).encode()).hexdigest())
        for index, key in enumerate(keys):
            role = engine.ROLES[index % 4]
            items = sorted(by_source[key], key=lambda v: v['item']['label'])
            loaded = [source(Path(v['root']), v['item']) for v in items]
            original, edited = loaded[0][1], loaded[1][1]
            if original.shape != edited.shape or not np.array_equal(np.any(original != edited, axis=-1), loaded[1][2]):
                raise ValueError('Retained truth must exactly match original/edit RGB changes')
            original_hash = items[0]['item']['image_sha256']
            if original_hash in original_roles and original_roles[original_hash] != role:
                raise ValueError('Duplicate original bytes connect different planned roles')
            original_roles[original_hash] = role
            height, width = original.shape[:2]
            regions = boxes(width, height)
            pixel_count += height * width
            group = {'id': key, 'family': family, 'role': role, 'items': items,
                'width': width, 'height': height, 'regions': regions,
                'original_sha256': original_hash}
            groups.append(group)
            for entry, values in zip(items, loaded, strict=True):
                for condition in ('native', 'jpeg75'):
                    planned.append(planned_row(group, entry, condition, *values))
    groups.sort(key=lambda g: (engine.ROLES.index(g['role']), g['family'], g['id']))
    trial.mkdir(exist_ok=False)
    planned.sort(key=lambda r: (engine.ROLES.index(r['role']), r['id']))
    observed_roles = {}
    for row in planned:
        for value in (row['source_sha256'], row['input_sha256']):
            if observed_roles.setdefault(value, row['role']) != row['role']:
                raise ValueError('Identical original/derived bytes connect different planned roles')
    plan = {'schema': 'chipmunk.image-specialists-plan/10', 'method': METHOD, 'groups': groups, 'files': planned,
        'manifests_sha256': manifest_hashes, 'exact_rgb_truth_pixels_checked': pixel_count,
        'new_model_forwards': 0, 'source_data_license': 'Publisher TGIF/TGIF2 CC-BY-SA-4.0; raw data retained privately.',
        'weight_license': 'Pinned TIDAL release Apache-2.0; exact training-source membership unknown.'}
    write(trial / 'plan.json', plan)
    weight = main / 'models/weights/splice/tidal-v30.1.pt'
    if sha(weight) != 'ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061':
        raise ValueError('Exact commercial model asset required')
    write(trial / 'freeze.json', {'schema': 'chipmunk.image-specialists-freeze/10', 'method': METHOD,
        'plan_sha256': sha(trial / 'plan.json'), 'code_sha256': code(), 'runtime': runtime(),
        'weight': str(weight.resolve()), 'weight_sha256': sha(weight),
        'commercial_license_record_sha256': sha(main / 'backend/app/models.yaml'), 'main': str(main.resolve()),
        'serving_admitted': False})
    print(json.dumps({'source_groups': len(groups), 'roles': dict(Counter(g['role'] for g in groups)),
        'exact_rgb_truth_pixels_checked': pixel_count, 'freeze_sha256': sha(trial / 'freeze.json'), 'new_model_forwards': 0}), flush=True)


def check(trial):
    lock = json.loads((trial / 'freeze.json').read_text())
    if lock['method'] != METHOD or lock['code_sha256'] != code() or lock['runtime'] != runtime() or lock['plan_sha256'] != sha(trial / 'plan.json'):
        raise ValueError('Frozen source, code, method or numerical runtime changed')
    return lock, json.loads((trial / 'plan.json').read_text())


def score(trial, confirmation):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import splice
    from scripts.evaluate_iml_vit import input_bytes
    from scripts.localization_benchmark import source

    lock, plan = check(trial)
    stage = 'confirmation' if confirmation else 'training'
    path = trial / (stage + '.jsonl')
    if path.exists():
        raise FileExistsError('Retain every prior native attempt')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = splice.load_model(Path(lock['weight']))
    cache_root = Path(lock['main']) / 'backend/data/localization-eval/tgif-pairs-v2-tidal'
    cached = {r['id']: r for r in [json.loads(s) for s in (cache_root / 'predictions.jsonl').read_text().splitlines()]}
    repeated, cache_matches, maximum = set(), 0, 0.
    groups = [g for g in plan['groups'] if (g['role'] == 'confirmation') == confirmation]
    count, started = 0, time.monotonic()
    with path.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.image-specialists-responses/10', 'stage': stage,
            'freeze_sha256': sha(trial / 'freeze.json'), 'planned': len(groups) * 4}) + '\n')
        for group in groups:
            for entry in group['items']:
                item = entry['item']
                original, pixels, truth = source(Path(entry['root']), item)
                for condition in ('native', 'jpeg75'):
                    data, rgb = input_bytes(pixels, original, condition)
                    image = Image.fromarray(rgb)
                    response = splice.predict(model, image)
                    cell = group['role'], item['label'], condition
                    if cell not in repeated:
                        repeat = splice.predict(model, image)
                        if not np.array_equal(response, repeat):
                            raise ValueError('Native image repeat differs bitwise')
                        repeated.add(cell)
                    if condition == 'native' and item['id'] in cached:
                        prior = cached[item['id']]
                        cache_path = cache_root / prior['raw_map']
                        if prior['image_sha256'] != item['image_sha256'] or sha(cache_path) != prior['raw_map_sha256']:
                            raise ValueError('Retained native reference bytes changed')
                        old = np.load(cache_path, allow_pickle=False)
                        difference = float(np.max(np.abs(old - response)))
                        if old.dtype != np.float32 or old.shape != response.shape or difference != 0:
                            raise ValueError('Current native model no longer matches original PNG response')
                        maximum = max(maximum, difference)
                        cache_matches += 1
                    projected = cv2.resize(response, (group['width'], group['height']), interpolation=cv2.INTER_NEAREST)
                    units = [{'id': 'whole', 'target': 'whole', 'raw': float(np.quantile(response, .95)),
                        'box': [0, 0, group['width'], group['height']], 'label': int(truth.any())}]
                    for index, (x, y, width, height) in enumerate(group['regions']):
                        units.append({'id': f'quadrant.{index}', 'target': 'region',
                            'box': [x, y, width, height], 'raw': float(projected[y:y + height, x:x + width].mean(dtype=np.float64)),
                            'label': int(truth[y:y + height, x:x + width].any())})
                    row = {'id': item['id'] + '-' + condition, 'group': group['id'], 'role': group['role'],
                        'label': item['label'], 'condition': condition, 'source_sha256': item['image_sha256'],
                        'input_sha256': hashlib.sha256(data).hexdigest(), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                        'native_response_sha256': hashlib.sha256(response.tobytes()).hexdigest(),
                        'source_geometry': {'width': group['width'], 'height': group['height']},
                        'label_provenance_sha256': item['mask_sha256'] or group['original_sha256'],
                        'lineage': [group['original_sha256']], 'status': 'ok', 'units': units}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    count += 1
            print(json.dumps({'stage': stage, 'inputs': count, 'planned': len(groups) * 4,
                'elapsed_s': round(time.monotonic() - started, 2)}), flush=True)
    write(trial / (stage + '-native.json'), {'schema': 'chipmunk.image-specialists-native/10',
        'complete_inputs': count, 'journal_sha256': sha(path), 'bitwise_repeated_cells': len(repeated),
        'retained_png_maps_exact': cache_matches, 'maximum_native_cache_difference': maximum,
        'source_pixels_or_maps_published': False, 'serving_admitted': False})


def manifest(trial, arm, confirmation):
    lock, plan = check(trial)
    contextual, regional = ARMS[arm]
    scored = {}
    for stage in ('training', 'confirmation') if confirmation else ('training',):
        journal = trial / (stage + '.jsonl')
        native = json.loads((trial / (stage + '-native.json')).read_text())
        loaded = [json.loads(line) for line in journal.read_text().splitlines()]
        if native['journal_sha256'] != sha(journal) or native['complete_inputs'] != len(loaded) - 1:
            raise ValueError('Complete source-bound native journal required')
        scored.update({r['id']: r for r in loaded[1:]})
    route = engine.digest({'condition': 'actual PNG/JPEG bytes', 'contextual': contextual})
    targets = ('whole', 'region') if regional else ('whole',)
    specs = [{'id': 'image.' + target, 'modality': 'image', 'family': 'splice', 'target': 'image.ai_edit',
        'contexts': ['native', 'jpeg75'] if contextual else ['pooled'], 'routing_sha256': route,
        'higher_means_ai': True, 'raw_range': [0., 1.], 'producer': {'kind': 'model',
            'implementation_sha256': engine.digest(lock['code_sha256']), 'weight_sha256': [lock['weight_sha256']],
            'commercial_license_record_sha256': lock['commercial_license_record_sha256'],
            'native_receipt_sha256': sha(trial / 'training-native.json')}} for target in targets]
    files = []
    for row in plan['files']:
        geometry = row['source_geometry']
        result = scored.get(row['id'])
        if result and ({k: v for k, v in result.items() if k not in ('units', 'status', 'native_response_sha256')}
                    != {k: v for k, v in row.items() if k != 'units'}
                    or [{k: v for k, v in u.items() if k != 'raw'} for u in result['units']] != row['units']):
            raise ValueError('Native response does not match prospectively frozen bytes, geometry or local truth')
        units = []
        for index, current in enumerate(row['units']):
            if current['target'] not in targets:
                continue
            x, y, width, height = current['box']
            value = result['units'][index] if result else None
            unit = {'id': current['id'], 'specialist': 'image.' + current['target'],
                'context': row['condition'] if contextual else 'pooled', 'routing_sha256': route,
                'context_evidence_sha256': engine.digest({'condition': row['condition'], 'input_sha256': row['input_sha256']}),
                'measurement_sha256': engine.digest({'native_response_sha256': result['native_response_sha256'], 'unit': value}) if result else engine.digest(current),
                'location': {'kind': 'pixels', 'x': x, 'y': y, 'width': width, 'height': height,
                    'source_width': geometry['width'], 'source_height': geometry['height']},
                'raw': value['raw'] if value else None, 'status': 'ok' if value else 'unavailable',
                'label': current['label'], 'label_scope': 'local_unit',
                'label_provenance_sha256': engine.digest({'reference': row['label_provenance_sha256'], 'box': current['box'], 'truth': current['label']})}
            if not result:
                unit['reason'] = 'Reserved confirmation input not yet scored.'
            units.append(unit)
        files.append({k: row[k] for k in ('id', 'group', 'role', 'source_sha256', 'input_sha256', 'lineage', 'label', 'source_geometry', 'label_provenance_sha256')}
            | {'modality': 'image', 'label_scope': 'planned_units', 'required_families': ['splice'], 'units': units})
    document = {'schema': engine.SCHEMA, 'policy': {'specialists': specs}, 'files': files}
    engine.audit(document)
    return document


def fit(trial):
    for arm in ARMS:
        document = manifest(trial, arm, False)
        write(trial / f'{arm}-manifest-before-confirmation.json', document)
        frozen = fitting.freeze(document, 25, .001)
        write(trial / f'{arm}-fit-freeze.json', frozen)
        artifact = fitting.fit(document, frozen)
        write(trial / f'{arm}-artifact.json', artifact)
        if any(h['status'] != 'fitted' for h in artifact['specialists'].values()) or any(
                c['response']['status'] != 'fitted' or c.get('calibration', {}).get('status') != 'fitted'
                for c in artifact['combiners'].values()):
            raise ValueError('Support floor failed; keep artifact and do not score confirmation')
    write(trial / 'fit-complete.json', {'artifacts': {arm: sha(trial / f'{arm}-artifact.json') for arm in ARMS},
        'new_confirmation_forwards': 0, 'serving_admitted': False})


def summarize(trial):
    completion = json.loads((trial / 'fit-complete.json').read_text())
    report = {'schema': 'chipmunk.image-specialists-result/10', 'method': METHOD, 'arms': {}, 'serving_admitted': False}
    for arm in ARMS:
        if sha(trial / f'{arm}-artifact.json') != completion['artifacts'][arm]:
            raise ValueError('Confirmation must not change any fitted artifact')
        document = manifest(trial, arm, True)
        result = engine.evaluate(document, json.loads((trial / f'{arm}-artifact.json').read_text()))
        write(trial / f'{arm}-confirmation.json', result)
        cells = {'false_ai_files': 0, 'missed_ai_files': 0, 'unassessed_or_tied_files': 0}
        for row in result['outcomes']:
            p = row['probability']
            if p is None or abs(p - .5) <= 1e-12:
                cells['unassessed_or_tied_files'] += 1
            elif row['label'] == 0 and p > .5:
                cells['false_ai_files'] += 1
            elif row['label'] == 1 and p < .5:
                cells['missed_ai_files'] += 1
        report['arms'][arm] = {**result['summaries']['image'], **cells}
    write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


def profile(main, trial, confirmation):
    import shutil
    import subprocess
    import sys

    import psutil

    check(trial)
    stage = 'confirmation' if confirmation else 'training'
    folder = trial / (stage + '-profile')
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'scripts.image_specialists_v10', 'score', str(main), str(trial)]
    if confirmation:
        command.append('--confirmation')
    started, maximum, guard = time.monotonic(), 0, None
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            tree = [process]
            try:
                tree += process.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                rss = 0
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - started > 1800):
                guard = 'memory_disk_or_time'
                for member in reversed(tree):
                    with suppress(psutil.NoSuchProcess):
                        member.terminate()
                break
            time.sleep(.25)
        try:
            status = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            status = child.wait()
    receipt = {'stage': stage, 'child_exit_code': status, 'guard': guard,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'duration_s': time.monotonic() - started}
    write(folder / 'profile.json', receipt)
    print(json.dumps(receipt), flush=True)
    if status or guard:
        raise RuntimeError('Retained failed/guarded image attempt; no silent restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score', 'profile', 'fit', 'summarize'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.main, args.trial)
    elif args.stage in ('score', 'profile'):
        if args.confirmation and not (args.trial / 'fit-complete.json').exists():
            raise ValueError('Confirmation remains reserved until all fits are frozen')
        if args.stage == 'profile':
            profile(args.main, args.trial, args.confirmation)
        else:
            score(args.trial, args.confirmation)
    elif args.stage == 'fit':
        fit(args.trial)
    else:
        summarize(args.trial)
