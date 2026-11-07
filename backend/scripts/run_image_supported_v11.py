"""Source-bound native scoring and separately calibrated actual image branches.

Research only: no application database, customer uploads, billing or serving
defaults. Run heavy role/format jobs sequentially after the identified audio and
video-preparation jobs. Every attempt and missing observation is retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.bench import specialists as engine
from scripts import image_supported_calibration_v11 as calibration
from scripts import prepare_image_supported_v11 as source

ROOT = Path(__file__).resolve().parents[2]
POLICY = {'schema': 'chipmunk.image-supported-policy/11', 'branches': source.BRANCHES,
    'arms': {arm: list(parts) for arm, parts in calibration.ARMS.items()}, 'minimum_groups_per_class': 25, 'l2': .001, 'reference_prior': .5,
    'routing': 'Original PNG/JPEG byte signature only; never use generator, truth, or response for routing.',
    'combination': 'One dependence family per supported format. Positive specialist-logit maxima within family; separate PNG/JPEG combination fitting and calibration.',
    'point': 'Same complete fusion_cal genuine source groups across all four arms. Match current localiser flagged-group count, capped atfloor(0.1*n). Strict greater than sorted genuine group maximum at rankn-budget. Minimum25 common complete groups. Never impute missing evidence.',
    'native_png': 'CPUfloat32,2threads. Retained-map parity and repeats bitwise exact; source bytes and all local coordinates bound.',
    'native_jpeg': 'MPSfloat32, batch1. Retained consumed JPEG image-head and source-projected-map parity; atol=rtol=1e-4. First new input per class repeated at the same tolerance; image-gate side must remain unchanged.',
    'source': 'All240 consumed TGIF pairs from unchanged image-v10 roles; native PNG and deterministic JPEGquality75. No new fresh source reserve.',
    'truth': 'Exact original/edit RGB change mask; local presence is mask intersection with source-coordinate quadrant. JPEG compression does not add AI truth.',
    'scope': 'Local editing/inpainting presence and existing localiser component selection. Not whole-generation coverage or a full application-verdict comparison.',
    'resource_guard': {'tree_rss_gib': 6, 'minimum_available_mib': 1536, 'minimum_disk_mib': 1536, 'maximum_seconds_per_role_format': 900},
    'serving_admitted': False}
CODE = ('backend/scripts/run_image_supported_v11.py', 'backend/scripts/prepare_image_supported_v11.py',
    'backend/scripts/replay_image_supported_v11.py', 'backend/scripts/image_supported_calibration_v11.py',
    'backend/app/bench/specialists.py', 'backend/app/detectors/image/splice.py', 'backend/app/detectors/image/fused.py',
    'backend/app/detectors/image/adapters/localisers.py', 'backend/app/detectors/image/adapters/common.py',
    'backend/app/detectors/image/localiser.py', 'backend/app/detectors/image/jpeg_bitstream.py',
    'backend/app/evidence/image_eligibility.py', 'backend/scripts/localization_benchmark.py',
    'backend/tests/test_image_supported_v11.py', 'backend/tests/test_image_supported_calibration_v11.py',
    'backend/tests/test_image_supported_native_v11.py',
    'docs/image-supported-v11-protocol.md')


def code():
    return {name: source.sha(ROOT / name) for name in CODE}


def runtime():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'torchvision',
        'timm', 'Pillow', 'opencv-python-headless', 'safetensors', 'segmentation-models-pytorch')}


def freeze(main, trial):
    plan_path = trial / 'source-plan.json'
    plan = json.loads(plan_path.read_text())
    source.require(plan['preparation_code_sha256'] == source.sha(Path(source.__file__))
        and len(plan['files']) == 960 and len(plan['groups']) == 240, 'Complete declared source input plan required')
    source.require(source.sha(main / 'backend/app/models.yaml') == plan['commercial_license_record_sha256'], 'Commercial model record changed')
    for name, expected in source.ASSETS.items():
        source.require(source.sha(main / name) == expected, 'Exact installed commercial asset bytes required')
    source.write(trial / 'freeze.json', {'schema': 'chipmunk.image-supported-freeze/11', 'policy': POLICY,
        'source_plan_sha256': source.sha(plan_path), 'code_sha256': code(), 'runtime': runtime(),
        'main': str(main.resolve()), 'assets': source.ASSETS, 'scoring_protocol_complete': True,
        'new_neural_forwards': 0, 'serving_admitted': False})
    print(json.dumps({'freeze_sha256': source.sha(trial / 'freeze.json'), 'inputs': 960, 'new_neural_forwards': 0}))


def check(trial):
    lock = json.loads((trial / 'freeze.json').read_text())
    source.require(lock['policy'] == POLICY and lock['code_sha256'] == code() and lock['runtime'] == runtime()
        and lock['source_plan_sha256'] == source.sha(trial / 'source-plan.json'), 'Frozen supported-branch source/runtime changed')
    return lock, json.loads((trial / 'source-plan.json').read_text())


def input_for(plan, row):
    entry = next(entry for group in plan['groups'] if group['id'] == row['group']
        for entry in group['items'] if entry['item']['id'] == row['item_id'])
    original, pixels, truth = source.source_item(entry)
    data, rgb = source.encoded(pixels, original, row['format'])
    source.require(hashlib.sha256(data).hexdigest() == row['input_sha256']
        and hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
        and source.route(data) == row['format'], 'Frozen actual inference input changed')
    return data, rgb, truth


def verify_map(raw, branch):
    side = source.BRANCHES[branch]['raw_side']
    source.require(raw.dtype == np.float32 and raw.shape == (side, side) and np.isfinite(raw).all()
        and raw.min() >= 0 and raw.max() <= 1, 'Invalid original model map')


def repeat_check(first, second, branch):
    a, p = first
    b, q = second
    verify_map(a, branch)
    verify_map(b, branch)
    difference = float(np.max(np.abs(a - b)))
    if branch == 'png':
        source.require(np.array_equal(a, b) and p is None and q is None, 'PNG native repeat changed bitwise')
    else:
        source.require(np.allclose(a, b, atol=1e-4, rtol=1e-4) and abs(p - q) <= 1e-4
            and (p >= .5) == (q >= .5), 'JPEG native repeat or image-gate side changed')
    return {'maximum_map_difference': difference, 'primary_difference': abs(p - q) if p is not None else None,
        'map_bitwise_exact': bool(np.array_equal(a, b))}


def score(trial, role, branch):
    import cv2
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    from app.detectors.image import fused, splice
    from scripts.replay_image_supported_v11 import replay

    lock, plan = check(trial)
    main = Path(lock['main'])
    if role == 'confirmation':
        complete = json.loads((trial / 'fit-complete.json').read_text())
        for arm in calibration.ARMS:
            for suffix, key in (('artifact', 'artifacts'), ('points', 'points')):
                source.require(source.sha(trial / f'{arm}-{suffix}.json') == complete[key][arm], 'All fitted artifacts/points must precede confirmation inference')
    name = role + '-' + branch
    journal = trial / (name + '.jsonl')
    source.require(not journal.exists(), 'Retain prior native attempts; no implicit resume')
    selected = [row for row in plan['files'] if row['role'] == role and row['format'] == branch]
    source.require(len(selected) == 120, 'Complete role/format denominator required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if branch == 'png':
        model = splice.load_model(main / 'models/weights/splice/tidal-v30.1.pt')
    else:
        source.require(torch.backends.mps.is_available(), 'The frozen native JPEG contract requires local MPS')
        model = fused.load_model(main / 'models/weights/fused/fused_opensdi.pth',
            main / 'models/weights/fused/convnext.safetensors', 'mps', 'float32')
    forwards = 0

    def forward(rgb):
        nonlocal forwards
        forwards += 1
        image = Image.fromarray(rgb)
        if branch == 'png':
            return splice.predict(model, image), None
        value = model.predict(image)
        return value.raw_mask, value.image_score

    native_checks = []
    if branch == 'png':
        for label in (0, 1):
            row = min((r for r in selected if r['label'] == label and r['item_id'] in plan['png_caches']),
                key=lambda r: r['input_sha256'])
            _, rgb, _ = input_for(plan, row)
            cached = plan['png_caches'][row['item_id']]
            source.require(source.sha(Path(cached['path'])) == cached['file_sha256'], 'Retained PNG native reference changed')
            expected = np.load(cached['path'], allow_pickle=False)
            actual, repeated = forward(rgb), forward(rgb)
            source.require(np.array_equal(actual[0], expected), 'PNG native model differs from retained exact map')
            native_checks.append({'input_sha256': row['input_sha256'], 'retained_map_exact': True,
                **repeat_check(actual, repeated, branch)})
    else:
        for reference in plan['fused_native_references']['cases']:
            path = Path(reference['source_path'])
            source.require(source.sha(path) == reference['source']['image_sha256']
                and source.sha(Path(reference['projected_map_path'])) == reference['projected_map_sha256'], 'Retained JPEG native reference bytes changed')
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            actual, repeated = forward(rgb), forward(rgb)
            expected = np.load(reference['projected_map_path'], allow_pickle=False)
            projected = functional.interpolate(torch.from_numpy(actual[0].copy())[None, None], expected.shape,
                mode='bilinear', align_corners=False)[0, 0].numpy()
            difference = float(np.max(np.abs(projected - expected)))
            source.require(np.allclose(projected, expected, atol=1e-4, rtol=1e-4)
                and abs(actual[1] - reference['score']) <= 1e-4
                and (actual[1] >= .5) == (reference['score'] >= .5), 'JPEG retained native map/head/gate parity failed')
            native_checks.append({'input_sha256': reference['source']['image_sha256'],
                'retained_source_projection_max_difference': difference,
                'retained_primary_difference': abs(actual[1] - reference['score']),
                **repeat_check(actual, repeated, branch)})
    map_dir = trial / (name + '-maps')
    map_dir.mkdir(exist_ok=False)
    completed, data_repeats = set(), set()
    with journal.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.image-supported-native-responses/11',
            'role': role, 'format': branch, 'freeze_sha256': source.sha(trial / 'freeze.json'), 'planned': 120}) + '\n')
        for row in selected:
            forwards_before = forwards
            record = {'id': row['id'], 'role': role, 'format': branch, 'input_sha256': row['input_sha256'],
                'rgb_sha256': row['rgb_sha256'], 'status': 'failed'}
            try:
                data, rgb, truth = input_for(plan, row)
                cached = plan['png_caches'].get(row['item_id']) if branch == 'png' else None
                if cached:
                    source.require(source.sha(Path(cached['path'])) == cached['file_sha256'], 'Retained PNG map changed')
                    raw, primary = np.load(cached['path'], allow_pickle=False), None
                    map_path, map_sha = cached['path'], cached['file_sha256']
                else:
                    raw, primary = forward(rgb)
                    if row['label'] not in data_repeats:
                        data_repeats.add(row['label'])
                        record['data_repeat'] = repeat_check((raw, primary), forward(rgb), branch)
                    map_path = map_dir / (hashlib.sha256(row['id'].encode()).hexdigest() + '.npy.gz')
                    with map_path.open('xb') as target, gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0, compresslevel=6) as compressed:
                        np.save(compressed, raw, allow_pickle=False)
                    map_sha = source.sha(map_path)
                    map_path = str(map_path.resolve())
                verify_map(raw, branch)
                signal, mask = replay(data, rgb, branch, raw, primary=primary,
                    boxes=[u['box'] for u in row['units'] if u['target'] == 'region'])
                from scripts.localization_benchmark import mask_metrics

                signal['postprocessing_neural_forwards'] = signal.pop('new_neural_forwards')
                signal['postprocessing_weights_loaded'] = signal.pop('weights_loaded')
                record.update(signal, status='ok', raw_map_path=str(map_path), raw_map_file_sha256=map_sha,
                    native_inference_forwards=forwards - forwards_before,
                    reused_retained_map=cached is not None, current_mask_metrics=mask_metrics(mask, truth),
                    current_source_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                completed.add(row['id'])
                if len(completed) % 20 == 0:
                    print(json.dumps({'role': role, 'format': branch, 'completed_inputs': len(completed),
                        'planned_inputs': len(selected), 'new_neural_forwards': forwards}), flush=True)
            except BaseException as error:
                record.update(status='failed', error_type=type(error).__name__, native_inference_forwards=forwards - forwards_before)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                raise
    source.require(completed == {r['id'] for r in selected} and data_repeats == {0, 1}, 'Every planned input and new-input class repeat must complete')
    source.write(trial / (name + '-native.json'), {'schema': 'chipmunk.image-supported-role-native/11',
        'role': role, 'format': branch, 'freeze_sha256': source.sha(trial / 'freeze.json'),
        'journal_sha256': source.sha(journal), 'complete_inputs': len(completed), 'new_neural_forwards': forwards,
        'model_weights_loaded': True,
        'native_reference_checks': native_checks, 'new_input_class_repeats': len(data_repeats), 'serving_admitted': False})
    print(json.dumps({'role': role, 'format': branch, 'complete_inputs': len(completed), 'new_neural_forwards': forwards}), flush=True)


def responses(trial, confirmation):
    records, receipts = {}, {}
    for role in engine.ROLES if confirmation else engine.ROLES[:3]:
        for branch in calibration.FORMATS:
            name = role + '-' + branch
            path = trial / (name + '.jsonl')
            receipt_path = trial / (name + '-native.json')
            receipt = json.loads(receipt_path.read_text())
            header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
            source.require(header['freeze_sha256'] == source.sha(trial / 'freeze.json')
                and receipt['journal_sha256'] == source.sha(path) and receipt['complete_inputs'] == len(rows) == 120
                and all(row['status'] == 'ok' and row['format'] == branch and row['role'] == role for row in rows),
                'Every planned native input and actual receipt required; do not drop a failed observation')
            for row in rows:
                source.require(row['id'] not in records, 'Repeated response identity')
                records[row['id']] = row
            receipts[name] = source.sha(receipt_path)
    return records, receipts


def producers(lock, plan, training_receipt_sha):
    return {branch: {'kind': 'model', 'implementation_sha256': engine.digest(lock['code_sha256']),
        'weight_sha256': [value for name, value in source.ASSETS.items() if ('/splice/' in name) == (branch == 'png')],
        'commercial_license_record_sha256': plan['commercial_license_record_sha256'],
        'native_receipt_sha256': training_receipt_sha} for branch in calibration.FORMATS}


def fit(trial):
    lock, plan = check(trial)
    measured, receipts = responses(trial, False)
    source.write(trial / 'training-native.json', {'schema': 'chipmunk.image-supported-training-native/11',
        'freeze_sha256': source.sha(trial / 'freeze.json'), 'actual_role_receipts': receipts})
    producer = producers(lock, plan, source.sha(trial / 'training-native.json'))
    documents, artifacts, point_rows = {}, {}, {}
    metadata = {row['id']: row for row in plan['files']}
    for arm in calibration.ARMS:
        document = calibration.manifest(plan, measured, producer, arm)
        artifact = engine.train(document, min_groups_per_class=25, l2=.001)
        point_rows[arm] = [{**{k: metadata[file['id']][k] for k in ('id', 'group', 'role', 'label', 'format')},
            'probability': calibration.predict(file, artifact),
            'current_assessed': file['id'] in measured and measured[file['id']]['current_reliable_cells'] > 0,
            'current_flag': measured[file['id']]['current_region_flag']
                if file['id'] in measured and measured[file['id']]['current_reliable_cells'] > 0 else None}
            for file in document['files'] if file['role'] == 'fusion_cal']
        documents[arm], artifacts[arm] = document, artifact
    selected = {branch: calibration.matched_points(point_rows, branch) for branch in calibration.FORMATS}
    for arm in calibration.ARMS:
        source.write(trial / (arm + '-artifact.json'), artifacts[arm])
        source.write(trial / (arm + '-points.json'), {branch: selected[branch][arm] for branch in calibration.FORMATS})
    source.write(trial / 'fit-complete.json', {'schema': 'chipmunk.image-supported-fit/11',
        'freeze_sha256': source.sha(trial / 'freeze.json'),
        'artifacts': {arm: source.sha(trial / (arm + '-artifact.json')) for arm in calibration.ARMS},
        'points': {arm: source.sha(trial / (arm + '-points.json')) for arm in calibration.ARMS},
        'confirmation_neural_inference_unlocked': True, 'serving_admitted': False})
    print(json.dumps({'all_four_artifacts_and_eight_points_frozen': True, 'serving_admitted': False}), flush=True)


def metrics(rows):
    counts = Counter({'planned': len(rows), 'genuine': 0, 'ai': 0, 'assessed': 0, 'inconclusive': 0,
        'false_ai': 0, 'ai_found': 0, 'ai_not_flagged_including_inconclusive': 0, 'exact_point_ties': 0})
    losses = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        label, probability = row['label'], row.get('probability')
        counts['ai' if label else 'genuine'] += 1
        counts['assessed' if row['assessed'] else 'inconclusive'] += 1
        if label:
            counts['ai_found' if row['flag'] else 'ai_not_flagged_including_inconclusive'] += 1
        else:
            counts['false_ai'] += row['flag']
        counts['exact_point_ties'] += row.get('exact_point_tie', False)
        if probability is not None:
            losses[label][row['group']].append((probability - label)**2)
    by_class = [sum(sum(v) / len(v) for v in entries.values()) / len(entries) if entries else None for entries in losses.values()]
    return {**counts, 'source_class_balanced_brier': sum(by_class) / 2 if all(v is not None for v in by_class) else None}


def local_metrics(document, result, metadata):
    """Check local probabilities against actual local truth without filling boxes."""
    truth = {(file['id'], unit['id']): unit for file in document['files'] for unit in file['units']}
    groups = {file['id']: file['group'] for file in document['files']}
    selected = defaultdict(list)
    for row in result['unit_outcomes']:
        unit = truth[row['file_id'], row['unit_id']]
        source.require(row['location'] == unit['location'], 'Calibrated evidence escaped its exact source box')
        key = metadata[row['file_id']]['format'] + '|' + ('region' if row['unit_id'].startswith('quadrant.') else row['unit_id'])
        selected[key].append({'label': unit['label'], 'group': groups[row['file_id']],
            'assessed': row['probability'] is not None, 'probability': row['probability'], 'flag': False})
    return {key: {'planned': len(rows), 'assessed': sum(row['assessed'] for row in rows),
        'genuine_units': sum(row['label'] == 0 for row in rows), 'ai_units': sum(row['label'] == 1 for row in rows),
        'source_class_balanced_brier': metrics(rows)['source_class_balanced_brier'],
        'scope': 'Local presence within the recorded source box; not a calibrated pixel-authorship map.'}
        for key, rows in selected.items()}


def review(trial):
    lock, plan = check(trial)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    measured, _ = responses(trial, True)
    metadata = {row['id']: row for row in plan['files']}
    producer = producers(lock, plan, source.sha(trial / 'training-native.json'))
    report = {'schema': 'chipmunk.image-supported-result/11', 'policy': POLICY, 'formats': {}, 'serving_admitted': False}
    evaluated = {}
    for arm in calibration.ARMS:
        source.require(source.sha(trial / (arm + '-artifact.json')) == complete['artifacts'][arm]
            and source.sha(trial / (arm + '-points.json')) == complete['points'][arm], 'Frozen parameters/points changed after confirmation')
        artifact = json.loads((trial / (arm + '-artifact.json')).read_text())
        document = calibration.manifest(plan, measured, producer, arm)
        result = engine.evaluate(document, artifact)
        result['local_summary'] = local_metrics(document, result, metadata)
        source.write(trial / (arm + '-confirmation.json'), result)
        points = json.loads((trial / (arm + '-points.json')).read_text())
        evaluated[arm] = []
        for row in result['outcomes']:
            branch = metadata[row['id']]['format']
            point = points[branch]
            assessed = row['probability'] is not None and point['status'] == 'fitted'
            flag = bool(assessed and row['probability'] > point['threshold'])
            evaluated[arm].append({**row, 'format': branch, 'assessed': assessed, 'flag': flag,
                'exact_point_tie': bool(assessed and row['probability'] == point['threshold'])})
    for branch in calibration.FORMATS:
        original = [row for row in plan['files'] if row['role'] == 'confirmation' and row['format'] == branch]
        current = metrics([{**row, 'assessed': measured[row['id']]['current_reliable_cells'] > 0, 'probability': None,
            'flag': measured[row['id']]['current_region_flag']} for row in original])
        arms = {arm: metrics([row for row in rows if row['format'] == branch]) for arm, rows in evaluated.items()}
        baseline = arms['primary']
        for arm, actual in arms.items():
            gates = {'complete_calibrated_coverage': actual['assessed'] == actual['planned'] > 0,
                'no_current_detection_loss': actual['ai_found'] >= current['ai_found'],
                'no_additional_current_false_flags': actual['false_ai'] <= current['false_ai'],
                'no_primary_detection_loss': actual['ai_found'] >= baseline['ai_found'],
                'no_additional_primary_false_flags': actual['false_ai'] <= baseline['false_ai'],
                'no_primary_coverage_loss': actual['assessed'] >= baseline['assessed'],
                'lower_brier_than_primary': actual['source_class_balanced_brier'] is not None
                    and baseline['source_class_balanced_brier'] is not None
                    and actual['source_class_balanced_brier'] < baseline['source_class_balanced_brier']}
            actual.update(continuation_gates=gates, calibration_continuation_eligible=arm != 'primary' and all(gates.values()),
                detection_improved_over_current=actual['ai_found'] > current['ai_found'] and actual['false_ai'] <= current['false_ai'])
        report['formats'][branch] = {'current_components': current, 'arms': arms,
            'current_edited_mask_mean_pixel_f1': float(np.mean([measured[row['id']]['current_mask_metrics']['pixel_f1']
                for row in original if row['label'] == 1])),
            'current_edited_mask_mean_iou': float(np.mean([measured[row['id']]['current_mask_metrics']['iou']
                for row in original if row['label'] == 1])),
            'localization_quality_unsupported_inputs': sum(not measured[row['id']]['current_reliable_cells'] for row in original)}
    report['limits'] = 'Consumed paired development,60 source groups per reserved format. References are actual localiser components, not the complete application verdict. Regional calibration is local presence in a quadrant, not pixel authorship. No new serving probability, fresh validation or outside human review.'
    source.write(trial / 'result.json', report)
    print(json.dumps({'schema': report['schema'], 'formats': report['formats'], 'serving_admitted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'score', 'fit', 'review'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--role', choices=engine.ROLES)
    parser.add_argument('--format', choices=calibration.FORMATS)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.main, args.trial)
    elif args.stage == 'score':
        source.require(args.role is not None and args.format is not None, 'Explicit role/format required')
        score(args.trial, args.role, args.format)
    else:
        {'fit': fit, 'review': review}[args.stage](args.trial)
