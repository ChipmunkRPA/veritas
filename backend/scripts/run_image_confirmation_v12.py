"""Frozen-parameter JPEG holdout comparison; no calibration or serving changes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from app.bench import specialists as engine
from scripts import image_supported_calibration_v11 as calibration
from scripts import prepare_image_confirmation_v12 as selection
from scripts import prepare_image_supported_v11 as source
from scripts import run_image_supported_v11 as old

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('primary', 'primary-regions')
POLICY = {'schema': 'chipmunk.image-jpeg-fresh-confirmation-policy/12',
    'scope': '48 new known-project source pairs, three existing TGIF generators, JPEGquality75 only. No unseen-generator, external pretraining-independence, pixel-authorship or full-app accuracy claim.',
    'source': 'All48 predeclared sources and96 original/edit inputs. Exclude every prior selected COCO ID, including source drafts. No response filtering, missing-case replacement or refitting.',
    'truth': 'Original RGB pair difference is primary edit mask. JPEG changes are ordinary compression, not additional AI truth. Unavailable secondary author masks are recorded.',
    'candidate': 'Exact v11 primary-regions artifact and JPEG point; exact primary-only comparator and point. Existing actual FUSED component selector is the current reference.',
    'native': 'Same cached commercial FUSED OpenSDI and ConvNeXt, MPSfloat32, batch1. Actual original JPEG bytes decoded before inference. Same source-coordinate readouts and production postprocessing.',
    'parity': 'Before new source forwards, hash-first consumed original/AI JPEG references match their retained heads and torch-bilinear projected maps at atol=rtol=1e-4 and original gate side; repeat both. Repeat first new original/edit input, same limits.',
    'retention': 'Private lossless512 float32 maps plus source/input/RGB/hash journals; no additional retained full-resolution projection arrays. Raw photos/maps remain private.',
    'missing': 'Every planned source retained. Any failed job remains partial with original log and no completed accuracy result. Unavailable required measurement has no combined score; no original verdict.',
    'gates': 'More AI files flagged than current components; no additional genuine false flags or coverage loss against current or primary-only; no AI loss against primary-only; lower Brier than primary-only. All required. No tuning or serving admission on this holdout.',
    'uncertainty': '5000 paired source-group bootstrap draws, seed20261107, percentile95% intervals. All48 sources together; retain both labels per source. Report each generator separately.',
    'resource_guard': {'tree_rss_gib': 6, 'available_mib': 1536, 'disk_mib': 1536, 'native_seconds': 900},
    'serving_admitted': False}
CODE = ('backend/scripts/run_image_confirmation_v12.py', 'backend/scripts/prepare_image_confirmation_v12.py',
    'backend/scripts/specialist_runtime_v12.py', 'backend/scripts/specialist_runtime_v4.py',
    'backend/scripts/review_image_confirmation_v12.py', 'backend/scripts/review_image_supported_v11.py',
    'backend/tests/test_image_confirmation_v12.py',
    'docs/image-fresh-v12-protocol.md', 'backend/scripts/prepare_tgif_localization.py', *old.CODE)


def read(path):
    return json.loads(Path(path).read_text())


def original_trial(main):
    return main / 'backend/data/independent-evaluation/image-supported-v11'


def models(main):
    parent = original_trial(main)
    artifacts = {a: read(parent / f'{a}-artifact.json') for a in ARMS}
    for artifact in artifacts.values():
        source.require(artifact['schema'] == 'chipmunk.specialist-calibration/2'
            and artifact['feature_pooling'] == 'positive_logit_max_per_dependency_family'
            and artifact['serving_admitted'] is False
            and artifact['implementation_sha256'] == source.sha(Path(engine.__file__))
            and artifact['packages'] == {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')}
            and artifact['sha256'] == engine.digest({k: v for k, v in artifact.items() if k != 'sha256'}),
            'Original exact compatible calibration artifact required')
    points = {a: read(parent / f'{a}-points.json')['jpeg'] for a in ARMS}
    source.require(all(p['status'] == 'fitted' for p in points.values()), 'Fixed fitted development points required')
    return artifacts, points


def license_binding(path, historical):
    """Bind today's selected records without rewriting the original producer."""
    ids = ('fused.opensdi', 'fused.convnext_xxlarge')
    rows = yaml.safe_load(path.read_text())['models']
    selected = {r['id']: r for r in rows if r['id'] in ids}
    source.require(len([r for r in rows if r['id'] in ids]) == len(ids)
        and selected == {i: historical['commercial_models'][i] for i in ids}
        and all(r['licence']['class'] == 'commercial' and r['licence']['spdx'] in ('MIT', 'Apache-2.0')
            for r in selected.values()), 'Selected commercial model source, dependency or licence changed')
    return {'historical_producer_manifest_sha256': historical['commercial_license_record_sha256'],
        'current_manifest_sha256': source.sha(path), 'selected_model_records_sha256': engine.digest(selected),
        'model_ids': list(ids), 'selected_records_equal_original': True}


def identity(main, trial):
    selected = selection.check(main, trial)
    old_lock, old_plan = old.check(original_trial(main))
    models(main)
    assets = {n: h for n, h in source.ASSETS.items() if '/fused/' in n}
    return {'schema': 'chipmunk.image-jpeg-fresh-freeze/12', 'policy': POLICY,
        'selection': selected, 'code_sha256': {n: source.sha(ROOT / n) for n in sorted(set(CODE))},
        'inputs_sha256': {n: source.sha(trial / n) for n in ('input-plan.json', 'sources/manifest.json', 'acquisition-profile/profile.json')},
        'development_freeze_sha256': source.sha(original_trial(main) / 'freeze.json'),
        'runtime': old.runtime(), 'assets': assets,
        'runtime_context': {'python': sys.version, 'machine': platform.machine(),
            'additional_packages': {n: importlib.metadata.version(n) for n in ('PyYAML', 'psutil')}},
        'commercial_license_binding': license_binding(main / 'backend/app/models.yaml', old_plan),
        'commercial_license_record_sha256': old_plan['commercial_license_record_sha256'],
        'original_native_code_sha256': old_lock['code_sha256'],
        'scoring_protocol_complete': True,
        'new_holdout_neural_forwards_before_freeze': 0, 'serving_admitted': False}


def prepare(main, trial):
    selection.check(main, trial)
    profile = read(trial / 'acquisition-profile/profile.json')
    source.require(profile['child_exit_code'] == 0 and profile['guard'] is None, 'Completed guarded source acquisition required')
    manifest = read(trial / 'sources/manifest.json')
    source.require(manifest['plan_sha256'] == source.sha(trial / 'sources/tgif-plan.json'), 'Source acquisition differs from predeclared selection')
    excluded = set(read(trial / 'source-exclusions.json')['ids'])
    previous = read(original_trial(main) / 'source-plan.json')
    previous_inputs = {r['input_sha256'] for r in previous['files']}
    files, pair_pixels = [], 0
    items = manifest['items']
    groups = sorted({r['source_id'] for r in items})
    source.require(len(items) == 96 and len(groups) == 48 and not set(groups) & excluded, 'All fresh source pairs required')
    for group in groups:
        paired = sorted([r for r in items if r['source_id'] == group], key=lambda r: r['label'])
        source.require([r['label'] for r in paired] == [0, 1] and len({r['family'] for r in paired}) == 1, 'Exact original/edit pair required')
        entries = [{'root': str((trial / 'sources').resolve()), 'item': r} for r in paired]
        loaded = [source.source_item(entry) for entry in entries]
        source.require(loaded[0][1].shape == loaded[1][1].shape
            and np.array_equal(np.any(loaded[0][1] != loaded[1][1], axis=2), loaded[1][2]), 'Exact pair RGB truth changed')
        height, width = loaded[0][1].shape[:2]
        source.require(min(width, height) >= 64, 'Unsupported source dimensions')
        boxes = [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ((0, height // 2), (height // 2, height))
            for x0, x1 in ((0, width // 2), (width // 2, width))]
        pair_pixels += width * height
        for entry, (original, pixels, truth) in zip(entries, loaded, strict=True):
            item = entry['item']
            data, rgb = source.encoded(pixels, original, 'jpeg')
            input_hash = hashlib.sha256(data).hexdigest()
            source.require(input_hash not in previous_inputs, 'Fresh encoded input duplicates earlier evaluated bytes')
            files.append({'id': item['id'] + '-jpeg75', 'group': group, 'role': 'confirmation', 'label': item['label'],
                'format': 'jpeg', 'generator': item['family'], 'entry': entry,
                'source_sha256': item['image_sha256'], 'input_sha256': input_hash,
                'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                'source_geometry': {'width': width, 'height': height},
                'label_provenance_sha256': item['mask_sha256'] or paired[0]['image_sha256'],
                'lineage': [paired[0]['image_sha256']], 'units': source.units(truth, boxes)})
    source.require(len({r['input_sha256'] for r in files}) == 96, 'Duplicate holdout input bytes')
    source.write(trial / 'input-plan.json', {'schema': 'chipmunk.image-fresh-input-plan/12', 'branches': source.BRANCHES,
        'files': sorted(files, key=lambda r: r['input_sha256']), 'paired_source_pixels_verified': pair_pixels,
        'new_neural_forwards': 0, 'serving_admitted': False})
    print(json.dumps({'source_pairs': 48, 'jpeg_inputs': 96, 'exact_pair_pixels': pair_pixels, 'new_neural_forwards': 0}), flush=True)


def check(main, trial):
    source.require(read(trial / 'native-freeze.json') == identity(main, trial), 'Frozen source, code, runtime, artifact or point changed')


def input_for(row):
    original, pixels, truth = source.source_item(row['entry'])
    data, rgb = source.encoded(pixels, original, 'jpeg')
    source.require(source.route(data) == 'jpeg' and hashlib.sha256(data).hexdigest() == row['input_sha256']
        and hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
        and source.units(truth, [u['box'] for u in row['units'] if u['target'] == 'region']) == row['units'],
        'Regenerated JPEG bytes, RGB or local truth changed')
    return data, rgb, truth


def forward(plan, responses, artifact, arm):
    """Bind fresh observations to unchanged heads without reopening their fit."""
    producers = {}
    for spec in artifact['policy']['specialists']:
        branch = spec['contexts'][0]
        source.require(branch in calibration.FORMATS
            and producers.setdefault(branch, spec['producer']) == spec['producer'], 'Inconsistent frozen measurement producer')
    document = calibration.manifest(plan, responses, producers, arm)
    source.require(document['policy'] == artifact['policy'], 'Fresh target, routing, readout or producer differs from frozen calibration')
    source.require(all(r['role'] == 'confirmation' and r['required_families'] == ['jpeg-local-edit'] for r in document['files']),
        'Fresh observations may not enter fitting or another format')
    return document


def native(main, trial):
    check(main, trial)
    import cv2
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    from app.detectors.image import fused
    from scripts.localization_benchmark import mask_metrics
    from scripts.replay_image_supported_v11 import replay

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    source.require(torch.backends.mps.is_available(), 'Frozen MPSfloat32 required')
    model = fused.load_model(main / 'models/weights/fused/fused_opensdi.pth',
        main / 'models/weights/fused/convnext.safetensors', 'mps', 'float32')
    calls = 0

    def predict(rgb):
        nonlocal calls
        calls += 1
        value = model.predict(Image.fromarray(rgb))
        old.verify_map(value.raw_mask, 'jpeg')
        return value.raw_mask, value.image_score

    checks = []
    refs = read(original_trial(main) / 'source-plan.json')['fused_native_references']['cases']
    for ref in refs:
        path = Path(ref['source_path'])
        source.require(source.sha(path) == ref['source']['image_sha256']
            and source.sha(Path(ref['projected_map_path'])) == ref['projected_map_sha256'], 'Consumed native reference changed')
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        actual, repeated = predict(rgb), predict(rgb)
        expected = np.load(ref['projected_map_path'], allow_pickle=False)
        projected = functional.interpolate(torch.from_numpy(actual[0].copy())[None, None], expected.shape,
            mode='bilinear', align_corners=False)[0, 0].numpy()
        source.require(np.allclose(projected, expected, atol=1e-4, rtol=1e-4)
            and abs(actual[1] - ref['score']) <= 1e-4 and (actual[1] >= .5) == (ref['score'] >= .5),
            'Retained native head, projected map or gate parity failed before holdout scoring')
        checks.append({'label': ref['source']['label'], 'projected_max_difference': float(np.max(np.abs(projected - expected))),
            'head_difference': abs(actual[1] - ref['score']), **old.repeat_check(actual, repeated, 'jpeg')})
    map_dir = trial / 'native-maps'
    map_dir.mkdir(exist_ok=False)
    rows, repeats = read(trial / 'input-plan.json')['files'], set()
    with (trial / 'native.jsonl').open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.image-fresh-native-journal/12',
            'freeze_sha256': source.sha(trial / 'native-freeze.json'), 'planned': 96}) + '\n')
        stream.flush()
        for index, row in enumerate(rows):
            before = calls
            result = {'id': row['id'], 'status': 'failed'}
            try:
                data, rgb, truth = input_for(row)
                raw, primary = predict(rgb)
                if row['label'] not in repeats:
                    result['native_repeat'] = old.repeat_check((raw, primary), predict(rgb), 'jpeg')
                    repeats.add(row['label'])
                path = map_dir / (hashlib.sha256(row['id'].encode()).hexdigest() + '.npy.gz')
                with path.open('xb') as target, gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0, compresslevel=6) as compressed:
                    np.save(compressed, raw, allow_pickle=False)
                packet, mask = replay(data, rgb, 'jpeg', raw, primary=primary,
                    boxes=[u['box'] for u in row['units'] if u['target'] == 'region'])
                packet['postprocessing_neural_forwards'] = packet.pop('new_neural_forwards')
                packet['postprocessing_weights_loaded'] = packet.pop('weights_loaded')
                result.update(packet, status='ok', raw_map_path=str(path.relative_to(trial)), raw_map_file_sha256=source.sha(path),
                    native_inference_forwards=calls - before, current_mask_metrics=mask_metrics(mask, truth),
                    current_source_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
            except BaseException as error:
                result.update(status='failed', error_type=type(error).__name__, native_inference_forwards=calls - before)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                raise
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if (index + 1) % 16 == 0:
                print(json.dumps({'completed_inputs': index + 1, 'planned': 96, 'native_forwards': calls}), flush=True)
    source.require(repeats == {0, 1} and calls == 102, 'All96 inputs, four probes and two new-input repeats required')
    check(main, trial)
    source.write(trial / 'native-receipt.json', {'schema': 'chipmunk.image-fresh-native-receipt/12',
        'completed_inputs': 96, 'native_forwards': calls, 'consumed_reference_checks': checks,
        'new_input_class_repeats': len(repeats), 'journal_sha256': source.sha(trial / 'native.jsonl'), 'serving_admitted': False})


def metrics(rows):
    count = Counter(planned=len(rows), genuine=0, ai=0, assessed=0, inconclusive=0, false_ai=0, ai_found=0, missed_ai_including_inconclusive=0, exact_point_ties=0)
    losses = []
    for row in rows:
        count['ai' if row['label'] else 'genuine'] += 1
        count['assessed' if row['assessed'] else 'inconclusive'] += 1
        count['exact_point_ties'] += bool(row.get('exact_point_tie', False))
        if row['label']:
            count['ai_found' if row['flag'] is True else 'missed_ai_including_inconclusive'] += 1
        else:
            count['false_ai'] += row['flag'] is True
        if row['probability'] is not None:
            losses.append((row['probability'] - row['label'])**2)
    return {**count, 'brier_on_complete_balanced_pairs': float(np.mean(losses)) if len(losses) == len(rows) else None}


def paired_bootstrap(outputs):
    """All planned source pairs stay together, including unassessed decisions."""
    grouped = {}
    for arm, rows in outputs.items():
        table = {}
        for row in rows:
            key = row['group'], row['label']
            source.require(key not in table, 'One original/edit observation per source and method required')
            table[key] = row
        source.require(len(table) == 2 * len({r['group'] for r in rows})
            and {k[1] for k in table} == {0, 1}, 'Every bootstrap source needs both original and edit')
        grouped[arm] = table
    keys = set(grouped['primary-regions'])
    source.require(all(set(t) == keys for t in grouped.values()), 'Paired bootstrap cannot omit another method case')
    groups = sorted({k[0] for k in keys})
    draws = np.random.default_rng(20261107).integers(0, len(groups), size=(5000, len(groups)))
    result = {}
    candidate = grouped['primary-regions']
    for arm in ('current', 'primary'):
        reference = grouped[arm]
        values = {}
        for name, label in (('ai_flag_rate_difference', 1), ('genuine_false_flag_rate_difference', 0)):
            difference = np.asarray([int(candidate[g, label]['flag'] is True) - int(reference[g, label]['flag'] is True) for g in groups], dtype=np.float64)
            bootstrapped = difference[draws].mean(axis=1)
            values[name] = {'candidate_minus_reference': float(difference.mean()),
                'percentile95_interval': np.quantile(bootstrapped, [.025, .975]).tolist()}
        difference = np.asarray([sum(int(candidate[g, y]['assessed']) - int(reference[g, y]['assessed'])
            for y in (0, 1)) / 2 for g in groups])
        values['coverage_rate_difference'] = {'candidate_minus_reference': float(difference.mean()),
            'percentile95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        if all(t[g, y]['probability'] is not None for t in (candidate, reference) for g in groups for y in (0, 1)):
            difference = np.asarray([sum((candidate[g, y]['probability'] - y)**2
                - (reference[g, y]['probability'] - y)**2 for y in (0, 1)) / 2 for g in groups])
            values['brier_difference'] = {'candidate_minus_reference': float(difference.mean()),
                'percentile95_interval': np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist()}
        result[arm] = values
    return {'draws': 5000, 'seed': 20261107, 'source_groups': len(groups), 'comparisons': result,
        'meaning': 'Differences in flag rates across all planned pairs, with unassessed cases retained as not flagged; this does not call them original. See coverage counts separately.'}


def mask_summary(rows, responses):
    edited = [responses[r['id']]['current_mask_metrics'] for r in rows if r['label'] == 1]
    genuine = [responses[r['id']]['current_mask_metrics'] for r in rows if r['label'] == 0]
    return {'ai_inputs': len(edited), 'ai_masks_iou_at_least_0_3': sum(m['iou'] >= .3 for m in edited),
        'ai_mean_iou_including_misses': float(np.mean([m['iou'] for m in edited])),
        'ai_mean_pixel_f1_including_misses': float(np.mean([m['pixel_f1'] for m in edited])),
        'genuine_inputs': len(genuine), 'genuine_inputs_with_selected_pixels': sum(m['predicted_positive_pixels'] > 0 for m in genuine),
        'genuine_mean_selected_area_fraction': float(np.mean([m['flagged_area_fraction'] for m in genuine])),
        'candidate_changes_pixel_selection': False}


def local_summary(rows):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for kind in ('primary', 'quadrant'):
            selected = [r for r in rows if r['arm'] == arm and r['unit_id'].split('.')[0] == kind]
            if not selected:
                continue
            available = [r for r in selected if r['probability'] is not None]
            result[arm][kind] = {'planned': len(selected), 'assessed': len(available),
                'positive_units': sum(r['label'] == 1 for r in selected),
                'negative_units': sum(r['label'] == 0 for r in selected),
                'descriptive_unweighted_brier': float(np.mean([(r['probability'] - r['label'])**2 for r in available]))
                    if len(available) == len(selected) else None,
                'scope': 'Whole-file editing presence for primary, source-quadrant editing presence for quadrant; not pixel authorship or a class-balanced performance estimate.'}
    return result


def review(main, trial):
    check(main, trial)
    profile = read(trial / 'native-profile/profile.json')
    receipt = read(trial / 'native-receipt.json')
    source.require(profile['child_exit_code'] == 0 and profile['guard'] is None
        and receipt['completed_inputs'] == 96 and receipt['journal_sha256'] == source.sha(trial / 'native.jsonl'), 'Complete guarded native run required')
    plan = read(trial / 'input-plan.json')
    journal = [json.loads(s) for s in (trial / 'native.jsonl').read_text().splitlines()]
    source.require(journal[0]['freeze_sha256'] == source.sha(trial / 'native-freeze.json'), 'Native run has another freeze')
    responses = {r['id']: r for r in journal[1:]}
    source.require(len(journal) == 97 and len(responses) == 96 and all(r['status'] == 'ok' for r in responses.values())
        and set(responses) == {r['id'] for r in plan['files']}, 'All unique native inputs required')
    artifacts, points = models(main)
    outputs = {a: [] for a in ('current', *ARMS)}
    by_id = {r['id']: r for r in plan['files']}
    for row in plan['files']:
        native = responses[row['id']]
        assessed = native['current_reliable_cells'] > 0
        outputs['current'].append({**{k: row[k] for k in ('id', 'group', 'label', 'generator')},
            'assessed': assessed, 'flag': native['current_region_flag'] if assessed else None, 'probability': None})
    local = []
    for arm in ARMS:
        document = forward(plan, responses, artifacts[arm], arm)
        for file in document['files']:
            p = calibration.predict(file, artifacts[arm])
            row = by_id[file['id']]
            outputs[arm].append({**{k: row[k] for k in ('id', 'group', 'label', 'generator')},
                'assessed': p is not None, 'flag': p > points[arm]['threshold'] if p is not None else None,
                'probability': p, 'exact_point_tie': p == points[arm]['threshold'] if p is not None else False})
            for unit in file['units']:
                model = artifacts[arm]['specialists'][unit['specialist'] + '|' + unit['context']]
                p = 1 / (1 + math.exp(-max(-700., min(700., engine._logit(model, [unit['raw']]))))) if unit['status'] == 'ok' else None
                local.append({'arm': arm, 'file_id': file['id'], 'unit_id': unit['id'], 'location': unit['location'],
                    'label': unit['label'], 'probability': p, 'scope': 'Source-box editing presence, not pixel authorship.'})
    summary = {a: metrics(rows) for a, rows in outputs.items()}
    c, b, p = (summary[a] for a in ('primary-regions', 'current', 'primary'))
    gates = {'more_ai_than_current': c['ai_found'] > b['ai_found'],
        'no_additional_current_false_flags': c['false_ai'] <= b['false_ai'],
        'no_current_coverage_loss': c['assessed'] >= b['assessed'],
        'no_primary_ai_loss': c['ai_found'] >= p['ai_found'],
        'no_additional_primary_false_flags': c['false_ai'] <= p['false_ai'],
        'no_primary_coverage_loss': c['assessed'] >= p['assessed'],
        'complete_candidate_coverage': c['assessed'] == 96,
        'lower_brier_than_primary': c['brier_on_complete_balanced_pairs'] is not None
            and p['brier_on_complete_balanced_pairs'] is not None and c['brier_on_complete_balanced_pairs'] < p['brier_on_complete_balanced_pairs']}
    report = {'schema': 'chipmunk.image-fresh-jpeg-result/12', 'summary': summary, 'gates': gates,
        'qualifies_for_full_app_integration_test': all(gates.values()),
        'per_generator': {g: {a: metrics([r for r in rows if r['generator'] == g]) for a, rows in outputs.items()}
            for g in sorted({r['generator'] for r in plan['files']})},
        'paired_bootstrap': paired_bootstrap(outputs),
        'current_pixel_selection': mask_summary(plan['files'], responses),
        'current_pixel_selection_per_generator': {g: mask_summary([r for r in plan['files'] if r['generator'] == g], responses)
            for g in sorted({r['generator'] for r in plan['files']})},
        'local_source_box_scores': local_summary(local),
        'source_pairs': 48, 'jpeg_inputs': 96, 'calibration_or_point_refitting': False,
        'scope': POLICY['scope'], 'serving_admitted': False, 'outside_human_reviews': 0}
    source.write(trial / 'private-outcomes.json', {'files': outputs, 'local': local})
    source.write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'freeze', 'native', 'review'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    main, trial = args.main.resolve(), args.trial.resolve()
    if args.stage == 'prepare':
        prepare(main, trial)
    elif args.stage == 'freeze':
        expected = identity(main, trial)
        for name, digest in expected['assets'].items():
            source.require(source.sha(main / name) == digest, 'Original commercial cached model bytes changed')
        source.write(trial / 'native-freeze.json', expected)
    elif args.stage == 'native':
        native(main, trial)
    else:
        review(main, trial)
