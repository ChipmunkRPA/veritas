"""Model-free source, postprocessing and independent forward-arithmetic audit."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_image_confirmation_v12 as frozen
from scripts.review_image_supported_v11 import verify_case


def affine(model, values):
    if model['status'] != 'fitted':
        return None
    frozen.source.require(len(values) == len(model['center']) == len(model['scale']) == len(model['coefficients']),
        'Exact calibration dimensions required')
    result = model['intercept']
    for value, center, scale, coefficient in zip(values, model['center'], model['scale'], model['coefficients'], strict=True):
        frozen.source.require(math.isfinite(value) and scale > 0, 'Finite available input and positive scaling required')
        result += coefficient * min(8., max(-8., (value - center) / scale))
    return result


def probability(row, native, artifact, arm):
    """Read the frozen scalar heads directly, without the scoring engine."""
    if native.get('status') != 'ok':
        return None
    keys = ['primary'] + ([u['id'] for u in row['units'] if u['target'] == 'region'] if arm == 'primary-regions' else [])
    logits = []
    for key in keys:
        target = 'region' if key.startswith('quadrant.') else 'primary'
        model = artifact['specialists']['image.jpeg.' + target + '|jpeg']
        value = native['raw_values'].get(key)
        if value is None or model['status'] != 'fitted':
            return None
        logits.append(affine(model, [value]))
    combination = artifact['combiners'].get('image|jpeg-local-edit')
    if not combination or any(combination[k]['status'] != 'fitted' for k in ('response', 'calibration')):
        return None
    score = affine(combination['response'], [max([0., *logits])])
    score = affine(combination['calibration'], [score])
    return 1 / (1 + math.exp(-min(700., max(-700., score))))


def nearest_quadrants(raw, width, height, units):
    """Independent nearest-grid indexing rather than OpenCV's resize helper."""
    y = np.arange(height, dtype=np.int64) * raw.shape[0] // height
    x = np.arange(width, dtype=np.int64) * raw.shape[1] // width
    projected = raw[y[:, None], x[None, :]]
    return {u['id']: float(projected[u['box'][1]:u['box'][1] + u['box'][3],
        u['box'][0]:u['box'][0] + u['box'][2]].mean(dtype=np.float64))
        for u in units if u['target'] == 'region'}


def audit(main, trial, output):
    import cv2

    from app.detectors.image import fused, splice

    frozen.check(main, trial)
    plan = frozen.read(trial / 'input-plan.json')
    receipt = frozen.read(trial / 'native-receipt.json')
    profile = frozen.read(trial / 'native-profile/profile.json')
    report = frozen.read(trial / 'result.json')
    outcomes = frozen.read(trial / 'private-outcomes.json')['files']
    journal = [json.loads(line) for line in (trial / 'native.jsonl').read_text().splitlines()]
    records = {r['id']: r for r in journal[1:]}
    frozen.source.require(len(records) == len(journal) - 1 == len(plan['files']) == 96
        and receipt['completed_inputs'] == 96 and receipt['native_forwards'] == 102
        and receipt['journal_sha256'] == frozen.source.sha(trial / 'native.jsonl')
        and profile['guard'] is None and profile['child_exit_code'] == 0, 'Complete guarded native result required')
    artifacts, points = frozen.models(main)
    differences, scalars, native_calls, repeats = [], 0, 4, 0
    cv2.setNumThreads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError('This audit must not load model weights')

    with patch.object(fused, 'load_model', forbidden), patch.object(splice, 'load_model', forbidden):
        for row in plan['files']:
            record = records[row['id']]
            data, rgb, truth = frozen.input_for(row)
            path = (trial / record['raw_map_path']).resolve()
            frozen.source.require(path.is_relative_to((trial / 'native-maps').resolve())
                and frozen.source.sha(path) == record['raw_map_file_sha256'], 'Bound lossless native map required')
            with gzip.open(path, 'rb') as stream:
                raw = np.load(stream, allow_pickle=False)
            frozen.old.verify_map(raw, 'jpeg')
            verify_case(row, record, data, rgb, truth, raw)
            values = nearest_quadrants(raw, rgb.shape[1], rgb.shape[0], row['units'])
            frozen.source.require(all(abs(value - record['raw_values'][key]) <= 1e-12 for key, value in values.items()),
                'Independent source-coordinate regional means differ')
            scalars += len(values)
            native_calls += record['native_inference_forwards']
            repeats += 'native_repeat' in record
            for arm in frozen.ARMS:
                manual = probability(row, record, artifacts[arm], arm)
                saved = next(r for r in outcomes[arm] if r['id'] == row['id'])
                frozen.source.require(manual is not None and saved['assessed'] is True
                    and saved['flag'] == (manual > points[arm]['threshold']), 'Independent fixed-point decision differs')
                differences.append(abs(manual - saved['probability']))
            saved = next(r for r in outcomes['current'] if r['id'] == row['id'])
            assessed = record['current_reliable_cells'] > 0
            frozen.source.require(saved['assessed'] == assessed
                and saved['flag'] == (record['current_region_flag'] if assessed else None), 'Current missingness or region decision differs')
    frozen.source.require(native_calls == 102 and repeats == 2 and max(differences) <= 1e-12,
        'Independent arithmetic, all forwards and class repeats must agree')
    for arm, rows in outcomes.items():
        counts = report['summary'][arm]
        frozen.source.require(counts['planned'] == len(rows)
            and counts['ai_found'] == sum(r['label'] == 1 and r['flag'] is True for r in rows)
            and counts['false_ai'] == sum(r['label'] == 0 and r['flag'] is True for r in rows)
            and counts['inconclusive'] == sum(not r['assessed'] for r in rows), 'Aggregate denominator or missingness changed')
        if arm != 'current':
            brier = math.fsum((r['probability'] - r['label'])**2 for r in rows) / len(rows)
            frozen.source.require(abs(brier - counts['brier_on_complete_balanced_pairs']) <= 1e-12, 'Independent Brier differs')
    frozen.check(main, trial)
    frozen.source.write(output, {'schema': 'chipmunk.image-fresh-source-arithmetic-audit/12',
        'freeze_sha256': frozen.source.sha(trial / 'native-freeze.json'), 'result_sha256': frozen.source.sha(trial / 'result.json'),
        'source_input_map_and_postprocessing_cases': 96, 'independent_source_quadrant_means': scalars,
        'independent_file_probabilities': len(differences), 'maximum_probability_difference': max(differences),
        'actual_native_forwards_verified': native_calls, 'this_audit_neural_forwards': 0,
        'this_audit_model_weights_loaded': False, 'outside_human_reviews': 0, 'serving_admitted': False})
    print(json.dumps({'audited_inputs': 96, 'independent_probabilities': len(differences),
        'maximum_probability_difference': max(differences), 'new_neural_forwards': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.main.resolve(), args.trial.resolve(), args.output.resolve())
