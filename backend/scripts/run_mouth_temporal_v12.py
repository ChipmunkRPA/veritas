"""Freeze and test one nonlinear learner on complete, consumed mouth evidence."""
from __future__ import annotations

import argparse
import gzip
import importlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

import numpy as np

from scripts import mouth_temporal_v12 as method
from scripts import review_mouth_temporal_v11 as arithmetic
from scripts import run_mouth_temporal_v11 as old

PROJECT = Path(__file__).resolve().parents[2]
PACKET = 'docs/evaluation-panels/mouth-v11-fitting-review.json.gz'
PACKET_SHA = 'ac7555bf05b0b4c013f93ce07144880a67886916d84c22748fead5cf1e791566'
FILES = {'backend/scripts/mouth_temporal_v12.py', 'backend/scripts/run_mouth_temporal_v12.py',
         'backend/tests/test_mouth_temporal_v12.py', 'backend/scripts/review_mouth_temporal_v11.py',
         'docs/mouth-temporal-v12-protocol.md'}
RULES = {
    'method': method.SPEC, 'roles': {'fit_actors': 16, 'point_actors': 4, 'development_actors': 8},
    'cases': {'fit': 160, 'calibration': 24, 'development': 88}, 'features': 49,
    'weights': 'Unchanged source/class/condition weights, sum equal number of fitting windows',
    'normalization': 'None; finite numeric raw49-variable inputs, no categorical/missing routing',
    'threads': 'OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1',
    'point': 'nextafter(max(0, every margin on all24 original ordinary point clips),+infinity)',
    'selection': 'Preserve V8 absolute evidence first, otherwise highest raw tree margin; original top12 and ties',
    'gates': 'All eleven unchanged V9/V11 localization/ordinary-flag/coverage requirements, plus complete native-estimator/export margin correspondence <=1e-10 on all6373 windows',
    'scope': 'Consumed actor-separated development with reused audited neural/native feature receipts; no fresh neural forwards or source-time decode in this trial, no customer confidence',
    'source_terms': 'CREMA-D ODbL1.0 and Database Contents License1.0; existing source links/attribution retained in original packet. Neural checkpoints/generators have separate previously audited commercial declarations; no new third-party weights.',
    'no_search': True, 'no_fresh_sources': True, 'serving_admitted': False,
}


def sha(path):
    return old.digest(path)


def write_new(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')


def implementation():
    return {**old.implementation(), **{name: sha(PROJECT / name) for name in sorted(FILES)}}


def installed():
    names = ('sklearn.ensemble._hist_gradient_boosting.gradient_boosting',
             'sklearn.ensemble._hist_gradient_boosting.grower', 'sklearn.ensemble._hist_gradient_boosting.binning',
             'sklearn.ensemble._hist_gradient_boosting.predictor', 'sklearn.ensemble._hist_gradient_boosting._predictor',
             'sklearn.ensemble._hist_gradient_boosting._binning', 'sklearn.ensemble._hist_gradient_boosting.splitting',
             'sklearn.ensemble._hist_gradient_boosting.histogram', 'sklearn._loss.loss', 'sklearn._loss._loss')
    return {name: sha(Path(importlib.import_module(name).__file__)) for name in names}


def runtime():
    return {'python': platform.python_version(), 'system': platform.system(), 'architecture': platform.machine(),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
            **{name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}


def inputs(root):
    packet_path = PROJECT / PACKET
    if sha(packet_path) != PACKET_SHA:
        raise ValueError('Original complete public evidence changed')
    packet = json.loads(gzip.decompress(packet_path.read_bytes()))
    lock_path = root / 'experiment/lock.json'
    lock = old.check_lock(lock_path)
    if sha(lock_path) != packet['original_lock_sha256']:
        raise ValueError('Wrong original source/feature lock')
    rows = {stage: old.checked_rows(lock_path, root / 'experiment' / (stage + '-features.jsonl'), stage, count)
            for stage, count in RULES['cases'].items()}
    for stage, cases in rows.items():
        if any(any(row[k] != value for k, value in public.items())
               for row, public in zip(cases, packet['panels'][stage]['cases'], strict=True)):
            raise ValueError('Native/feature/source correspondence changed')
        for row in cases:
            if row['source_license'] != 'ODbL 1.0 and Database Contents License 1.0':
                raise ValueError('Original source terms required')
            for mask in row.get('masks', []):
                if sha(Path(row['mask_root']) / mask['path']) != mask['sha256']:
                    raise ValueError('Original fitting/development truth mask changed')
    if old.training_windows(rows['fit']) != packet['fitting_windows']:
        raise ValueError('Mask/time-derived fitting labels changed')
    groups = [{r['group'] for r in rows[s]} for s in RULES['cases']]
    if [len(g) for g in groups] != [16, 4, 8] or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Original source roles or independence changed')
    return lock, packet, rows


def freeze(root, output):
    if os.environ.get('OMP_NUM_THREADS') != '1' or os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise ValueError('Use one frozen CPU thread for each numerical runtime')
    if output.exists() or any((output.parent / name).exists() for name in ('head.json', 'summary.json')):
        raise ValueError('Never fitted output required')
    _, packet, _ = inputs(root)
    # Separate prior arithmetic review; shared source truths remain declared.
    arithmetic.review(packet)
    write_new(output, {'schema': 'chipmunk.mouth-temporal-histogram-lock/12', 'root': str(root.resolve()),
        'rules': RULES, 'packet_sha256': PACKET_SHA, 'implementation_sha256': implementation(),
        'installed_sha256': installed(), 'runtime': runtime(), 'project_fitting_started': False,
        'project_development_scored': False, 'deployment_eligible': False})
    print(json.dumps({'lock_sha256': sha(output), 'project_fitting_started': False}), flush=True)


def check(path):
    lock = json.loads(path.read_text())
    if (lock['schema'] != 'chipmunk.mouth-temporal-histogram-lock/12' or lock['rules'] != RULES or lock['packet_sha256'] != PACKET_SHA or lock['implementation_sha256'] != implementation()
            or lock['installed_sha256'] != installed() or lock['runtime'] != runtime()
            or any(lock[k] is not False for k in ('project_fitting_started', 'project_development_scored', 'deployment_eligible'))):
        raise ValueError('Complete original prospective source/method/runtime freeze required')
    return lock, inputs(Path(lock['root']))


def public(path, output):
    lock, _ = check(path)
    body = {k: v for k, v in lock.items() if k != 'root'}
    body['implementation_sha256'] = [{'path': k, 'sha256': v} for k, v in body['implementation_sha256'].items()]
    write_new(output, {'schema': 'chipmunk.mouth-temporal-histogram-public-freeze/12',
                       'private_lock_sha256': sha(path), 'lock': body, 'scope': 'No new fitted tree, source recordings or customer data'})


def execute(path, output):
    lock, (parent, packet, rows) = check(path)
    if output != path.parent or any((output / name).exists() for name in ('head.json', 'summary.json')):
        raise ValueError('Original new single-attempt output required')
    model, head = method.fit(packet['fitting_windows'])
    control_max = max(float(v) for row in rows['calibration'] for v in method.margins([w['features'] for w in row['windows']], head))
    head.update(lock_sha256=sha(path), packet_sha256=PACKET_SHA, ordinary_max_margin=control_max,
                review_point=math.nextafter(max(0., control_max), math.inf))
    write_new(output / 'head.json', head)  # Written before any new development response.
    full = np.asarray([w['features'] for cases in rows.values() for row in cases for w in row['windows']], np.float64)
    error = float(np.max(np.abs(model.decision_function(full) - method.margins(full, head))))
    if error > 1e-10:
        raise ValueError('Native estimator/complete export response differs')
    parent_lock = old.context.check_lock(Path(parent['parent_lock']))
    _, absolute = old.load_rows(Path(parent_lock['prior_scores']['v8']['path']))
    if [{k: r[k] for k in ('id', 'probe_windows')} for r in absolute] != packet['absolute_reference']['cases']:
        raise ValueError('Retained absolute evidence changed')
    old_cases, temporal, combined, scored, signals = packet['absolute_reference']['outcomes'], [], [], [], []
    absolute_point = packet['absolute_reference']['review_point']
    if [r['id'] for r in rows['development']] != [r['id'] for r in absolute]:
        raise ValueError('Development source ordering changed')
    for row, reference in zip(rows['development'], absolute, strict=True):
        windows = [{**w, 'response': float(v)} for w, v in zip(row['windows'], method.margins([w['features'] for w in row['windows']], head), strict=True)]
        windows.sort(key=lambda w: (-w['response'], w['start_s'], w['track_id']))
        new = {**row, 'probe_windows': windows[:12], 'roi_bindings': {'probe': windows[0]['region']}}
        root = Path(row['mask_root'])
        temporal += old.baseline.evaluate([new], 'probe', head['review_point'], root)
        signal = old.method.selected_signal(reference['probe_windows'], absolute_point, new['probe_windows'], head['review_point'])
        combined += old.baseline.evaluate([reference] if signal == 'absolute' else [new], 'probe', absolute_point if signal == 'absolute' else head['review_point'], root)
        scored.append({'id': row['id'], 'windows': windows})
        signals.append({'id': row['id'], 'signal': signal})
    pseudo = [{'status': r['status'], 'baseline_replay_parity': all(h['baseline_replay_parity'] for h in r['heads'].values()),
               'prior_native_input_parity': r['native_input_replay'] and all(h['prior_native_and_margin_parity'] for h in r['heads'].values())} for r in rows['development']]
    gates = old.paired.gates(old_cases, combined, [r['family'] for r in rows['development']], pseudo)
    gates['complete_export_correspondence'] = len(full) == 6373 and error <= 1e-10
    result = {'schema': 'chipmunk.mouth-temporal-histogram-development/12', 'lock_sha256': sha(path), 'head_sha256': sha(output / 'head.json'),
        'packet_sha256': PACKET_SHA, 'baseline': old.baseline.summary(old_cases), 'temporal_only': old.baseline.summary(temporal),
        'combined': old.baseline.summary(combined), 'gates': gates, 'passes': all(gates.values()),
        'cases': {'temporal': temporal, 'combined': combined}, 'scored_windows': scored, 'selected_signals': signals,
        'native_estimator_export_windows': len(full), 'max_native_export_margin_difference': error,
        'new_native_video_decodes': 0, 'new_neural_forwards': 0, 'deployment_eligible': False, 'outside_reviewers': 0,
        'scope': RULES['scope']}
    write_new(output / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ('baseline', 'temporal_only', 'combined', 'gates', 'passes')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'public', 'execute'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'public': public, 'execute': execute}[args.stage](args.input, args.output)
