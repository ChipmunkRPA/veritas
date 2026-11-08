"""Research-only dependent evidence fitting that preserves each view's signal.

Legacy v2/v3 and serving stay unchanged. Positive maxima within a view preserve
ANY-AI scope. Separately fitted features are not independent votes. Repeated-max
controls have the same dimensionality and slope penalty as per-view features.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
from collections import defaultdict
from pathlib import Path

from app.bench import specialists as legacy
from scripts import monotone_newton_v4 as solver

MODES = ('family-max', 'family-max-repeat', 'view-max', 'raw-view-max')
check, digest, _logit = legacy.check, legacy.digest, legacy._logit


def implementation():
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (('engine', __file__), ('solver', solver.__file__), ('legacy', legacy.__file__))}


def audit(manifest):
    result = legacy.audit(manifest)
    specs = {s['id']: s for s in manifest['policy']['specialists']}
    for file in manifest['files']:
        seen = set()
        for unit in file['units']:
            key = unit['specialist'], specs[unit['specialist']]['target'], digest(unit['location'])
            check(key not in seen, 'Duplicate view at one source location cannot create more evidence')
            seen.add(key)
        expected = {s['id'] for s in specs.values() if s['modality'] == file['modality'] and s['family'] in file['required_families']}
        check({u['specialist'] for u in file['units']} == expected, 'Every required declared view needs planned observations, including missing responses')
    return result


def layout(file, artifact):
    by_family = defaultdict(list)
    for spec in artifact['policy']['specialists']:
        if spec['modality'] == file['modality'] and spec['family'] in file['required_families']:
            by_family[spec['family']].append(spec['id'])
    features, penalties = [], []
    for family in sorted(file['required_families']):
        views = sorted(by_family[family])
        count = len(views)
        check(count > 0, 'Declared view layout required')
        if artifact['mode'] == 'family-max':
            features.append(('family', family))
            penalties.append(1)
        elif artifact['mode'] == 'family-max-repeat':
            features.extend(('family', family) for _ in views)
            penalties.extend([count] * count)
        else:
            features.extend(('view', name) for name in views)
            penalties.extend([count] * count)
    return features, penalties


def features(file, artifact):
    by_family, by_view = defaultdict(list), defaultdict(list)
    specs = {s['id']: s for s in artifact['policy']['specialists']}
    for unit in file['units']:
        spec = specs[unit['specialist']]
        model = artifact['specialists'][unit['specialist'] + '|' + unit['context']]
        if unit['status'] != 'ok' or model['status'] != 'fitted':
            return None
        value = unit['raw'] if artifact['mode'] == 'raw-view-max' else max(0., _logit(model, [unit['raw']]))
        by_view[spec['id']].append(value)
        by_family[spec['family']].append(value)
    order, _ = layout(file, artifact)
    return [max((by_family if kind == 'family' else by_view)[name]) for kind, name in order]


def train(manifest, mode, *, minimum=25, l2=.001):
    audit(manifest)
    check(mode in MODES, 'Explicit preserved-view feature mode required')
    if mode == 'raw-view-max':
        check(all(len({u['specialist'] for u in f['units']}) == 1 for f in manifest['files']),
            'Signed raw maximum is a single-view reference only; other-area negatives cannot cancel another view')
    training = [f for f in manifest['files'] if f['role'] != 'confirmation']
    artifact = {'schema': 'chipmunk.specialist-preserved-calibration/4', 'policy': copy.deepcopy(manifest['policy']),
        'mode': mode, 'implementation_sha256': implementation(),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'plan_sha256': legacy.plan_digest(manifest), 'training_sha256': digest(training),
        'minimum_groups_per_class': minimum, 'l2': l2, 'reference_prior': .5,
        'specialists': {}, 'combiners': {}, 'serving_admitted': False}
    for spec in manifest['policy']['specialists']:
        for context in spec['contexts']:
            selected = [(f, u) for f in training if f['role'] == 'specialist_cal' for u in f['units']
                if u['specialist'] == spec['id'] and u['context'] == context and u['status'] == 'ok' and u['label'] is not None]
            artifact['specialists'][spec['id'] + '|' + context] = solver.fit([u['raw'] for _, u in selected],
                [u['label'] for _, u in selected], [f['group'] for f, _ in selected], minimum=minimum, l2=l2)
    for key in sorted({legacy.pattern(f) for f in training if f['role'] == 'fusion_fit'}):
        selected = [(f, features(f, artifact)) for f in training if f['role'] == 'fusion_fit' and legacy.pattern(f) == key]
        usable = [(f, x) for f, x in selected if x is not None]
        order, penalties = layout(selected[0][0], artifact)
        response = solver.fit([x for _, x in usable], [f['label'] for f, _ in usable],
            [f['group'] for f, _ in usable], minimum=minimum, l2=l2, penalty_multipliers=penalties if usable else None)
        entry = {'response': response, 'features': [list(item) for item in order], 'penalty_multipliers': penalties,
            'fitting_files': len(selected), 'fitting_unassessed': len(selected) - len(usable)}
        if response['status'] == 'fitted':
            selected_cal = [(f, features(f, artifact)) for f in training if f['role'] == 'fusion_cal' and legacy.pattern(f) == key]
            available = [(f, x) for f, x in selected_cal if x is not None]
            entry['calibration'] = solver.fit([_logit(response, x) for _, x in available],
                [f['label'] for f, _ in available], [f['group'] for f, _ in available], minimum=minimum, l2=l2)
            entry.update(calibration_files=len(selected_cal), calibration_unassessed=len(selected_cal) - len(available))
        artifact['combiners'][key] = entry
    artifact['sha256'] = digest(artifact)
    return artifact


def probability(model, values):
    return 1 / (1 + math.exp(-max(-700., min(700., _logit(model, values)))))


def predict(file, artifact):
    values, combined = features(file, artifact), artifact['combiners'].get(legacy.pattern(file))
    if values is None or not combined or combined['response']['status'] != 'fitted' or combined.get('calibration', {}).get('status') != 'fitted':
        return None
    return probability(combined['calibration'], [_logit(combined['response'], values)])


def evaluate(manifest, artifact):
    audit(manifest)
    check(artifact.get('schema') == 'chipmunk.specialist-preserved-calibration/4'
        and artifact.get('mode') in MODES and artifact.get('serving_admitted') is False
        and artifact.get('sha256') == digest({k: v for k, v in artifact.items() if k != 'sha256'}), 'Exact new research artifact required')
    check(artifact['implementation_sha256'] == implementation()
        and artifact['packages'] == {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')}
        and artifact['plan_sha256'] == legacy.plan_digest(manifest) and artifact['policy'] == manifest['policy']
        and artifact['training_sha256'] == digest([f for f in manifest['files'] if f['role'] != 'confirmation']),
        'Frozen implementation/runtime/source plan/training changed')
    files, local = [], []
    for file in manifest['files']:
        if file['role'] != 'confirmation':
            continue
        p = predict(file, artifact)
        row = {key: file[key] for key in ('id', 'group', 'label', 'modality')}
        row.update(status='ok' if p is not None else 'inconclusive', probability=p,
            scope='planned_evidence_units', reference_prior=.5)
        if p is None:
            row['reason'] = 'Incomplete evidence or unsupported fit; no originality verdict.'
        files.append(row)
        for unit in file['units']:
            model = artifact['specialists'][unit['specialist'] + '|' + unit['context']]
            value = probability(model, [unit['raw']]) if unit['status'] == 'ok' and model['status'] == 'fitted' else None
            local.append({'file_id': file['id'], 'unit_id': unit['id'], 'specialist': unit['specialist'],
                'context': unit['context'], 'location': copy.deepcopy(unit['location']), 'probability': value,
                'status': 'ok' if value is not None else 'unavailable', 'reference_prior': .5,
                **({} if value is not None else {'reason': unit.get('reason') or 'Insufficient local calibration support.'})})
    return {'schema': 'chipmunk.specialist-preserved-evaluation/4', 'outcomes': files,
        'unit_outcomes': local, 'artifact_sha256': artifact['sha256'], 'serving_admitted': False,
        'limits': 'Consumed-source research probability under a50:50 reference prior; not whole-file authorship or customer accuracy.'}
