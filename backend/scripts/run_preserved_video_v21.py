"""Frozen checked-solver audit and preserved-view video scalar comparison."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.bench import specialists as legacy
from scripts import monotone_newton_v4 as solver
from scripts import scoped_writing_v6 as writing
from scripts import score_video_specialists_v20 as previous
from scripts import specialists_preserved_v4 as engine

ROOT = Path(__file__).resolve().parents[2]
ARMS = {'appearance-direct': ['face-appearance', 'raw-view-max'],
    'appearance-positive': ['face-appearance', 'family-max'],
    'family-max': ['all', 'family-max'], 'family-repeat': ['all', 'family-max-repeat'],
    'view-max': ['all', 'view-max']}
METHOD = {'schema': 'chipmunk.preserved-evidence-method/21', 'arms': ARMS, 'solver_policy': solver.POLICY,
    'minimum': 25, 'l2': .001, 'reference_prior': .5, 'old_objective_slack': 1e-11,
    'repeat_probability_limit': 1e-7, 'replay_probability_limit': 1e-6,
    'point': 'Common complete genuine fusion_cal groups, min25; existing raw appearance false-component budget capped atfloor(.1*n). Strict greater than ranked group maximum.',
    'raw_appearance_point': previous.RAW_POINTS['face-appearance'],
    'scope': 'Consumed source-bound public scalars. No native inference, fresh source or serving admission.'}
PINS = ('backend/scripts/run_preserved_video_v21.py', 'backend/scripts/monotone_newton_v4.py',
    'backend/scripts/specialists_preserved_v4.py', 'backend/scripts/specialist_runtime_v8.py',
    'backend/tests/test_monotone_newton_v4.py', 'backend/tests/test_specialists_preserved_v4.py',
    'backend/tests/test_preserved_video_v21.py', 'docs/preserved-evidence-v21-protocol.md')
SOURCES = ('audio-specialists-v15', 'image-supported-v11', 'video-face-specialists-v20')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_files():
    names = set(PINS)
    for study in SOURCES + ('writing-specialists-v5',):
        name = f'docs/evaluation-panels/{study}-release.json'
        release = read(ROOT / name)
        names.add(name)
        for path, expected in release['files_sha256'].items():
            legacy.check(sha(ROOT / path) == expected, 'Original published source changed')
            names.add(path)
    names.update(writing.PINS)
    names.update(('docs/evaluation-panels/scoped-writing-v6-expected.json.gz',
        'docs/evaluation-panels/scoped-writing-v6-release.json', 'backend/scripts/scoped_writing_v6.py',
        'backend/app/bench/specialists_scoped_v3.py'))
    return sorted(names)


def identity():
    return {'schema': 'chipmunk.preserved-evidence-freeze/21', 'method': METHOD,
        'files_sha256': [{'path': n, 'sha256': sha(ROOT / n)} for n in source_files()],
        'runtime': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'psutil')},
        'new_neural_forwards': 0, 'serving_admitted': False}


def check(trial):
    frozen = read(trial / 'freeze.json')
    legacy.check(frozen == identity(), 'Frozen source/code/runtime/protocol changed')
    return frozen


def documents():
    release = read(ROOT / 'docs/evaluation-panels/video-face-specialists-v20-release.json')
    original = json.loads(gzip.decompress((ROOT / release['manifests']).read_bytes()))['all']
    result = {}
    for arm, (view, _) in ARMS.items():
        document = copy.deepcopy(original)
        if view != 'all':
            document['policy']['specialists'] = [s for s in document['policy']['specialists'] if s['id'] == view]
            for file in document['files']:
                file['units'] = [u for u in file['units'] if u['specialist'] == view]
                legacy.check(file['label'] == int(any(u['label'] for u in file['units'])),
                    'View projection changes source truth; refuse rather than drop or relabel files')
        engine.audit(document)
        result[arm] = document
    return result


def blind(document):
    result = copy.deepcopy(document)
    for file in result['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                if unit['status'] == 'ok':
                    unit.update(raw=None, status='unavailable', reason='Reserved scalar response blinded before fitting.',
                        measurement_sha256=legacy.digest({'blinded_unit': unit['id'], 'input': file['input_sha256']}))
    return result


def old_cases():
    for study in SOURCES:
        release = read(ROOT / f'docs/evaluation-panels/{study}-release.json')
        packet = json.loads(gzip.decompress((ROOT / release['manifests']).read_bytes()))
        for arm, document in packet.items():
            yield study, arm, document, read(ROOT / f'docs/evaluation-panels/{study}-{arm}-artifact.json'), legacy
    expected = json.loads(gzip.decompress((ROOT / 'docs/evaluation-panels/scoped-writing-v6-expected.json.gz').read_bytes()))
    for arm, document in writing.documents().items():
        yield 'scoped-writing-v6', arm, document, expected['arms'][arm]['artifact'], writing.engine_for(arm)


def inputs_for(document, artifact, source_engine):
    training = [f for f in document['files'] if f['role'] != 'confirmation']
    for spec in document['policy']['specialists']:
        for context in spec['contexts']:
            selected = [(f, u) for f in training if f['role'] == 'specialist_cal' for u in f['units']
                if u['specialist'] == spec['id'] and u['context'] == context and u['status'] == 'ok' and u['label'] is not None]
            yield 'head.' + spec['id'] + '.' + context, [u['raw'] for _, u in selected], [u['label'] for _, u in selected], \
                [f['group'] for f, _ in selected], artifact['specialists'][spec['id'] + '|' + context]
    for key, combination in artifact['combiners'].items():
        for role, stage in (('fusion_fit', 'response'), ('fusion_cal', 'calibration')):
            if stage not in combination:
                continue
            selected = [(f, source_engine._features(f, artifact)) for f in training if f['role'] == role and legacy.pattern(f) == key]
            selected = [(f, x) for f, x in selected if x is not None]
            x = [values if stage == 'response' else source_engine._logit(combination['response'], values) for _, values in selected]
            yield stage + '.' + key, x, [f['label'] for f, _ in selected], [f['group'] for f, _ in selected], combination[stage]


def solver_audit(trial):
    check(trial)
    counts, maximum, old_residual, worse = Counter(), 0., 0., 0.
    journal = trial / 'solver-audit.jsonl'
    with journal.open('x') as stream:
        for study, arm, document, original, source_engine in old_cases():
            source_engine.audit(document)
            for stage, x, y, groups, model in inputs_for(document, original, source_engine):
                context = {'study': study, 'arm': arm, 'stage': stage,
                    'input_sha256': legacy.digest({'x': x, 'y': y, 'groups': groups})}
                try:
                    fitted = solver.fit(x, y, groups, minimum=25, l2=.001)
                    legacy.check(fitted['status'] == model['status'], 'Solver changes source support')
                    counts[study + '.all'] += 1
                    if fitted['status'] != 'fitted':
                        stream.write(json.dumps({**context, 'status': fitted['status']}) + '\n')
                        continue
                    values = np.asarray(x, np.float64)
                    if values.ndim == 1:
                        values = values[:, None]
                    labels = np.asarray(y)
                    count = {v: len({g for g, t in zip(groups, labels, strict=True) if t == v}) for v in (0, 1)}
                    cells = Counter(zip(groups, labels, strict=True))
                    weight = np.asarray([1 / (2 * count[int(t)] * cells[g, t]) for g, t in zip(groups, labels, strict=True)])
                    target = np.where(labels == 1, (count[1] + 1) / (count[1] + 2), 1 / (count[0] + 2))
                    legacy.check(np.array_equal(fitted['center'], model['center']) and np.array_equal(fitted['scale'], model['scale']),
                        'Optimizer comparison changed source standardization')
                    z = np.clip((values - model['center']) / model['scale'], -8., 8.)
                    theta = np.r_[model['coefficients'], model['intercept']]
                    value, gradient, _ = solver.objective(theta, z, target, weight, np.full(z.shape[1], .001))
                    difference = fitted['numerical_check']['independent_objective'] - value
                    legacy.check(difference <= METHOD['old_objective_slack'], 'New fitted objective is worse than original')
                    drift = max(abs(legacy._logit(model, row) - legacy._logit(fitted, row)) for row in values)
                    maximum, worse = max(maximum, drift), max(worse, difference)
                    old_residual = max(old_residual, solver.residual(theta, gradient))
                    counts[study + '.checked'] += 1
                    stream.write(json.dumps({**context, 'status': 'checked', 'new_fit': fitted,
                        'old_kkt_infinity': solver.residual(theta, gradient), 'objective_difference': difference,
                        'maximum_training_logit_difference': drift}, allow_nan=False) + '\n')
                    stream.flush()
                except BaseException as error:
                    stream.write(json.dumps({**context, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}) + '\n')
                    stream.flush()
                    raise
    write(trial / 'solver-audit.json', {'schema': 'chipmunk.frozen-real-problem-solver-audit/21',
        'problems': dict(counts), 'maximum_training_logit_drift': maximum, 'maximum_old_kkt_infinity': old_residual,
        'maximum_new_objective_excess': worse, 'journal_sha256': sha(journal),
        'new_neural_forwards': 0, 'serving_admitted': False,
        'limits': 'Exact frozen training-input optimization comparison; not new detection evidence or outside human review.'})


def mapped_point(artifact):
    file = {'modality': 'video', 'required_families': ['video-face']}
    combination = artifact['combiners'][legacy.pattern(file)]
    a, b = combination['response'], combination.get('calibration', {})
    result = {'status': 'unsupported', 'threshold': None, 'raw_point': METHOD['raw_appearance_point'],
        'scope': 'Predeclared point-preserving control; not more detection or public threshold'}
    if a['status'] == b.get('status') == 'fitted' and a['coefficients'][0] > 0 and b['coefficients'][0] > 0:
        first = (METHOD['raw_appearance_point'] - a['center'][0]) / a['scale'][0]
        eta = legacy._logit(a, [METHOD['raw_appearance_point']])
        second = (eta - b['center'][0]) / b['scale'][0]
        if -8 < first < 8 and -8 < second < 8:
            result.update(status='fitted', threshold=engine.probability(b, [eta]))
    return result


def fit(trial):
    check(trial)
    legacy.check((trial / 'solver-audit.json').exists(), 'Complete solver audit before new combination fitting')
    docs, artifacts, values = {}, {}, {}
    original = documents()
    for arm, (_, mode) in ARMS.items():
        document = blind(original[arm])
        artifact = engine.train(document, mode, minimum=25, l2=.001)
        write(trial / (arm + '-artifact.json'), artifact)
        docs[arm], artifacts[arm] = document, artifact
        grouped = {}
        selected = [f for f in document['files'] if f['role'] == 'fusion_cal' and f['label'] == 0]
        for file in selected:
            p = engine.predict(file, artifact)
            if p is not None:
                grouped.setdefault(file['group'], []).append(p)
        values[arm] = {g: max(v) for g, v in grouped.items() if len(v) == sum(f['group'] == g for f in selected)}
    common = set.intersection(*(set(v) for v in values.values()))
    reference = {}
    for file in docs['appearance-direct']['files']:
        if file['role'] == 'fusion_cal' and file['label'] == 0 and file['group'] in common:
            reference[file['group']] = reference.get(file['group'], False) or any(u['raw'] > METHOD['raw_appearance_point'] for u in file['units'])
    for arm in ARMS:
        write(trial / (arm + '-point.json'), previous.matched_point({g: v for g, v in values[arm].items() if g in common}, reference))
    write(trial / 'appearance-mapped-point.json', mapped_point(artifacts['appearance-direct']))
    names = [arm + '-' + kind + '.json' for arm in ARMS for kind in ('artifact', 'point')] + ['appearance-mapped-point.json']
    write(trial / 'fit-complete.json', {'files_sha256': [{'path': n, 'sha256': sha(trial / n)} for n in names],
        'common_complete_genuine_groups': len(common), 'reserved_role_scored_with_new_method': False, 'serving_admitted': False})


def verify_fit(trial):
    frozen = read(trial / 'fit-complete.json')
    legacy.check(all(sha(trial / r['path']) == r['sha256'] for r in frozen['files_sha256']), 'Frozen fitted parameters/points changed')
    return frozen


def evaluate(trial):
    check(trial)
    verify_fit(trial)
    docs, results, artifacts = documents(), {}, {}
    report = {'schema': 'chipmunk.preserved-video-result/21', 'method': METHOD, 'arms': {},
        'serving_admitted': False, 'fresh_sources': 0, 'outside_human_reviews': 0}
    for arm in ARMS:
        artifact = read(trial / (arm + '-artifact.json'))
        result = engine.evaluate(docs[arm], artifact)
        write(trial / (arm + '-evaluation.json'), result)
        results[arm], artifacts[arm] = result, artifact
        report['arms'][arm] = {'all': previous.summarize(result['outcomes'], read(trial / (arm + '-point.json'))),
            'mode': artifact['mode'], 'combination_coefficients': {k: v['response'].get('coefficients') for k, v in artifact['combiners'].items()}}
    raw = []
    for file in docs['appearance-direct']['files']:
        if file['role'] == 'confirmation':
            complete = all(u['status'] == 'ok' for u in file['units'])
            raw.append({'id': file['id'], 'group': file['group'], 'label': file['label'],
                'probability': float(any(u['raw'] > METHOD['raw_appearance_point'] for u in file['units'])) if complete else None})
    summary = previous.summarize(raw, {'status': 'fitted', 'threshold': .5})
    summary.pop('source_class_balanced_brier_on_scored_cases')
    report['raw_appearance_reference'] = summary
    mapped = read(trial / 'appearance-mapped-point.json')
    mapped_summary = previous.summarize(results['appearance-direct']['outcomes'], mapped)
    if mapped['status'] == 'fitted':
        for a, b in zip(results['appearance-direct']['outcomes'], raw, strict=True):
            legacy.check((a['probability'] is None) == (b['probability'] is None), 'Mapped point changes scope/coverage')
            if a['probability'] is not None:
                legacy.check((a['probability'] > mapped['threshold']) == bool(b['probability']), 'Mapped point changes the existing raw direction')
    report['appearance_direct_mapped_point_control'] = mapped_summary
    a, b = results['family-max']['outcomes'], results['family-repeat']['outcomes']
    difference = 0.
    for left, right in zip(a, b, strict=True):
        legacy.check(left['id'] == right['id'] and (left['probability'] is None) == (right['probability'] is None), 'Repeated feature changed scope/coverage')
        if left['probability'] is not None:
            difference = max(difference, abs(left['probability'] - right['probability']))
    legacy.check(difference <= METHOD['repeat_probability_limit'], 'Equal-dimension control changes effective regularization')
    report['matched_duplicate_control_max_probability_difference'] = difference
    baseline_brier = report['arms']['appearance-direct']['all']['source_class_balanced_brier_on_scored_cases']
    for current in report['arms'].values():
        metric = current['all']
        current['qualifies_for_fresh_detection_followup'] = (metric['ai_flagged'] > summary['ai_flagged']
            and metric['false_ai'] <= summary['false_ai'] and metric['false_ai_groups'] <= summary['false_ai_groups']
            and metric['inconclusive'] <= summary['inconclusive']
            and metric['source_class_balanced_brier_on_scored_cases'] is not None and baseline_brier is not None
            and metric['source_class_balanced_brier_on_scored_cases'] < baseline_brier)
    # Refit after adversarial reserved-response changes; the canonical artifact
    # digest must stay fixed. Missing status is retained, never made numeric.
    revised = copy.deepcopy(docs['view-max'])
    changed = preserved = 0
    ranges = {s['id']: s['raw_range'] for s in revised['policy']['specialists']}
    for file in revised['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                if unit['status'] == 'ok':
                    unit['raw'] = sum(ranges[unit['specialist']]) - unit['raw']
                    changed += 1
                else:
                    preserved += 1
    legacy.check(engine.train(revised, 'view-max')['sha256'] == artifacts['view-max']['sha256'], 'Reserved measurements entered new fitting')
    assessed = {r['id'] for r in results['view-max']['outcomes'] if r['probability'] is not None}
    revised = copy.deepcopy(docs['view-max'])
    if assessed:
        file = next(f for f in revised['files'] if f['id'] in assessed)
        file['units'][0].update(status='skipped', raw=None, reason='Adverse required observation removed.')
        legacy.check(engine.predict(file, artifacts['view-max']) is None, 'Missing required evidence became a probability')
    report['adverse'] = {'reserved_not_fitted': True, 'changed_reserved_units': changed,
        'preserved_missing_units': preserved, 'missing_check_withheld': bool(assessed)}
    write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'solver-audit', 'fit', 'evaluate'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    trial = args.trial.resolve()
    if args.stage == 'freeze':
        trial.mkdir(exist_ok=False)
        write(trial / 'freeze.json', identity())
        print(json.dumps({'freeze_sha256': sha(trial / 'freeze.json'), 'new_neural_forwards': 0}))
    elif args.stage == 'solver-audit':
        solver_audit(trial)
    elif args.stage == 'fit':
        fit(trial)
    else:
        evaluate(trial)
