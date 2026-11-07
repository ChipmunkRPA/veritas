"""Frozen post-hoc comparison of passage-scope consistency; no neural inference."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.metadata
import json
import math
from pathlib import Path

from app.bench import specialists as legacy
from app.bench import specialists_scoped_v3 as scoped
from scripts import reproduce_writing_specialists_v5 as writing
from scripts import specialist_source_tail_v1 as tail

ROOT = Path(__file__).resolve().parents[2]
ARMS = {context + '-' + variant: [context, variant] for context in ('global', 'context')
    for variant in ('baseline', 'original-combined', 'shared-max', 'shared-scoped')}
METHOD = {'schema': 'chipmunk.scoped-writing-method/6', 'arms': ARMS,
    'minimum_source_groups_per_class': 25, 'l2': .001, 'reference_prior': .5,
    'point': tail.METHOD['point'], 'alpha': .1,
    'same_scope': 'Exact semantic target and source location including extracted-text hash. Distinct declared specialists; no duplicate view or across-passage agreement.',
    'family': 'The shared-max and shared-scoped arms put both authorship detectors in one dependent writing-view family. Original-combined preserves v5 separate families as a separate reference, not an independence claim.',
    'selection': 'All eight artifacts and genuine-only points precede reserved-role evaluation. No generator-specific routing, threshold tuning, new media or native inference.',
    'limits': 'Consumed writing-v5 sources and known diagnostic outcomes. Artificial60-word first/100-word second mixtures; no whole document/language/generator generalization or fresh validation.',
    'serving_admitted': False}
PINS = ('backend/scripts/scoped_writing_v6.py', 'backend/app/bench/specialists.py',
    'backend/app/bench/specialists_scoped_v3.py', 'backend/scripts/specialist_source_tail_v1.py',
    'backend/scripts/reproduce_writing_specialists_v5.py', 'backend/tests/test_specialist_scoped_v3.py',
    'backend/tests/test_scoped_writing_v6.py', 'docs/scoped-consistency-v3-design.md')


def engine_for(arm):
    return scoped if ARMS[arm][1] == 'shared-scoped' else legacy


def documents():
    release = json.loads((ROOT / 'docs/evaluation-panels/writing-specialists-v5-release.json').read_text())
    for name, expected in release['files_sha256'].items():
        legacy.check(tail.sha(ROOT / name) == expected, 'Original writing source/release changed')
    original = json.loads(gzip.decompress((ROOT / 'docs/evaluation-panels/writing-specialists-v5-scalars.json.gz').read_bytes()))
    result = {}
    for arm, (context, variant) in ARMS.items():
        original_arm = ('baseline' if variant == 'baseline' else 'combined') + '-' + context
        document = writing.derive(original, original_arm)
        legacy.check(legacy.digest(document) == release['manifest_sha256'][original_arm], 'Original scalar/role bindings changed')
        if variant.startswith('shared-'):
            for spec in document['policy']['specialists']:
                spec['family'] = 'writing-views'
            for file in document['files']:
                file['required_families'] = ['writing-views']
        engine_for(arm).audit(document)
        result[arm] = document
    return result


def identity():
    return {'method': METHOD, 'code_sha256': {name: tail.sha(ROOT / name) for name in PINS},
        'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
        'manifest_sha256': {arm: legacy.digest(doc) for arm, doc in documents().items()},
        'new_neural_forwards': 0, 'serving_admitted': False}


def freeze(trial):
    trial.mkdir(exist_ok=False)
    tail.write(trial / 'freeze.json', identity())
    print(json.dumps({'freeze_sha256': tail.sha(trial / 'freeze.json'), 'arms': len(ARMS), 'new_neural_forwards': 0}))


def check(trial):
    legacy.check(json.loads((trial / 'freeze.json').read_text()) == identity(), 'Frozen scoped-writing protocol/source changed')


def predict(file, artifact, engine):
    values, combined = engine._features(file, artifact), artifact['combiners'].get(engine.pattern(file))
    if (values is None or not combined or combined['response']['status'] != 'fitted'
            or combined.get('calibration', {}).get('status') != 'fitted'):
        return None
    eta = engine._logit(combined['calibration'], [engine._logit(combined['response'], values)])
    return 1 / (1 + math.exp(-max(-700., min(700., eta))))


def fit(trial):
    check(trial)
    for arm, doc in documents().items():
        engine = engine_for(arm)
        artifact = engine.train(doc, min_groups_per_class=25)
        point = tail.point([{**{k: file[k] for k in ('id', 'role', 'group', 'label')},
            'probability': predict(file, artifact, engine)} for file in doc['files'] if file['role'] == 'fusion_cal'])
        tail.write(trial / (arm + '-artifact.json'), artifact)
        tail.write(trial / (arm + '-point.json'), point)
    tail.write(trial / 'fit-complete.json', {'freeze_sha256': tail.sha(trial / 'freeze.json'),
        'artifacts': {arm: tail.sha(trial / (arm + '-artifact.json')) for arm in ARMS},
        'points': {arm: tail.sha(trial / (arm + '-point.json')) for arm in ARMS}, 'serving_admitted': False})


def adverse(doc, artifact, engine):
    revised = copy.deepcopy(doc)
    for file in revised['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                if unit['status'] == 'ok':
                    unit['raw'] = 1 - unit['raw']
    legacy.check(engine.train(revised, min_groups_per_class=25) == artifact, 'Confirmation entered fitting')
    original = engine.evaluate(doc, artifact)
    assessed = {r['id'] for r in original['outcomes'] if r['probability'] is not None}
    legacy.check(bool(assessed), 'No assessed file available for the missing-evidence test')
    revised = copy.deepcopy(doc)
    file = next(f for f in revised['files'] if f['id'] in assessed)
    file['units'][0].update(status='skipped', raw=None, reason='Adverse test removes one planned check')
    row = next(r for r in engine.evaluate(revised, artifact)['outcomes'] if r['id'] == file['id'])
    legacy.check(row['status'] == 'inconclusive' and row['probability'] is None, 'Missing check became an originality verdict')
    return {'confirmation_response_invariance': True, 'missing_required_check_withheld': True}


def review(trial):
    check(trial)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    report = {'schema': 'chipmunk.scoped-writing-result/6', 'method': METHOD, 'arms': {}, 'serving_admitted': False}
    for arm, doc in documents().items():
        legacy.check(tail.sha(trial / (arm + '-artifact.json')) == complete['artifacts'][arm]
            and tail.sha(trial / (arm + '-point.json')) == complete['points'][arm], 'Frozen parameters/point changed')
        artifact = json.loads((trial / (arm + '-artifact.json')).read_text())
        point = json.loads((trial / (arm + '-point.json')).read_text())
        engine = engine_for(arm)
        result = engine.evaluate(doc, artifact)
        tail.write(trial / (arm + '-confirmation.json'), result)
        by_id = {f['id']: f for f in doc['files']}
        rows = [{**r, 'condition': tail.condition(by_id[r['id']])} for r in result['outcomes']]
        report['arms'][arm] = {'all': tail.summarize(rows, point),
            'conditions': {c: tail.summarize([r for r in rows if r['condition'] == c], point)
                for c in sorted({r['condition'] for r in rows})},
            'point': point, 'adverse': adverse(doc, artifact, engine)}
    for arm, result in report['arms'].items():
        context, variant = ARMS[arm]
        result['vs_baseline'] = tail.gates(result, report['arms'][context + '-baseline'])
        if variant == 'shared-scoped':
            result['vs_same_family_max'] = tail.gates(result, report['arms'][context + '-shared-max'])
            result['development_continuation_eligible'] = all(result['vs_baseline'].values()) and all(result['vs_same_family_max'].values())
        else:
            result['development_continuation_eligible'] = False
    tail.write(trial / 'result.json', report)
    print(json.dumps({'arms': {arm: result['all'] for arm, result in report['arms'].items()},
        'new_neural_forwards': 0, 'serving_admitted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'fit', 'review'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'review': review}[args.stage](args.trial)
