"""Refit real-text scalar calibrations without text, weights or inference."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from app.bench import specialists as engine

LIMIT = 1e-6
PUBLIC = Path(__file__).resolve().parents[2] / 'docs/evaluation-panels'
ARMS = ('baseline-global', 'baseline-context', 'combined-global', 'combined-context')


def derive(source, arm):
    manifest = copy.deepcopy(source)
    if arm.startswith('baseline'):
        manifest['policy']['specialists'] = [s for s in manifest['policy']['specialists'] if s['family'] == 'baseline']
        for file in manifest['files']:
            file['required_families'] = ['baseline']
            file['units'] = [u for u in file['units'] if u['specialist'] == 'writing.baseline']
    if arm.endswith('global'):
        route = engine.digest({'word_count': '<=75 short; >75 long', 'contextual': False})
        for spec in manifest['policy']['specialists']:
            spec.update(contexts=['pooled'], routing_sha256=route)
        for file in manifest['files']:
            for unit in file['units']:
                unit.update(context='pooled', routing_sha256=route)
    return manifest


def compare(actual, expected):
    maximum, count = 0., 0
    for key in ('outcomes', 'unit_outcomes'):
        engine.check(len(actual[key]) == len(expected[key]), 'Complete recorded confirmation required')
        for a, b in zip(actual[key], expected[key], strict=True):
            engine.check({k: v for k, v in a.items() if k != 'probability'} ==
                {k: v for k, v in b.items() if k != 'probability'}, 'Source/position/context/status binding changed')
            if a['probability'] is None or b['probability'] is None:
                engine.check(a['probability'] is b['probability'], 'Assessment availability changed')
            else:
                maximum = max(maximum, abs(a['probability'] - b['probability']))
                count += 1
    engine.check(maximum <= LIMIT, 'Real-text calibration numerical agreement exceeded1e-6')
    return maximum, count


def reproduce(output):
    release = json.loads((PUBLIC / 'writing-specialists-v5-release.json').read_text())
    root = PUBLIC.parents[1]
    for path, expected in release['files_sha256'].items():
        engine.check(hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, 'Reviewed source/packet bytes changed')
    packet = json.loads(gzip.decompress((PUBLIC / 'writing-specialists-v5-scalars.json.gz').read_bytes()))
    expected = json.loads(gzip.decompress((PUBLIC / 'writing-specialists-v5-expected.json.gz').read_bytes()))
    maximum, count = 0., 0
    for arm in ARMS:
        manifest = derive(packet, arm)
        engine.check(engine.digest(manifest) == release['manifest_sha256'][arm], 'Arm derivation changed')
        artifact = engine.train(manifest)
        result = engine.evaluate(manifest, artifact)
        difference, current = compare(result, expected[arm])
        maximum, count = max(maximum, difference), count + current
        if arm == 'combined-context':
            changed = copy.deepcopy(manifest)
            for file in changed['files']:
                if file['role'] == 'confirmation':
                    for unit in file['units']:
                        unit['raw'] = 1 - unit['raw']
            engine.check(engine.train(changed) == artifact, 'Confirmation scores fitted parameters')
            changed = copy.deepcopy(manifest)
            file = next(f for f in changed['files'] if f['role'] == 'confirmation')
            file['units'][0].update(status='skipped', raw=None, reason='Missing-evidence adverse replay')
            row = next(r for r in engine.evaluate(changed, artifact)['outcomes'] if r['id'] == file['id'])
            engine.check(row['probability'] is None and row['status'] == 'inconclusive', 'Missing check became originality evidence')
            changed = copy.deepcopy(manifest)
            first = changed['files'][0]
            other = next(f for f in changed['files'] if f['role'] != first['role'])
            other['lineage'] = copy.deepcopy(first['lineage'])
            try:
                engine.audit(changed)
            except ValueError:
                pass
            else:
                raise ValueError('Cross-role real source lineage accepted')
    receipt = {'schema': 'chipmunk.writing-specialists-scalar-reproduction/5',
        'passed': True, 'arms': len(ARMS), 'source_groups': 128, 'constructed_documents': 512,
        'confirmation_documents_per_arm': 128, 'probabilities_compared': count,
        'max_absolute_probability_difference': maximum, 'predeclared_absolute_limit': LIMIT,
        'confirmation_not_fitted': True, 'missing_check_withheld': True, 'cross_role_lineage_refused': True,
        'new_neural_forwards': 0, 'source_text_or_weights_loaded': False,
        'serving_admitted': False,
        'limits': 'Real retained scalar numerical replay only. Does not establish native model equivalence, unseen base training, author independence, broader quality or outside human review.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reproduce(args.output)
