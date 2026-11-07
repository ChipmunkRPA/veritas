"""Model-free independent-host replay of the frozen source-tail comparison."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import tempfile
from pathlib import Path

from app.bench import specialists as engine
from scripts import specialist_source_tail_v1 as study

ROOT = Path(__file__).resolve().parents[2]


def compare(actual, expected, *, limit, at='root'):
    if isinstance(expected, dict):
        engine.check(isinstance(actual, dict) and set(actual) == set(expected), 'Changed result fields at ' + at)
        return max((compare(actual[k], v, limit=limit, at=at + '.' + k) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        engine.check(isinstance(actual, list) and len(actual) == len(expected), 'Changed result denominator at ' + at)
        return max((compare(a, b, limit=limit, at=at + f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected, strict=True))), default=0.)
    if isinstance(expected, float):
        engine.check(engine.finite(actual) and abs(actual - expected) <= limit, 'Numerical agreement exceeded limit at ' + at)
        return abs(actual - expected)
    engine.check(type(actual) is type(expected) and actual == expected, 'Changed count/status/decision at ' + at)
    return 0.


def reproduce(release_path, output):
    release = json.loads(release_path.read_text())
    for name, expected in release['files_sha256'].items():
        path = (ROOT / name).resolve()
        engine.check(path.is_relative_to(ROOT.resolve()) and study.sha(path) == expected, 'Reviewed code/packet bytes changed')
    expected = json.loads((ROOT / release['expected']).read_text())
    with tempfile.TemporaryDirectory(prefix='source-tail-refit-') as directory:
        trial = Path(directory)
        (trial / 'freeze.json').write_bytes((ROOT / release['freeze']).read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            study.fit(trial)
            study.review(trial)
        actual = json.loads((trial / 'result.json').read_text())
        # Artifact bytes can differ with last-bit numerical fits across hosts.
        # Counts, decisions, source bindings and every numeric result still have
        # to agree; this one hash is separately bound in each local fit receipt.
        actual.pop('fit_complete_sha256')
        expected.pop('fit_complete_sha256')
        maximum = compare(actual, expected, limit=release['absolute_numerical_limit'])
        packets = study.check(trial)
        adverse_arms = 0
        for name, arms in packets.items():
            for arm, manifest in arms.items():
                artifact = json.loads((trial / (name + '-' + arm + '-artifact.json')).read_text())
                altered = copy.deepcopy(manifest)
                specs = {s['id']: s for s in manifest['policy']['specialists']}
                for file in altered['files']:
                    if file['role'] == 'confirmation':
                        for unit in file['units']:
                            if unit['status'] == 'ok':
                                low, high = specs[unit['specialist']]['raw_range']
                                unit['raw'] = low + high - unit['raw']
                engine.check(engine.train(altered, min_groups_per_class=25, l2=.001) == artifact,
                    'Reserved responses changed fitted calibration')
                chosen = [{**{k: file[k] for k in ('role', 'label', 'group')}, 'probability': study.predict(file, artifact)}
                    for file in altered['files'] if file['role'] == 'fusion_cal' and file['label'] == 0]
                engine.check(study.point(chosen) == json.loads((trial / (name + '-' + arm + '-operating-point.json')).read_text()),
                    'Reserved responses changed genuine-source operating point')
                adverse_arms += 1
    receipt = {'schema': 'chipmunk.specialist-source-tail-numerical-reproduction/1',
        'studies': len(actual['studies']), 'arms': sum(map(len, actual['studies'].values())),
        'maximum_absolute_numeric_difference': maximum, 'absolute_numerical_limit': release['absolute_numerical_limit'],
        'counts_and_decisions_exact': True, 'reserved_response_parameter_and_point_invariance_arms': adverse_arms,
        'new_neural_forwards': 0, 'raw_media_or_weights_loaded': False, 'serving_admitted': False,
        'scope': 'Numerical refit of consumed-source development scalars only; not native inference, fresh quality or outside human review.'}
    study.write(output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reproduce(args.release, args.output)
