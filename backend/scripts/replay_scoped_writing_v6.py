"""Refit the immutable passage study, comparing all parameters and decisions.

The external harness/CI is not part of the original frozen science. No native
model, raw text, customer file, face or voice representation is loaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'docs/evaluation-panels/scoped-writing-v6-release.json'


def safe(name):
    if not isinstance(name, str) or not name or name.startswith('-'):
        raise ValueError('Explicit repository-relative source path required')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Invalid repository source path')
    return name


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Different object keys: ' + path)
        return max([0.] + [compare(actual[k], v, path + '.' + k) for k, v in expected.items()])
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Different planned denominator: ' + path)
        return max([0.] + [compare(a, b, path + '.' + str(i)) for i, (a, b) in enumerate(zip(actual, expected, strict=True))])
    if type(expected) is float:
        if type(actual) is not float or not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError('Different finite numerical availability: ' + path)
        return abs(actual - expected)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('Different exact status/count/source/decision: ' + path)
    return 0.


def packet(trial, arms):
    result = {'result': json.loads((trial / 'result.json').read_text()), 'arms': {}}
    for arm in arms:
        values = {kind: json.loads((trial / (arm + '-' + kind + '.json')).read_text())
            for kind in ('artifact', 'point', 'confirmation')}
        values['artifact'].pop('sha256')
        values['confirmation'].pop('artifact_sha256')
        result['arms'][arm] = values
    return result


def replay(output):
    if output.exists():
        raise FileExistsError('Retain the previous numerical replay')
    release = json.loads((ROOT / RELEASE).read_text())
    science = release['science_commit']

    def blob(name):
        return subprocess.check_output(['git', 'show', science + ':' + safe(name)], cwd=ROOT)

    frozen_name = 'docs/evaluation-panels/scoped-writing-v6-freeze.json'
    frozen_bytes = blob(frozen_name)
    if hashlib.sha256(frozen_bytes).hexdigest() != release['freeze_sha256']:
        raise ValueError('Immutable science freeze changed')
    frozen = json.loads(frozen_bytes)
    prior_name = 'docs/evaluation-panels/writing-specialists-v5-release.json'
    prior = json.loads(blob(prior_name))
    names = {frozen_name, prior_name, *map(safe, frozen['code_sha256']), *map(safe, prior['files_sha256'])}
    for name in list(names):
        if name.endswith('.py'):
            parent = PurePosixPath(name).parent
            while str(parent) not in ('.', 'backend'):
                candidate = str(parent / '__init__.py')
                if subprocess.run(['git', 'cat-file', '-e', science + ':' + candidate], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    names.add(candidate)
                parent = parent.parent
    expected_file = ROOT / safe(release['expected'])
    if hashlib.sha256(expected_file.read_bytes()).hexdigest() != release['expected_sha256']:
        raise ValueError('Expected scalar/parameter packet changed')
    expected = json.loads(gzip.decompress(expected_file.read_bytes()))
    with tempfile.TemporaryDirectory(prefix='chipmunk-scoped-writing-replay-') as temp:
        folder = Path(temp)
        archive, tree, trial = folder / 'science.tar', folder / 'tree', folder / 'trial'
        tree.mkdir()
        trial.mkdir()
        subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), science,
            '--', *sorted(names)], cwd=ROOT, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter='data')
        (trial / 'freeze.json').write_bytes(frozen_bytes)
        environment = {**os.environ, 'PYTHONPATH': str(tree / 'backend'), 'OPENBLAS_NUM_THREADS': '1'}
        for stage in ('fit', 'review'):
            subprocess.run([sys.executable, '-m', 'scripts.scoped_writing_v6', stage, str(trial)],
                cwd=tree, env=environment, check=True, capture_output=True, text=True)
        maximum = compare(packet(trial, release['arms']), expected)
        if maximum > release['absolute_numerical_limit']:
            raise ValueError('Numerical refit exceeds the fixed tolerance')
        receipt = {'schema': 'chipmunk.scoped-writing-numerical-replay/6', 'science_commit': science,
            'freeze_sha256': release['freeze_sha256'], 'arms': len(release['arms']),
            'selected_repository_files': len(names), 'maximum_absolute_numerical_difference': maximum,
            'absolute_numerical_limit': release['absolute_numerical_limit'], 'all_counts_statuses_decisions_exact': True,
            'confirmation_invariance_and_missing_check_tests': 'passed_all_eight_arms',
            'new_neural_forwards': 0, 'serving_admitted': False,
            'scope': 'Immutable consumed-source scalar/parameter replay; not fresh detection, native input reproduction or outside human review.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    replay(parser.parse_args().output)
