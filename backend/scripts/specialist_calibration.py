"""Freeze, fit and test scoped specialists without starting the serving app."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from app.bench import specialists as engine


def implementation():
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
        [('engine', engine.__file__), ('command', __file__)]}


def freeze(manifest, minimum, l2):
    engine.audit(manifest)
    engine.check(type(minimum) is int and minimum >= 2 and engine.finite(l2) and l2 > 0, 'Explicit fitting options required')
    value = {'schema': 'chipmunk.specialist-fit-freeze/1', 'plan_sha256': engine.plan_digest(manifest),
        'training_sha256': engine.digest([f for f in manifest['files'] if f['role'] != 'confirmation']),
        'implementation_sha256': implementation(),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'minimum_groups_per_class': minimum, 'l2': l2, 'serving_admitted': False}
    return {**value, 'sha256': engine.digest(value)}


def fit(manifest, lock):
    engine.check(lock.get('schema') == 'chipmunk.specialist-fit-freeze/1'
        and lock.get('sha256') == engine.digest({k: v for k, v in lock.items() if k != 'sha256'}), 'Exact prospective fit freeze required')
    actual = freeze(manifest, lock['minimum_groups_per_class'], lock['l2'])
    engine.check(actual == lock, 'Plan, fitting/calibration observations, code or runtime changed after freeze')
    return engine.train(manifest, min_groups_per_class=lock['minimum_groups_per_class'], l2=lock['l2'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'freeze', 'fit', 'evaluate'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--minimum-groups-per-class', type=int, default=25)
    parser.add_argument('--l2', type=float, default=.001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse overwrite of a prior calibration or test receipt')
    document = json.loads(args.manifest.read_text())
    if args.mode == 'audit':
        result = engine.audit(document)
    elif args.mode == 'freeze':
        result = freeze(document, args.minimum_groups_per_class, args.l2)
    elif args.mode == 'fit':
        if args.frozen is None:
            parser.error('Published fitting freeze required')
        result = fit(document, json.loads(args.frozen.read_text()))
    else:
        if args.artifact is None:
            parser.error('Exact fitted artifact required')
        result = engine.evaluate(document, json.loads(args.artifact.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'schema': result.get('schema'), 'output': str(args.output), 'serving_admitted': False}), flush=True)
