"""Replay the original new-method fits/checks from exact public scalar inputs."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import tempfile
from pathlib import Path

from scripts import run_preserved_video_v22 as study

ROOT = Path(__file__).resolve().parents[2]


def close(actual, expected, *, path='', limit=1e-6, record=None):
    if record is None:
        record = {'compared_floats': 0, 'maximum_absolute_difference': 0.}
    if type(actual) is float and type(expected) is float:
        difference = abs(actual - expected)
        study.legacy.check(difference <= limit, 'Numerical replay disagreement: ' + path)
        record['compared_floats'] += 1
        record['maximum_absolute_difference'] = max(record['maximum_absolute_difference'], difference)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        study.legacy.check(actual.keys() == expected.keys(), 'Replay object scope changed: ' + path)
        for key in actual:
            close(actual[key], expected[key], path=path + '/' + key, limit=limit, record=record)
    elif isinstance(actual, list) and isinstance(expected, list):
        study.legacy.check(len(actual) == len(expected), 'Replay source/output count changed: ' + path)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close(a, b, path=path + '/' + str(i), limit=limit, record=record)
    else:
        study.legacy.check(type(actual) is type(expected) and actual == expected, 'Replay nonnumeric status/scope/count/decision changed: ' + path)
    return record


def normal(value, kind):
    result = copy.deepcopy(value)
    if kind == 'artifact':
        result.pop('sha256')
        models = list(result['specialists'].values())
        for entry in result['combiners'].values():
            models += [entry[k] for k in ('response', 'calibration') if k in entry]
        for model in models:
            if 'numerical_check' in model:
                # Iteration count is a diagnostic, not a detection/source count.
                model['numerical_check'].pop('iterations')
    elif kind == 'evaluation':
        result.pop('artifact_sha256')
    elif kind == 'solver-audit':
        result.pop('journal_sha256')
    elif kind == 'solver-row' and result.get('new_fit', {}).get('numerical_check'):
        result['new_fit']['numerical_check'].pop('iterations')
    return result


def reproduce(release_path, output):
    release = study.read(release_path)
    study.legacy.check(release['schema'] == 'chipmunk.preserved-evidence-release/22'
        and release['serving_admitted'] is False and release['absolute_limit'] == 1e-6, 'Exact research release required')
    for row in release['files_sha256']:
        path = (ROOT / row['path']).resolve()
        study.legacy.check(path.is_relative_to(ROOT) and study.sha(path) == row['sha256'], 'Frozen scalar/source bytes changed')
    expected = json.loads(gzip.decompress((ROOT / release['expected']).read_bytes()))
    record = {'compared_floats': 0, 'maximum_absolute_difference': 0.}
    # Preserve a failed attempt's partial fitting journals; remove only success.
    folder = Path(tempfile.mkdtemp(prefix='chipmunk-preserved-v22-numerical-', dir=output.parent))
    try:
        (folder / 'freeze.json').write_bytes((ROOT / release['freeze']).read_bytes())
        study.solver_audit(folder)
        close(normal(study.read(folder / 'solver-audit.json'), 'solver-audit'),
            normal(expected['solver-audit.json'], 'solver-audit'), record=record)
        rows = [json.loads(line) for line in (folder / 'solver-audit.jsonl').read_text().splitlines()]
        close([normal(row, 'solver-row') for row in rows], [normal(row, 'solver-row') for row in expected['solver-audit-rows']], record=record)
        study.fit(folder)
        study.evaluate(folder)
        for arm in study.ARMS:
            for suffix in ('artifact', 'point', 'evaluation'):
                name = arm + '-' + suffix + '.json'
                close(normal(study.read(folder / name), suffix), normal(expected[name], suffix), record=record)
        close(study.read(folder / 'appearance-mapped-point.json'), expected['appearance-mapped-point.json'], record=record)
        close(study.read(folder / 'result.json'), expected['result.json'], record=record)
        receipt = {'schema': 'chipmunk.preserved-evidence-numerical-reproduction/22', **record,
            'video_arms': len(study.ARMS), 'real_training_problems_checked': len(rows),
            'statuses_source_counts_and_decisions_exact': True, 'new_fit_checks_and_adverse_checks_passed': True,
            'new_neural_forwards': 0, 'raw_media_or_model_weights_loaded': False,
            'serving_admitted': False, 'outside_human_reviews': 0,
            'limits': 'Immutable consumed-source numerical replay; not native media execution or fresh detection quality.'}
        study.write(output, receipt)
    except BaseException as error:
        study.write(folder / 'replay-failure.json', {'error_type': type(error).__name__, 'error': str(error),
            'frozen_science_modified': False, 'new_neural_forwards': 0})
        raise
    else:
        import shutil

        shutil.rmtree(folder)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    study.legacy.check(not args.output.exists(), 'Preserve prior numerical replay output')
    reproduce(args.release.resolve(), args.output.resolve())
