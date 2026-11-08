"""Prospective storage-only amendment; original failed science remains immutable."""
from __future__ import annotations

import argparse
import ast
import json
import string
from pathlib import Path

import numpy as np

from scripts import run_ltx_marking_audio_v1 as old

PREVIOUS = old.ROOT / 'backend/data/independent-evaluation/ltx-marking-audio-v1'
PINS = old.PINS + ('backend/scripts/continue_ltx_marking_audio_v1.py',
    'backend/scripts/ltx_marking_continuation_runtime_v1.py', 'backend/tests/test_ltx_marking_continuation_v1.py',
    'docs/ltx-marking-audio-v1-storage-amendment.md')
AMENDMENT = {'change': 'Encode the single slash in a codec case identifier as double underscore before constructing its packet filename.',
    'unchanged': 'All80 cases, waveforms,payloads,noise,codecs,points,model calls,arithmetic and no-authentication policy.',
    'previous': 'Original six-case prefix and exit1 export failure retained byte-for-byte; no implicit restart.',
    'check': 'Require all six earlier waveforms,probabilities,messages and interpreted results to reproduce exactly.'}


def case_id(value):
    old.base.require(isinstance(value, str) and value and all(c in string.ascii_lowercase + string.digits + '-/' for c in value)
                     and value.count('/') <= 1, 'Known case identifier grammar required')
    return value.replace('/', '__')


def amended_run():
    tree = ast.parse((old.ROOT / 'backend/scripts/run_ltx_marking_audio_v1.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    count = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'identity':
            old.base.require(isinstance(node.value, ast.JoinedStr), 'Exact frozen packet identifier expression required')
            node.value = ast.Call(func=ast.Name(id='case_id', ctx=ast.Load()), args=[node.value], keywords=[])
            count += 1
    old.base.require(count == 1, 'Exactly one storage expression may change')
    namespace = {**vars(old), 'case_id': case_id, 'check': check}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<storage-only-ltx-amendment>', 'exec'), namespace)
    return namespace['run']


def identity():
    old.check(PREVIOUS)
    profile = old.base.read(PREVIOUS / 'profile/profile.json')
    rows = [json.loads(s) for s in (PREVIOUS / 'journal.jsonl').read_text().splitlines()]
    old.base.require(profile['child_exit_code'] == 1 and profile['guard'] is None and len(rows) == 6
        and not (PREVIOUS / 'result.json').exists() and 'FileNotFoundError' in (PREVIOUS / 'profile/worker.log').read_text(), 'Original terminal export failure and complete prefix required')
    files = [PREVIOUS / n for n in ('freeze.json', 'journal.jsonl', 'profile/profile.json', 'profile/worker.log')]
    files += [PREVIOUS / 'private' / r['packet'] for r in rows]
    return {'original_identity': old.identity(), 'amendment': AMENDMENT,
        'code_sha256': {p: old.base.digest(old.ROOT / p) for p in PINS},
        'prior_attempt_sha256': {str(p.relative_to(PREVIOUS)): old.base.digest(p) for p in files},
        'prior_persisted_cases': 6, 'prior_final_model_call_counter': None,
        'prior_final_model_call_counter_note': 'Run stopped before final counters were written; do not invent an exact recorded total.'}


def freeze(trial):
    value = identity()
    trial.mkdir(exist_ok=False)
    old.base.write(trial / 'freeze.json', value)
    old.base.write(old.ROOT / 'docs/evaluation-panels/ltx-marking-audio-v1-continuation-freeze.json',
        {**value, 'private_freeze_sha256': old.base.digest(trial / 'freeze.json'), 'new_attempt_media_forwards': 0})


def check(trial):
    old.base.require(old.base.read(trial / 'freeze.json') == identity(), 'Frozen amendment or original failure changed')


def run(trial):
    check(trial)
    amended_run()(trial)
    original = [json.loads(s) for s in (PREVIOUS / 'journal.jsonl').read_text().splitlines()]
    current = {r['id']: r for r in (json.loads(s) for s in (trial / 'journal.jsonl').read_text().splitlines())}
    for row in original:
        new = current[row['id']]
        old.base.require(row['result'] == new['result'] and row['pcm_sha256'] == new['pcm_sha256'], 'Original completed native result changed')
        with np.load(PREVIOUS / 'private' / row['packet'], allow_pickle=False) as left, np.load(trial / 'private' / new['packet'], allow_pickle=False) as right:
            old.base.require(set(left.files) == set(right.files) and all(np.array_equal(left[k], right[k]) for k in left.files), 'Original prefix arrays did not reproduce exactly')
    old.base.write(trial / 'original-prefix-parity.json', {'cases': 6, 'waveforms_probabilities_messages_and_results_exact': True,
        'previous_journal_sha256': old.base.digest(PREVIOUS / 'journal.jsonl'),
        'new_journal_sha256': old.base.digest(trial / 'journal.jsonl'), 'original_failure_unchanged': True})
    print({'completed_cases': 80, 'original_prefix_cases_exact': 6, 'serving_admitted': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    freeze(args.trial) if args.action == 'freeze' else run(args.trial)
