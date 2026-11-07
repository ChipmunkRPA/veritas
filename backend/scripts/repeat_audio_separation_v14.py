"""New full audio extraction, unchanged method, guarded resources and stderr capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import psutil

from scripts import profile_video_synthesis_v13 as guard
from scripts import run_audio_separation_v14 as original

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL = '93223afb27132aa3557b70f5c433aec126409ebd06d2beea6d5f34b3551e744b'
PARTIAL = '7084fd34ee80948ec1180f9dc848b1235665df29580006e148703c3c052a7b9b'
FILES = {'backend/scripts/repeat_audio_separation_v14.py', 'docs/audio-v14-full-replication.md',
         'backend/scripts/profile_video_synthesis_v13.py', 'backend/tests/test_repeat_audio_separation_v14.py'}
POLICY = {
    'operation': 'New full4248 ordered extraction from input0, original CPUfloat32/four threads and sole original inter-op owner; no retries, skip, denoising, PCM substitution, scoring or feature changes.',
    'parity': 'All3818 prior successful/unassessed records must match exactly excluding runtime_s only; original failed3819-row attempt retained unchanged. No fitting partial/new failed journal.',
    'diagnostics': 'Capture original failure exception chain and complete CalledProcessError.stderr privately, no stdout or raw audio. Original exception printing, failed row and stop behavior preserved.',
    'limits': {**{k: v for k, v in guard.LIMITS.items() if k != 'source_indices'}, 'timeout_s': 21600},
    'reserved_confirmation_scored': False, 'serving_admitted': False,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins():
    return {name: sha(PROJECT / name) for name in sorted(FILES)}


def source(path):
    if sha(path) != ORIGINAL or sha(path.parent / 'features.jsonl') != PARTIAL:
        raise ValueError('Unchanged original lock and terminal failed extraction required')
    result = original.check(path)
    meta, rows = original.load_rows(path.parent / 'features.jsonl')
    if (meta != original.header(path) or len(rows) != 3819 or rows[-1]['status'] != 'failed'
            or any(r['status'] not in ('ok', 'not_applicable') for r in rows[:-1])):
        raise ValueError('Preserve original3819 rows and terminal failure')
    return result, rows[:-1]


def freeze(path, output):
    if output.exists():
        raise ValueError('New unexecuted operational freeze required')
    source(path)
    guard.write(output, {'schema': 'chipmunk.audio-full-replication-lock/14', 'original': str(path.resolve()),
        'original_sha256': ORIGINAL, 'partial_sha256': PARTIAL, 'policy': POLICY,
        'implementation_sha256': pins(), 'new_forwards_started': False, 'deployment_eligible': False})


def check(path):
    body = json.loads(path.read_text())
    if (body['schema'] != 'chipmunk.audio-full-replication-lock/14' or body['policy'] != POLICY
            or body['implementation_sha256'] != pins() or body['original_sha256'] != ORIGINAL
            or body['partial_sha256'] != PARTIAL or body['new_forwards_started'] is not False
            or body['deployment_eligible'] is not False):
        raise ValueError('Original method and prospective operational freeze changed')
    return body, source(Path(body['original']))


def public(path, output):
    body, _ = check(path)
    clean = {k: v for k, v in body.items() if k != 'original'}
    clean['implementation_sha256'] = [{'path': k, 'sha256': v} for k, v in clean['implementation_sha256'].items()]
    guard.write(output, {'schema': 'chipmunk.audio-full-replication-public-freeze/14',
                        'private_lock_sha256': sha(path), 'lock': clean})


def failure_diagnostics(error, output):
    """Private diagnostics only; never persist decoded stdout/audio."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        row = {'error_type': type(error).__name__}
        if isinstance(error, subprocess.CalledProcessError):
            row['returncode'] = error.returncode
            if error.stderr is not None:
                raw = error.stderr if isinstance(error.stderr, bytes) else error.stderr.encode('utf-8')
                name = f'failure-{len(chain)}.stderr.bin'
                (output / name).write_bytes(raw)
                row.update(stderr_file=name, stderr_bytes=len(raw), stderr_sha256=hashlib.sha256(raw).hexdigest())
        chain.append(row)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    guard.write(output / 'failure-diagnostics.json', {'exception_chain': chain, 'raw_audio_or_stdout_saved': False})


def worker(path, output):
    saved_print = traceback.print_exc
    try:
        body, (_, previous) = check(path)
        target = output / 'features.jsonl'
        if target.exists():
            raise ValueError('Never resume or overwrite an extraction attempt')

        def captured_print(*args, **kwargs):
            try:
                failure_diagnostics(sys.exception(), output)
            except Exception as error:
                print('Failure diagnostic capture failed: ' + type(error).__name__, flush=True)
            saved_print(*args, **kwargs)

        traceback.print_exc = captured_print
        try:
            original.extract(Path(body['original']), target)
        finally:
            traceback.print_exc = saved_print
        original.feature_rows(Path(body['original']), target)
        _, records = original.load_rows(target)
        exact = all({k: v for k, v in a.items() if k != 'runtime_s'} == {k: v for k, v in b.items() if k != 'runtime_s'}
                    for a, b in zip(records, previous, strict=False))
        counts = Counter(r['status'] for r in records)
        if not exact or len(records) != 4248 or counts != {'ok': 4186, 'not_applicable': 62}:
            raise ValueError('Full extraction and all3818 prior semantic records must reproduce exactly')
        guard.write(output / 'child-result.json', {'completed': True, 'records': len(records), 'counts': counts,
            'prior_records_reproduced': len(previous), 'prior_semantic_records_exact': exact,
            'features_sha256': sha(target), 'fitting_started': False, 'serving_admitted': False})
    except Exception as error:
        saved_print()
        guard.write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__,
            'fitting_started': False, 'serving_admitted': False})
    finally:
        traceback.print_exc = saved_print
    (output / 'ready').touch()
    deadline = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.05)


def execute(path, output):
    check(path)
    modules = {'scripts.run_audio_separation_v14', 'scripts.repeat_audio_separation_v14', 'scripts.run_video_demamba_v13',
               'scripts.repeat_video_synthesis_v13', 'scripts.repeat_video_release_v14', 'scripts.verify_video_native_v14'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and '-m' in command and any(m in command for m in modules):
            raise ValueError('Run heavy studies sequentially')
    guard.profile(path, output, worker_module='scripts.repeat_audio_separation_v14', limits=POLICY['limits'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'public', 'worker', 'execute'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'public': public, 'worker': worker, 'execute': execute}[args.mode](args.input, args.output)
