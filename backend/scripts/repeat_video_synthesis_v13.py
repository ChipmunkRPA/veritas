"""Complete sequential replication of an interrupted frozen baseline; no admission.

Keep the failed original journal and successful candidate. Run the unchanged
original baseline on every288 source rows in a new owned worker, with resource
guards. No model, numerical criterion, crop/clock or source subset is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import psutil

from scripts import profile_video_synthesis_v13 as guard

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL_LOCK = '87c8db6e370993cb9aea6c77f5ca1dce45e62246483ac130f169f76e872299ae'
INTERRUPTED = '55de1e6a759a68a37da7d636759aff37d484b7eee724eff6ba1f996a038d7f26'
CANDIDATE = '1f2970c6a31ae37f76e31c1b5d4f3fbda396d1a58c1c1ffef2891a5dbeec60b3'
LIMITS = {**guard.LIMITS, 'timeout_s': 7200, 'source_indices': list(range(288))}
FILES = {'backend/scripts/repeat_video_synthesis_v13.py', 'backend/scripts/profile_video_synthesis_v13.py',
         'backend/tests/test_repeat_video_synthesis_v13.py', 'docs/video-v13-sequential-replication.md'}
POLICY = {
    'complete_original_order': 288, 'baseline_method': 'Unchanged original full/commercial MPSfloat32 adapters, mean of available checks,2CPUthreads',
    'candidate': 'Reuse complete original288-row source/input-bound journal; not a new candidate forward or fresh data',
    'prior_failure': 'Preserve original84-row journal with exit137; never overwrite, append, fit on the prefix or fill gaps from diagnostic',
    'new_attempt': 'Fresh complete baseline journal, one heavy project evaluation at a time; original lock/method/gates unchanged',
    'additional_correspondence': 'All84 prior source identities, check statuses/applicability and raw probabilities repeat; tolerance1e-5 unchanged native reference',
    'resources': LIMITS, 'quality_thresholds_changed': False, 'serving_admitted': False,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins():
    return {name: sha(PROJECT / name) for name in sorted(FILES)}


def source(lock_path):
    from scripts import run_video_demamba_v13 as original

    if sha(lock_path) != ORIGINAL_LOCK:
        raise ValueError('Unchanged complete original execution lock required')
    body = original.check(lock_path)
    failed, candidate = lock_path.parent / 'baseline.jsonl', lock_path.parent / 'candidate.jsonl'
    if sha(failed) != INTERRUPTED or sha(candidate) != CANDIDATE:
        raise ValueError('Preserved original interrupted/successful journals required')
    original.read_scores(candidate, lock_path, body, 'candidate')
    records = [json.loads(line) for line in failed.read_text().splitlines()]
    if records[0]['lock_sha256'] != ORIGINAL_LOCK or records[0]['system'] != 'baseline' or len(records) != 85:
        raise ValueError('Original84-row prefix required')
    return original, body, records[1:]


def freeze(lock_path, output):
    if output.exists():
        raise ValueError('New replication lock required')
    source(lock_path)
    guard.write(output, {'schema': 'chipmunk.video-v13-sequential-replication/1', 'original_lock': str(lock_path.resolve()),
        'original_lock_sha256': ORIGINAL_LOCK, 'interrupted_baseline_sha256': INTERRUPTED, 'successful_candidate_sha256': CANDIDATE,
        'policy': POLICY, 'implementation_sha256': pins(), 'new_baseline_forwards_started': False, 'deployment_eligible': False})


def check(path):
    body = json.loads(path.read_text())
    if (body['schema'] != 'chipmunk.video-v13-sequential-replication/1' or body['policy'] != POLICY or body['implementation_sha256'] != pins()
            or body['original_lock_sha256'] != ORIGINAL_LOCK or body['interrupted_baseline_sha256'] != INTERRUPTED
            or body['successful_candidate_sha256'] != CANDIDATE or body['new_baseline_forwards_started'] is not False or body['deployment_eligible'] is not False):
        raise ValueError('Unchanged prospective replication policy required')
    return body, source(Path(body['original_lock']))


def public(path, output):
    body, _ = check(path)
    value = {k: v for k, v in body.items() if k != 'original_lock'}
    value['implementation_sha256'] = [{'path': k, 'sha256': v} for k, v in value['implementation_sha256'].items()]
    guard.write(output, {'schema': 'chipmunk.video-v13-public-sequential-replication/1', 'private_replication_lock_sha256': sha(path), 'lock': value})


def correspondence(previous, repeated):
    if len(previous) != 84 or len(repeated) != 288:
        raise ValueError('Every prior and repeated source required')
    largest = 0.
    for old, new in zip(previous, repeated[:84], strict=True):
        if any(old[k] != new[k] for k in ('dataset', 'id', 'variant', 'sha256', 'label', 'split', 'status')) or set(old['checks']) != set(new['checks']):
            raise ValueError('Prior source/status correspondence changed')
        for name, expected in old['checks'].items():
            actual = new['checks'][name]
            if actual['status'] != expected['status'] or (actual['prob'] is None) != (expected['prob'] is None):
                raise ValueError('Prior check applicability changed')
            if expected['prob'] is not None:
                difference = abs(actual['prob'] - expected['prob'])
                if not math.isfinite(difference) or difference > 1e-5:
                    raise ValueError('Prior raw check response differs')
                largest = max(largest, difference)
    return largest


def worker(path, output):
    try:
        body, (original, original_body, previous) = check(path)
        original_path = Path(body['original_lock'])
        original.score_baseline(original_path, output / 'baseline.jsonl')
        repeated = original.read_scores(output / 'baseline.jsonl', original_path, original_body, 'baseline')
        maximum = correspondence(previous, repeated)
        guard.write(output / 'child-result.json', {'completed': True, 'replication_lock_sha256': sha(path), 'baseline_sha256': sha(output / 'baseline.jsonl'),
            'planned_inputs': len(repeated), 'prior_inputs_reproduced': 84, 'max_prior_probability_difference': maximum,
            'new_calibration_fitted': False, 'deployment_eligible': False})
    except Exception as error:
        traceback.print_exc()
        guard.write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__, 'new_calibration_fitted': False, 'deployment_eligible': False})
    (output / 'ready').touch()
    deadline = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.05)


def execute(path, output):
    check(path)
    modules = {'scripts.run_audio_separation_v14', 'scripts.run_video_demamba_v13', 'scripts.profile_video_synthesis_v13', 'scripts.repeat_video_synthesis_v13'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        command = process.info['cmdline'] or []
        if '-m' in command and any(module in command for module in modules):
            raise ValueError('Another heavy project evaluation is active; run sequentially')
    guard.profile(path, output, worker_module='scripts.repeat_video_synthesis_v13', limits=LIMITS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'public', 'execute', 'worker'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'public': public, 'execute': execute, 'worker': worker}[args.stage](args.input, args.output)
