"""Complete sequential baseline repeat with post-analysis object collection.

Keep original scoring/inputs/gates and both interrupted attempts. The sole new
operation is gc.collect after each scorer call, never a prediction fallback.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import psutil

from scripts import repeat_video_synthesis_v13 as prior

PROJECT = Path(__file__).resolve().parents[2]
DIAG = 'docs/evaluation-panels/video-v14-release-diagnosis.json'
SECOND_PARTIAL = 'd0fd36769b3dc83caa5a59b278ad9d12aa4e24893519650dd94584fe6735b1b3'
FILES = {'backend/scripts/repeat_video_release_v14.py', 'backend/scripts/diagnose_video_release_v14.py',
         'backend/tests/test_repeat_video_release_v14.py', 'docs/video-v14-object-release-replication.md'}
POLICY = {**prior.POLICY,
          'new_attempt': 'Fresh complete288-row baseline, same order/method; gc.collect after each scorer call. No MPSempty_cache, model reset, input/score change or exception fallback.',
          'cleanup_audit': 'Exactly284 ready-input scorer attempts with unchanged task identity; all4 originally unavailable inputs stay explicit. Guard/event/failure invalidates completion.',
          'first_two_attempts': 'Preserve84row exit137 and42row fixed-RSS guard stop; never merge, overwrite or fill partial journals.',
          'full_recipe': 'Posthoc six-input cleanup diagnosis informs operational release only; complete288 comparison still requires original calibration/quality/native rules and all84 prior raw/status correspondence.'}


def score_and_collect(score, task, cleanup=gc.collect):
    """Return the exact score object or propagate its exception; always collect."""
    try:
        return score(task)
    finally:
        cleanup()


def pins():
    return {name: prior.sha(PROJECT / name) for name in sorted(FILES)}


def sources(original_path):
    result = prior.source(original_path)
    if prior.sha(original_path.parent / 'sequential-repeat-1/baseline.jsonl') != SECOND_PARTIAL:
        raise ValueError('Preserve second interrupted journal unchanged')
    receipt = json.loads((PROJECT / DIAG).read_text())
    if (receipt['completed'] is not True or receipt['guard'] is not None or receipt['child_exit_code'] != 0
            or receipt['indices'] != [0, 1, 40, 41, 42, 84] or not receipt['all_reference_statuses_exact']
            or not receipt['all_reference_probabilities_exact']
            or receipt['diagnostic_code_sha256'] != prior.sha(PROJECT / 'backend/scripts/diagnose_video_release_v14.py')):
        raise ValueError('Complete unchanged cleanup diagnosis required')
    return result


def freeze(original_path, output):
    if output.exists():
        raise ValueError('New never-executed replication lock required')
    sources(original_path)
    prior.guard.write(output, {'schema': 'chipmunk.video-object-release-replication/14', 'original_lock': str(original_path.resolve()),
        'original_lock_sha256': prior.ORIGINAL_LOCK, 'original_partial_sha256': prior.INTERRUPTED,
        'second_partial_sha256': SECOND_PARTIAL, 'candidate_sha256': prior.CANDIDATE, 'policy': POLICY,
        'implementation_sha256': pins(), 'prior_wrapper_sha256': prior.pins(),
        'diagnostic_receipt_sha256': prior.sha(PROJECT / DIAG), 'new_forwards_started': False, 'deployment_eligible': False})


def check(path):
    body = json.loads(path.read_text())
    if (body['schema'] != 'chipmunk.video-object-release-replication/14' or body['policy'] != POLICY or body['implementation_sha256'] != pins()
            or body['prior_wrapper_sha256'] != prior.pins() or body['original_lock_sha256'] != prior.ORIGINAL_LOCK
            or body['original_partial_sha256'] != prior.INTERRUPTED or body['second_partial_sha256'] != SECOND_PARTIAL
            or body['candidate_sha256'] != prior.CANDIDATE or body['diagnostic_receipt_sha256'] != prior.sha(PROJECT / DIAG)
            or body['new_forwards_started'] is not False or body['deployment_eligible'] is not False):
        raise ValueError('Unchanged prospective complete release-replication freeze required')
    return body, sources(Path(body['original_lock']))


def public(path, output):
    body, _ = check(path)
    value = {k: v for k, v in body.items() if k != 'original_lock'}
    for field in ('implementation_sha256', 'prior_wrapper_sha256'):
        value[field] = [{'path': k, 'sha256': v} for k, v in value[field].items()]
    prior.guard.write(output, {'schema': 'chipmunk.video-object-release-public-freeze/14', 'private_lock_sha256': prior.sha(path), 'lock': value})


def worker(path, output):
    import torch

    audit = []
    try:
        lock, (original, original_body, previous) = check(path)
        factory = original.baseline_scorer
        process = psutil.Process()

        def wrapped_factory(body, scratch):
            scorer = factory(body, scratch)
            original_score = scorer.score

            def cleaned_score(task):
                before = process.memory_info().rss

                def collect():
                    objects = gc.collect()
                    row = {'task': list(task.key), 'objects_collected': objects, 'rss_before_scorer_bytes': before,
                           'rss_after_collection_bytes': process.memory_info().rss,
                           'mps_current_bytes': torch.mps.current_allocated_memory(), 'mps_driver_bytes': torch.mps.driver_allocated_memory()}
                    audit.append(row)
                    with (output / 'collection.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')

                return score_and_collect(original_score, task, collect)

            scorer.score = cleaned_score
            return scorer

        original.baseline_scorer = wrapped_factory
        try:
            original_path = Path(lock['original_lock'])
            original.score_baseline(original_path, output / 'baseline.jsonl')
        finally:
            original.baseline_scorer = factory
        rows = original.read_scores(output / 'baseline.jsonl', original_path, original_body, 'baseline')
        largest = prior.correspondence(previous, rows)
        expected = [[r['id'], r['variant'], 'none'] for r in original_body['rows'] if r['status'] == 'ready']
        if len(audit) != 284 or [r['task'] for r in audit] != expected:
            raise ValueError('Every original ready-input scorer/cleanup attempt required')
        prior.guard.write(output / 'child-result.json', {'completed': True, 'replication_lock_sha256': prior.sha(path),
            'baseline_sha256': prior.sha(output / 'baseline.jsonl'), 'collection_sha256': prior.sha(output / 'collection.jsonl'),
            'planned_inputs': 288, 'collection_attempts': 284, 'prior_inputs_reproduced': 84, 'max_prior_raw_difference': largest,
            'new_calibration_fitted': False, 'deployment_eligible': False})
    except Exception as error:
        traceback.print_exc()
        prior.guard.write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__,
                    'collection_attempts': len(audit), 'new_calibration_fitted': False, 'deployment_eligible': False})
    (output / 'ready').touch()
    until = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < until:
        time.sleep(.05)


def execute(path, output):
    check(path)
    modules = {'scripts.run_audio_separation_v14', 'scripts.run_video_demamba_v13', 'scripts.repeat_video_synthesis_v13',
               'scripts.profile_video_synthesis_v13', 'scripts.diagnose_video_release_v14', 'scripts.repeat_video_release_v14'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and '-m' in command and any(m in command for m in modules):
            raise ValueError('Run heavy evaluations sequentially')
    prior.guard.profile(path, output, worker_module='scripts.repeat_video_release_v14', limits=prior.LIMITS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'public', 'worker', 'execute'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'public': public, 'worker': worker, 'execute': execute}[args.stage](args.input, args.output)
