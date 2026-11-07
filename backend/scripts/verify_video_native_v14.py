"""Guard the original sixteen-case native audit with post-scorer collection.

No frame, clock, tensor, tolerance, scoring or gate change. Bind the completed
baseline/collection proof and original candidate before any native forward.
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

from scripts import repeat_video_release_v14 as release

PROJECT = Path(__file__).resolve().parents[2]
FILES = {'backend/scripts/verify_video_native_v14.py', 'docs/video-v14-native-audit.md'}
COMPLETED_FILES = {'profile.json', 'child-result.json', 'baseline.jsonl', 'collection.jsonl', 'summary.json'}
POLICY = {'source': 'Original16 response-independent dataset/truth/format/role cases; original RGB/tensor exactness, PTS1.1e-6s and output1e-5 tolerances unchanged',
          'operation': 'Original full compare reconstruction and native function; only gc.collect after original baseline scorer returns; no diagnostic gap filling or fallback',
          'quality': 'Original development requirements remain decisive; a failed quality result cannot be rescued by native replay',
          'limits': {**release.prior.LIMITS, 'timeout_s': 900}, 'serving_admitted': False}


def pins():
    return {name: release.prior.sha(PROJECT / name) for name in sorted(FILES)}


def completed(replication_path):
    lock, data = release.check(replication_path)
    root = replication_path.parent / 'object-release-repeat-1'
    profile = json.loads((root / 'profile.json').read_text())
    child = json.loads((root / 'child-result.json').read_text())
    if (profile['guard'] is not None or profile['child_exit_code'] != 0 or child['completed'] is not True
            or child['planned_inputs'] != 288 or child['collection_attempts'] != 284
            or child['prior_inputs_reproduced'] != 84 or child['max_prior_raw_difference'] > 1e-5
            or child['baseline_sha256'] != release.prior.sha(root / 'baseline.jsonl')
            or child['collection_sha256'] != release.prior.sha(root / 'collection.jsonl')):
        raise ValueError('Completed guarded full baseline and prior-input correspondence required')
    return lock, data, root


def freeze(replication_path, output):
    if output.exists():
        raise ValueError('New native-audit lock required')
    _, _, root = completed(replication_path)
    files = {name: release.prior.sha(root / name) for name in sorted(COMPLETED_FILES)}
    release.prior.guard.write(output, {'schema': 'chipmunk.video-native-release-lock/14', 'replication': str(replication_path.resolve()),
        'replication_sha256': release.prior.sha(replication_path), 'implementation_sha256': pins(), 'policy': POLICY,
        'files_sha256': files, 'native_forwards_started': False, 'deployment_eligible': False})


def check(path):
    body = json.loads(path.read_text())
    if (body['schema'] != 'chipmunk.video-native-release-lock/14' or body['policy'] != POLICY or body['implementation_sha256'] != pins()
            or body['replication_sha256'] != release.prior.sha(Path(body['replication']))
            or set(body['files_sha256']) != COMPLETED_FILES
            or body['native_forwards_started'] is not False or body['deployment_eligible'] is not False):
        raise ValueError('Unchanged prospective native-audit freeze required')
    result = completed(Path(body['replication']))
    if any(release.prior.sha(result[2] / name) != digest for name, digest in body['files_sha256'].items()):
        raise ValueError('Completed comparison/source receipts changed')
    return body, result


def public(path, output):
    body, _ = check(path)
    clean = {k: v for k, v in body.items() if k != 'replication'}
    clean['implementation_sha256'] = [{'path': k, 'sha256': v} for k, v in clean['implementation_sha256'].items()]
    release.prior.guard.write(output, {'schema': 'chipmunk.video-native-release-public-freeze/14', 'private_lock_sha256': release.prior.sha(path), 'lock': clean})


def worker(path, output):
    try:
        _, (replication, (original, _, _), root) = check(path)
        factory = original.baseline_scorer

        def collected_factory(body, scratch):
            scorer = factory(body, scratch)
            score = scorer.score
            scorer.score = lambda task: release.score_and_collect(score, task, gc.collect)
            return scorer

        original.baseline_scorer = collected_factory
        try:
            original.audit_native(Path(replication['original_lock']), root / 'baseline.jsonl', root.parent / 'candidate.jsonl', root / 'summary.json', output / 'native-review.json')
        finally:
            original.baseline_scorer = factory
        receipt = json.loads((output / 'native-review.json').read_text())
        release.prior.guard.write(output / 'child-result.json', {'completed': len(receipt['cases']) == 16,
            'native_sha256': release.prior.sha(output / 'native-review.json'), 'native_passed': receipt['gates']['native_correspondence'],
            'advance_to_fresh_confirmation': receipt['advance_to_fresh_confirmation'], 'deployment_eligible': False})
    except Exception as error:
        traceback.print_exc()
        release.prior.guard.write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__, 'deployment_eligible': False})
    (output / 'ready').touch()
    deadline = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.05)


def execute(path, output):
    check(path)
    modules = {'scripts.run_audio_separation_v14', 'scripts.run_video_demamba_v13', 'scripts.repeat_video_synthesis_v13',
               'scripts.repeat_video_release_v14', 'scripts.verify_video_native_v14'}
    for p in psutil.process_iter(['pid', 'cmdline']):
        command = p.info['cmdline'] or []
        if p.pid != os.getpid() and '-m' in command and any(m in command for m in modules):
            raise ValueError('Run heavy evaluations sequentially')
    release.prior.guard.profile(path, output, worker_module='scripts.verify_video_native_v14', limits=POLICY['limits'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'public', 'worker', 'execute'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'public': public, 'worker': worker, 'execute': execute}[args.mode](args.input, args.output)
