"""Post-hoc memory-release diagnosis on consumed inputs; no quality admission.

Run unchanged synthesis checks, measure post-score retention, collect unreachable
Python objects, then release unused MPS allocator cache. Keep each step separate
and compare prior raw responses. Never edit or resume a failed study journal.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import psutil

from scripts import profile_video_synthesis_v13 as guard

INDICES = [0, 1, 40, 41, 42, 84]
LIMITS = {**guard.LIMITS, 'source_indices': INDICES, 'timeout_s': 300}


def worker(lock_path, output):
    import torch

    from app.bench.score import ScoreTask
    from scripts import run_video_demamba_v13 as original

    body = original.check(lock_path)
    torch.set_num_threads(2)
    previous = [json.loads(x) for x in (lock_path.parent / 'baseline.jsonl').read_text().splitlines()][1:]
    process = psutil.Process()

    def snapshot():
        return {'rss_bytes': process.memory_info().rss, 'mps_current_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory(), 'system_available_bytes': psutil.virtual_memory().available}

    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix='video-v14-release-') as scratch:
            scorer = original.baseline_scorer(body, scratch)
            try:
                for index in INDICES:
                    source = body['rows'][index]
                    path = Path(body['base']) / source['relative_path']
                    if original.digest(path) != source['sha256']:
                        raise ValueError('Original diagnostic input changed')
                    before = snapshot()
                    result = scorer.score(ScoreTask(source['id'], source['variant'], 'none', path))
                    checks = {k: result[scorer.keys[k]] for k in original.CHECKS}
                    after_score = snapshot()
                    collected = gc.collect()
                    after_gc = snapshot()
                    torch.mps.empty_cache()
                    after_empty = snapshot()
                    parity = None
                    if index < len(previous):
                        old = previous[index]
                        if (old['dataset'], old['id'], old['variant'], old['sha256']) != (source['dataset'], source['id'], source['variant'], source['sha256']):
                            raise ValueError('Prior source correspondence changed')
                        changes = []
                        for name, actual in checks.items():
                            expected = old['checks'][name]
                            if actual['status'] != expected['status'] or (actual['prob'] is None) != (expected['prob'] is None):
                                raise ValueError('Prior applicability differs')
                            if actual['prob'] is not None:
                                changes.append(abs(actual['prob'] - expected['prob']))
                        parity = max(changes, default=0.)
                        if parity > 1e-5:
                            raise ValueError('Prior raw response differs')
                    row = {'index': index, 'id': source['id'], 'variant': source['variant'], 'sha256': source['sha256'],
                           'checks': checks, 'before': before, 'after_score': after_score, 'unreachable_python_objects_collected': collected,
                           'after_gc': after_gc, 'after_empty_cache': after_empty, 'max_prior_raw_difference': parity}
                    cases.append(row)
                    guard.write(output / f'case-{index}.json', row)
            finally:
                scorer.close()
        guard.write(output / 'child-result.json', {'completed': True, 'cases': cases, 'quality_comparison': False,
                    'diagnostic_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'original_lock_sha256': original.digest(lock_path), 'serving_changed': False})
    except Exception as error:
        traceback.print_exc()
        guard.write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__, 'cases': cases, 'quality_comparison': False})
    (output / 'ready').touch()
    until = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < until:
        time.sleep(.05)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('profile', 'worker'))
    parser.add_argument('lock', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.stage == 'worker':
        worker(args.lock, args.output)
    else:
        if any(p.info['pid'] != os.getpid() and '-m' in (p.info['cmdline'] or [])
               and any(m in (p.info['cmdline'] or []) for m in ('scripts.repeat_video_synthesis_v13', 'scripts.run_audio_separation_v14'))
               for p in psutil.process_iter(['pid', 'cmdline'])):
            raise ValueError('Run heavy evaluations sequentially')
        guard.profile(args.lock, args.output, worker_module='scripts.diagnose_video_release_v14', limits=LIMITS)
