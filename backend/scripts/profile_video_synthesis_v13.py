"""Bounded post-hoc resource diagnosis, never a quality/admission benchmark.

Repeat the two inputs immediately preceding the interrupted 4K input, then that
input, through the unchanged original scorer. A fresh child prevents accidental
reuse of the failed journal. Parent guards terminate only this owned worker.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil

LIMITS = {'parent_rss_bytes': int(4.5 * 1024**3), 'tree_rss_bytes': 6 * 1024**3,
          'min_system_available_bytes': int(1.5 * 1024**3), 'min_free_disk_bytes': 1024**3,
          'timeout_s': 180, 'interval_s': .1, 'source_indices': [82, 83, 84]}


def write(path, body):
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')


def worker(lock_path, output):
    import tempfile

    import torch

    from app.bench.score import ScoreTask
    from app.engine import pipelines
    from scripts import run_video_demamba_v13 as original

    body = original.check(lock_path)
    torch.set_num_threads(2)
    process = psutil.Process()
    mutex = threading.Lock()

    def record(stage, phase, **fields):
        row = {'stage': stage, 'phase': phase, 'monotonic_s': time.monotonic(),
               'rss_bytes': process.memory_info().rss, 'system_available_bytes': psutil.virtual_memory().available,
               'mps_current_bytes': torch.mps.current_allocated_memory(), 'mps_driver_bytes': torch.mps.driver_allocated_memory(), **fields}
        with mutex, (output / 'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def instrument(stage):
        def invoke(ctx):
            record(stage.id, 'begin', analysis_id=ctx.analysis_id)
            try:
                return stage.fn(ctx)
            finally:
                record(stage.id, 'end', analysis_id=ctx.analysis_id)
        return dataclasses.replace(stage, fn=invoke)

    pipelines.PIPELINES['video'] = tuple(instrument(s) for s in pipelines.PIPELINES['video'])
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix='video-v13-profile-') as scratch:
            scorer = original.baseline_scorer(body, scratch)
            try:
                for index in LIMITS['source_indices']:
                    source = body['rows'][index]
                    path = Path(body['base']) / source['relative_path']
                    if original.digest(path) != source['sha256']:
                        raise ValueError('Original diagnostic source changed')
                    result = scorer.score(ScoreTask(source['id'], source['variant'], 'none', path))
                    checks = {k: result[scorer.keys[k]] for k in original.CHECKS}
                    rows.append({'index': index, 'id': source['id'], 'variant': source['variant'], 'sha256': source['sha256'], 'checks': checks})
                    write(output / f'case-{index}.json', rows[-1])
            finally:
                scorer.close()
        write(output / 'child-result.json', {'completed': True, 'cases': rows, 'quality_comparison': False})
    except Exception as error:
        traceback.print_exc()
        write(output / 'child-result.json', {'completed': False, 'error_type': type(error).__name__, 'cases': rows, 'quality_comparison': False})
    # Keep the root alive until the parent sampler's final observation.
    (output / 'ready').touch()
    deadline = time.monotonic() + 30
    while not (output / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.05)


def profile(lock_path, output):
    from app.bench.process_memory import ProcessMemorySampler

    if output.exists():
        raise ValueError('New diagnostic directory required; never restart existing work')
    if psutil.disk_usage(output.parent).free < 2 * 1024**3 or psutil.virtual_memory().available < 6 * 1024**3:
        raise ValueError('Insufficient initial resource headroom for diagnosis')
    output.mkdir(parents=True)
    write(output / 'limits.json', LIMITS)
    started = time.monotonic()
    with (output / 'worker.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.profile_video_synthesis_v13', 'worker', str(lock_path), str(output)], stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        created = process.create_time()
        guard = None
        with ProcessMemorySampler([child.pid], interval_s=.1) as sampler:
            while child.poll() is None and not (output / 'ready').exists():
                try:
                    if process.create_time() != created:
                        raise ProcessLookupError('Owned PID changed')
                    parent_rss = process.memory_info().rss
                    tree_rss = parent_rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    readings = {'parent_rss_bytes': parent_rss, 'tree_rss_bytes': tree_rss,
                                'system_available_bytes': psutil.virtual_memory().available, 'free_disk_bytes': psutil.disk_usage(output).free,
                                'elapsed_s': time.monotonic() - started}
                    reasons = [name for name in ('parent_rss_bytes', 'tree_rss_bytes') if readings[name] > LIMITS[name]]
                    reasons += ['system_available_bytes'] if readings['system_available_bytes'] < LIMITS['min_system_available_bytes'] else []
                    reasons += ['free_disk_bytes'] if readings['free_disk_bytes'] < LIMITS['min_free_disk_bytes'] else []
                    reasons += ['timeout'] if readings['elapsed_s'] > LIMITS['timeout_s'] else []
                    if reasons:
                        guard = {'reasons': reasons, 'readings': readings}
                        break
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                except Exception as error:
                    guard = {'reasons': ['observation_failed'], 'error_type': type(error).__name__}
                    break
                time.sleep(.1)
        if guard and child.poll() is None:
            owned = process.children(recursive=True)
            child.terminate()
            for descendant in owned:
                with contextlib.suppress(psutil.NoSuchProcess):
                    descendant.terminate()
        (output / 'release').touch()
        try:
            exit_code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            exit_code = child.wait()
    result = {'schema': 'chipmunk.video-v13-resource-diagnostic/13', 'pid': child.pid, 'created': created,
              'profile_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              'limits': LIMITS, 'guard': guard, 'child_exit_code': exit_code, 'memory': sampler.report(),
              'child_result_available': (output / 'child-result.json').exists(), 'quality_comparison': False,
              'serving_changed': False, 'root_cause_of_previous_termination_isolated': False}
    write(output / 'profile.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('profile', 'worker'))
    parser.add_argument('lock', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'profile': profile, 'worker': worker}[args.stage](args.lock, args.output)
