"""Image-only guarded runner; existing frozen audio/video runners remain unchanged."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

MODULES = {'scripts.run_image_supported_v11'}


def run(module, arguments, folder, seconds):
    if module not in MODULES or not 0 < seconds <= 900:
        raise ValueError('Explicit supported specialist module and bounded duration required')
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', module, *arguments]
    started, maximum, guard, sample_errors = time.monotonic(), 0, None, 0
    child, tree, status = None, [], None
    with (folder / 'worker.log').open('x') as log:
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            while child.poll() is None:
                tree, rss = [process], 0
                try:
                    tree += process.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in tree)
                    maximum = max(maximum, rss)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    sample_errors += 1
                if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                        or shutil.disk_usage(folder).free < 1536 * 1024**2
                        or time.monotonic() - started > seconds):
                    guard = 'memory_disk_or_time'
                    break
                time.sleep(.25)
        except BaseException as error:
            guard = 'monitor_' + type(error).__name__
        finally:
            if child is not None:
                if child.poll() is None:
                    for member in reversed(tree):
                        try:
                            member.terminate()
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            sample_errors += 1
                    if not tree:
                        child.terminate()
                try:
                    status = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status = child.wait()
    receipt = {'schema': 'chipmunk.specialist-local-resource-profile/5', 'module': module,
        'arguments': arguments, 'child_exit_code': status, 'guard': guard,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'sample_errors': sample_errors,
        'duration_s': time.monotonic() - started, 'seconds_limit': seconds}
    with (folder / 'profile.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)
    if status != 0 or guard is not None:
        raise RuntimeError('Failed or guarded job retained; no implicit restart or omitted case')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', choices=sorted(MODULES))
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--validate-only', action='store_true', help='Validate the real queued command without creating output or starting a worker')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 900:
        parser.error('Duration must be positive and at most900seconds')
    if args.validate_only:
        print(json.dumps({'schema': 'chipmunk.specialist-runtime-command-preflight/5',
            'module': args.module, 'arguments': args.arguments, 'seconds_limit': args.seconds,
            'supported': True, 'child_started': False, 'output_created': False}))
    else:
        run(args.module, args.arguments, args.folder, args.seconds)
