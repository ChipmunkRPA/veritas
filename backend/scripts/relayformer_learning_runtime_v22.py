"""Bounded explicit research workers; acquisition/review monitors stay frozen."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts.data_resource_guard_v21 import sample

MODULES = {"scripts.relayformer_learning_v22", "scripts.image_source_groups_v22"}
RESERVE = 1536 * 1024**2


def run(module, arguments, folder):
    if module not in MODULES:
        raise ValueError("Explicit supported local research module required")
    folder.mkdir(parents=True, exist_ok=False)
    child, status, guard, maximum, errors, terminal_events = None, None, None, 0, [], []
    start = time.monotonic()
    with (folder / "worker.log").open("x") as log:
        try:
            if psutil.virtual_memory().available < RESERVE or shutil.disk_usage(folder).free < RESERVE:
                guard = "initial_reserve_unavailable"
            else:
                child = subprocess.Popen([sys.executable, "-m", module, *arguments], stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        rss, transition = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        errors.append({"elapsed_s": time.monotonic() - start, "type": type(error).__name__, "classification": "unresolved_live_sampling_failure"})
                        guard = "live_sampling_unavailable"
                        break
                    maximum = max(maximum, rss)
                    if transition:
                        terminal_events.append({"elapsed_s": time.monotonic() - start, **transition})
                        break
                    if rss > 6 * 1024**3 or psutil.virtual_memory().available < RESERVE or shutil.disk_usage(folder).free < RESERVE or time.monotonic() - start > 900:
                        guard = "memory_disk_or_time"
                        break
                    time.sleep(0.25)
        except BaseException as error:
            guard = "monitor_" + type(error).__name__
            errors.append({"elapsed_s": time.monotonic() - start, "type": type(error).__name__, "classification": "monitor_exception"})
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()  # This single-process probe creates no children.
                try:
                    status = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status = child.wait()
    receipt = {"schema": "chipmunk.regional-learning-resource-profile/22", "module": module, "arguments": arguments, "child_exit_code": status, "guard": guard, "sampled_tree_rss_lower_bound_bytes": maximum, "sample_errors": len(errors), "sampling_error_events": errors, "terminal_observation_events": terminal_events, "duration_s": time.monotonic() - start, "seconds_limit": 900, "scope": "Explicit local research worker; neural/source execution counts come from its own actual receipt. Same900s/6GiB and1536MiB reserves. Verified terminal transitions retained; missed live samples fail. RSS is a sampled lower bound, not a peak guarantee."}
    with (folder / "profile.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if status != 0 or guard is not None or errors:
        raise RuntimeError("Failed/guarded research phase retained; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=sorted(MODULES))
    parser.add_argument("profile_folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run(args.module, args.arguments, args.profile_folder)
