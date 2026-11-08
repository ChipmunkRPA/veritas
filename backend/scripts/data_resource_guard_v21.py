"""Bounded data-worker monitor with explicit terminal observation events."""

import json
import shutil
import subprocess
import sys
import time

import psutil


def sample(root, child):
    """A vanished worker is terminal only after authoritative waitpid polling."""
    try:
        tree = [root, *root.children(recursive=True)]
        rss = sum(p.memory_info().rss for p in tree)
        return rss, None
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
        code = child.poll()
        if code is not None:
            return 0, {"type": type(error).__name__, "child_exit_code": code, "classification": "verified_terminal_transition"}
        raise


def run(module, arguments, folder):
    if module != "scripts.image_positive_cached_v21":
        raise ValueError("Explicit data-only worker required")
    folder.mkdir(parents=True, exist_ok=False)
    child, status, guard, maximum, errors, terminal_events = None, None, None, 0, [], []
    start = time.monotonic()
    with (folder / "worker.log").open("x") as log:
        try:
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
                if rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2 or shutil.disk_usage(folder).free < 1536 * 1024**2 or time.monotonic() - start > 900:
                    guard = "memory_disk_or_time"
                    break
                time.sleep(0.25)
        except BaseException as error:
            guard = "monitor_" + type(error).__name__
            errors.append({"elapsed_s": time.monotonic() - start, "type": type(error).__name__, "classification": "monitor_exception"})
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        members = psutil.Process(child.pid).children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        members = []
                    for member in reversed(members):
                        try:
                            member.terminate()
                        except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                            terminal_events.append({"elapsed_s": time.monotonic() - start, "type": type(error).__name__, "pid": member.pid, "classification": "descendant_exited_before_cleanup"})
                    child.terminate()
                try:
                    status = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status = child.wait()
    receipt = {
        "schema": "chipmunk.data-worker-resource-profile/21", "module": module, "arguments": arguments,
        "child_exit_code": status, "guard": guard, "sampled_tree_rss_lower_bound_bytes": maximum,
        "sample_errors": len(errors), "sampling_error_events": errors,
        "terminal_observation_events": terminal_events, "duration_s": time.monotonic() - start, "seconds_limit": 900,
        "scope": "Verified terminal transitions are retained separately from inability to sample a live process. RSS is sampled, not a true peak guarantee.",
    }
    with (folder / "profile.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if status != 0 or guard is not None or errors:
        raise RuntimeError("Failed/guarded data phase retained; no automatic retry")
