"""Record exact stopping observations; preflight never starts a model worker."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample

MODULE = "scripts.relayformer_checkpointed_probe_v24"


def run(arm, folder, worker_folder, *, preflight_only=False):
    if arm not in {"baseline", "checkpointed"}:
        raise ValueError("Explicit authored comparison arm required")
    folder.mkdir(parents=True, exist_ok=False)
    if worker_folder.exists():
        raise FileExistsError("A new worker folder is required; failed results are never overwritten")
    started = time.monotonic()
    child, status, guard, last = None, None, None, None
    errors, transitions, observations = [], [], []
    with (folder / "worker.log").open("x") as log, (folder / "resource-samples.jsonl").open("x") as journal:
        def record(rss, *, initial=False):
            nonlocal last
            last = limits.observe(folder, started, rss)
            row = {**last, "stage": "initial" if initial else "running", "violations": limits.violations(last, initial=initial)}
            observations.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if record(0, initial=True):
                guard = "initial_headroom_unavailable"
            elif not preflight_only:
                child = subprocess.Popen([sys.executable, "-m", MODULE, arm, str(worker_folder)], stdout=log, stderr=subprocess.STDOUT)
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                    if child.poll() is None:
                        raise
                    transitions.append({"type": type(error).__name__, "classification": "verified_exit_before_first_sample", "child_exit_code": child.poll()})
                    process = None
                while process is not None and child.poll() is None:
                    try:
                        rss, event = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        errors.append({"type": type(error).__name__, "classification": "unresolved_live_sampling_failure"})
                        guard = "live_sampling_unavailable"
                        break
                    if record(rss):
                        guard = "recorded_resource_limit"
                        break
                    if event:
                        transitions.append(event)
                        break
                    time.sleep(0.25)
        except BaseException as error:
            errors.append({"type": type(error).__name__, "classification": "monitor_exception"})
            guard = "monitor_unavailable"
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()  # The explicit worker creates no child processes.
                try:
                    status = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status = child.wait()
                try:
                    if record(0) and guard is None:
                        guard = "recorded_final_resource_limit"
                except BaseException as error:
                    errors.append({"type": type(error).__name__, "classification": "final_observation_unavailable"})
                    guard = guard or "monitor_unavailable"
    journal_path = folder / "resource-samples.jsonl"
    with journal_path.open("rb") as stream:
        journal_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "schema": "chipmunk.authored-comparison-resource-profile/24", "arm": arm, "module": MODULE,
        "arguments": [arm, str(worker_folder)], "preflight_only": preflight_only, "child_started": child is not None, "child_exit_code": status,
        "eligible_at_initial_observation": bool(observations) and not observations[0]["violations"],
        "guard": guard, "sample_errors": len(errors), "sampling_error_events": errors, "terminal_observations": transitions,
        "resource_limit_observations": [r for r in observations if r["violations"]], "last_observation": last,
        "sampled_tree_rss_lower_bound_bytes": max((r["tree_rss_bytes"] for r in observations), default=0),
        "minimum_observed_available_ram_bytes": min((r["ram_available_bytes"] for r in observations), default=None),
        "minimum_observed_free_disk_bytes": min((r["disk_free_bytes"] for r in observations), default=None),
        "resource_journal_sha256": journal_sha, "duration_s": time.monotonic() - started,
        "limits": {"seconds": limits.SECONDS, "sampled_tree_rss_bytes": limits.RSS_LIMIT, "ram_and_disk_reserve_bytes": limits.RESERVE, "initial_ram_headroom_bytes": limits.START_RAM, "initial_disk_headroom_bytes": limits.START_DISK},
        "scope": "An observation-only preflight is not a completed model test. Every observed limit and stopping measurement is retained; no implicit retry. RSS is a sampled lower bound, not a peak guarantee.",
    }
    with (folder / "profile.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if guard is not None or errors or (not preflight_only and status != 0):
        raise RuntimeError("Stopped/failed authored phase retained; no automatic retry")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("baseline", "checkpointed"))
    parser.add_argument("profile_folder", type=Path)
    parser.add_argument("worker_folder", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.arm, args.profile_folder, args.worker_folder, preflight_only=args.preflight_only)
