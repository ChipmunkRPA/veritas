"""Sequential, bounded full-source fitting with explicit resource receipts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample

MODULE = "scripts.image_regional_training_v26"


def run(study, number):
    if type(number) is not int or number < 0:
        raise ValueError("Explicit nonnegative phase required")
    folder = study / f"phase-{number:03d}-profile"
    if (study / f"phase-{number:03d}").exists():
        raise FileExistsError("Previous/failed fitting output must never be overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint_root = Path(json.loads((study / "plan.json").read_text())["checkpoint_root"])
    started = time.monotonic()
    child, status, guard, last = None, None, None, None
    errors, transitions, observations = [], [], []
    arguments = ["train", str(study), str(number)]
    with (folder / "worker.log").open("x") as log, (folder / "resource-samples.jsonl").open("x") as journal:

        def record(rss, *, initial=False):
            nonlocal last
            last = limits.observe(folder, started, rss)
            last["checkpoint_disk_free_bytes"] = shutil.disk_usage(checkpoint_root).free
            violations = limits.violations(last, initial=initial)
            if last["checkpoint_disk_free_bytes"] < (limits.START_DISK if initial else limits.RESERVE):
                violations.append("checkpoint_disk_reserve")
            row = {**last, "stage": "initial" if initial else "running", "violations": violations}
            observations.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            return violations

        try:
            if not checkpoint_root.is_dir():
                raise FileNotFoundError("Private checkpoint destination unavailable")
            if record(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen(
                    [sys.executable, "-m", MODULE, *arguments], stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                    if child.poll() is None:
                        raise
                    transitions.append(
                        {
                            "type": type(error).__name__,
                            "classification": "verified_exit_before_first_sample",
                            "child_exit_code": child.poll(),
                        }
                    )
                    process = None
                while process is not None and child.poll() is None:
                    try:
                        rss, event = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        errors.append(
                            {
                                "type": type(error).__name__,
                                "classification": "unresolved_live_sampling_failure",
                            }
                        )
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
                    child.terminate()
                try:
                    status = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status = child.wait()
                try:
                    if record(0) and guard is None:
                        guard = "recorded_final_resource_limit"
                except BaseException as error:
                    errors.append(
                        {"type": type(error).__name__, "classification": "final_observation_unavailable"}
                    )
                    guard = guard or "monitor_unavailable"
    with (folder / "resource-samples.jsonl").open("rb") as stream:
        journal_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "schema": "chipmunk.regional-training-resource-profile/26",
        "module": MODULE,
        "arguments": arguments,
        "phase": number,
        "child_started": child is not None,
        "child_exit_code": status,
        "guard": guard,
        "sample_errors": len(errors),
        "sampling_error_events": errors,
        "terminal_observations": transitions,
        "resource_limit_observations": [r for r in observations if r["violations"]],
        "last_observation": last,
        "sampled_tree_rss_lower_bound_bytes": max((r["tree_rss_bytes"] for r in observations), default=0),
        "minimum_observed_available_ram_bytes": min(
            (r["ram_available_bytes"] for r in observations), default=None
        ),
        "minimum_observed_free_disk_bytes": min((r["disk_free_bytes"] for r in observations), default=None),
        "resource_journal_sha256": journal_sha,
        "duration_s": time.monotonic() - started,
        "limits": {
            "seconds": limits.SECONDS,
            "sampled_tree_rss_bytes": limits.RSS_LIMIT,
            "ram_and_disk_reserve_bytes": limits.RESERVE,
            "initial_ram_headroom_bytes": limits.START_RAM,
            "initial_disk_headroom_bytes": limits.START_DISK,
        },
        "scope": "One explicit phase; no restart. All observed stopping values and failures retained. RSS is sampled, not a true peak guarantee.",
    }
    with (folder / "profile.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if guard is not None or errors or status != 0:
        raise RuntimeError("Failed/guarded fitting phase retained; explicit successor required")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("number", type=int)
    args = parser.parse_args()
    run(args.study, args.number)
