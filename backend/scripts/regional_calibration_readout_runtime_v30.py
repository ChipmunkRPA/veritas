"""Bounded model-free calibration map readout; no retries or model loading."""

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

MODULE = "scripts.replay_regional_calibration_v30"


def run(trial, command, fmt=None, number=None):
    if command == "phase":
        if fmt not in {"native", "jpeg75"} or type(number) is not int or number < 0:
            raise ValueError("Explicit readout format and phase required")
        name = f"readout-v30-{fmt}-{number:03d}"
        arguments = ["phase", str(trial), fmt, str(number)]
        output = trial / name
    elif command == "seal" and fmt is None and number is None:
        name = "calibration-v30-seal"
        arguments = ["seal", str(trial)]
        output = trial / "calibration-v30-summary.json"
    else:
        raise ValueError("Explicit phase or seal command required")
    folder = trial / (name + "-profile")
    if output.exists():
        raise FileExistsError("Previous/failed readout output must never be overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint_root = Path(json.loads((trial / "private-paths.json").read_text())["raw_map_root"])
    started = time.monotonic()
    child, status, guard, last = None, None, None, None
    errors, transitions, observations = [], [], []
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
        "schema": "chipmunk.regional-calibration-readout-resource-profile/30",
        "module": MODULE,
        "arguments": arguments,
        "stage": command,
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
        raise RuntimeError("Failed/guarded readout retained; explicit successor required")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    phase = commands.add_parser("phase")
    phase.add_argument("trial", type=Path)
    phase.add_argument("format", choices=("native", "jpeg75"))
    phase.add_argument("number", type=int)
    commands.add_parser("seal").add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.command == "phase":
        run(args.trial, args.command, args.format, args.number)
    else:
        run(args.trial, args.command)
