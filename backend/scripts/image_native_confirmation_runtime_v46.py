"""Fixed confirmation dispatch with the unchanged corrected resource-observer mathematics."""

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
from scripts.image_regional_continuation_runtime_v41 import verified_terminal_transition


def run(command, arguments):
    if command == "prepare" and len(arguments) == 3:
        parent, trial, checkpoint_root = (Path(v).resolve() for v in arguments)
        module = "scripts.image_native_confirmation_v46"
        worker_arguments = ["prepare", str(parent), str(trial), str(checkpoint_root)]
        folder = trial.parent / (trial.name + "-preparation-profile")
        output = trial
    elif (
        command == "score"
        and len(arguments) == 3
        and arguments[1] in {"candidate", "baseline"}
        and arguments[2] in {str(n) for n in range(10)}
    ):
        from scripts.image_native_confirmation_v46 import lease

        trial = Path(arguments[0]).resolve()
        lease(trial)
        number = int(arguments[2])
        name = f"{arguments[1]}-native-{number:03d}"
        module = "scripts.image_native_confirmation_v46"
        worker_arguments = ["score", str(trial), arguments[1], "native", str(number)]
        folder, output = trial / (name + "-profile"), trial / name
        checkpoint_root = Path(json.loads((trial / "private-paths.json").read_text())["raw_map_root"])
    elif command == "phase" and len(arguments) == 2 and arguments[1] in {str(n) for n in range(10)}:
        trial = Path(arguments[0]).resolve()
        number = int(arguments[1])
        module = "scripts.image_native_confirmation_readout_v46"
        worker_arguments = ["phase", str(trial), str(number)]
        folder, output = trial / f"readout-{number:03d}-profile", trial / f"readout-{number:03d}"
        checkpoint_root = Path(json.loads((trial / "private-paths.json").read_text())["raw_map_root"])
    elif command == "seal" and len(arguments) == 1:
        trial = Path(arguments[0]).resolve()
        module = "scripts.image_native_confirmation_readout_v46"
        worker_arguments = ["seal", str(trial)]
        folder, output = trial / "comparison-seal-profile", trial / "comparison-summary.json"
        checkpoint_root = Path(json.loads((trial / "private-paths.json").read_text())["raw_map_root"])
    else:
        raise ValueError("One explicit complete native comparison phase required")
    if output.exists():
        raise FileExistsError("Existing or failed calibration output must never be overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    disk_path = checkpoint_root
    while not disk_path.exists():
        disk_path = disk_path.parent
    started, child, status, guard = time.monotonic(), None, None, None
    observations, errors, transitions = [], [], []
    with (folder / "worker.log").open("x") as log, (folder / "resource-samples.jsonl").open("x") as journal:

        def record(rss, initial=False):
            row = limits.observe(folder, started, rss)
            row["checkpoint_disk_free_bytes"] = shutil.disk_usage(disk_path).free
            row["violations"] = limits.violations(row, initial=initial)
            if not Path("/Volumes/SEAGATE").is_mount():
                row["violations"].append("checkpoint_volume_unavailable")
            if row["checkpoint_disk_free_bytes"] < (limits.START_DISK if initial else limits.RESERVE):
                row["violations"].append("checkpoint_disk_reserve")
            observations.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if record(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen(
                    [sys.executable, "-m", module, *worker_arguments], stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                    transitions.append(verified_terminal_transition(child, error))
                    process = None
                while process is not None and child.poll() is None:
                    try:
                        rss, event = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        try:
                            transitions.append(verified_terminal_transition(child, error))
                        except subprocess.TimeoutExpired:
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
    result = {
        "schema": "chipmunk.regional-calibration-fixed-monitor-profile/46",
        "module": module,
        "arguments": worker_arguments,
        "child_started": child is not None,
        "child_exit_code": status,
        "guard": guard,
        "sample_errors": len(errors),
        "sampling_error_events": errors,
        "terminal_observations": transitions,
        "resource_limit_observations": [row for row in observations if row["violations"]],
        "last_observation": observations[-1] if observations else None,
        "sampled_tree_rss_lower_bound_bytes": max((row["tree_rss_bytes"] for row in observations), default=0),
        "minimum_observed_available_ram_bytes": min(
            (row["ram_available_bytes"] for row in observations), default=None
        ),
        "minimum_observed_free_disk_bytes": min(
            (row["disk_free_bytes"] for row in observations), default=None
        ),
        "resource_journal_sha256": hashlib.sha256(
            (folder / "resource-samples.jsonl").read_bytes()
        ).hexdigest(),
        "duration_s": time.monotonic() - started,
        "limits": {
            "seconds": limits.SECONDS,
            "sampled_tree_rss_bytes": limits.RSS_LIMIT,
            "ram_and_disk_reserve_bytes": limits.RESERVE,
            "initial_ram_headroom_bytes": limits.START_RAM,
            "initial_disk_headroom_bytes": limits.START_DISK,
        },
        "monitor_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "One explicit confirmation phase with methods fixed before source inference and corrected terminal observation. No retry, fallback or overwrite. Missing live workers and all resource violations remain failures. RSS is sampled, not a true peak guarantee.",
    }
    if command in {"score", "phase"}:
        result["phase"] = number
    with (folder / "profile.json").open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result), flush=True)
    if guard is not None or errors or status != 0:
        raise RuntimeError("Failed/guarded calibration retained; no automatic retry")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score", "phase", "seal"))
    parser.add_argument("arguments", nargs="+")
    args = parser.parse_args()
    run(args.command, args.arguments)
