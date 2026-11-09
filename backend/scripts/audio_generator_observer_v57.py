"""Fixed offline CPU generator qualification; unchanged corrected terminal/resource observer."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample
from scripts.image_regional_continuation_runtime_v41 import verified_terminal_transition


def run(command, arguments):
    import os

    from scripts.audio_generator_qualification_v57 import RUNTIME, TRIAL

    if command == "prepare" and not arguments:
        folder = TRIAL.parent / (TRIAL.name + "-preparation-profile")
        output = TRIAL
    elif command == "smoke" and not arguments:
        folder, output = TRIAL / "smoke-profile", TRIAL / "smoke"
    else:
        raise ValueError("One explicit generator qualification stage required")
    os.environ["VERITAS_DEVICE"] = "cpu"
    os.environ["VERITAS_MODEL_MEMORY_BUDGET_MB"] = "4096"
    module = "scripts.audio_generator_qualification_v57"
    worker_arguments = [command, *arguments]
    checkpoint_root = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
    if output.exists():
        raise FileExistsError("Existing or failed calibration output must never be overwritten")
    folder.mkdir(parents=True, exist_ok=False)
    disk_path = checkpoint_root
    while not disk_path.exists():
        disk_path = disk_path.parent
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"}}
    environment.update(PYTHONPATH=str(Path.cwd()), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPYCACHEPREFIX=str(TRIAL.parent / "generator-qualification-empty-python-cache-v57"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
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
                    [str(RUNTIME / "bin/python"), "-B", "-m", module, *worker_arguments], stdout=log, stderr=subprocess.STDOUT, env=environment
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
        "schema": "chipmunk.regional-calibration-fixed-monitor-profile/57",
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
        "scope": "One explicit offline CPU generator qualification on authored prompts only and corrected terminal observation. No retry, fallback or overwrite. Missing live workers and all resource violations remain failures. RSS is sampled, not a true peak guarantee.",
    }
    with (folder / "profile.json").open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result), flush=True)
    if guard is not None or errors or status != 0:
        raise RuntimeError("Failed/guarded calibration retained; no automatic retry")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    run(args.command, args.arguments)
