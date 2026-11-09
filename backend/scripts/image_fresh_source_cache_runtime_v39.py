"""Bound one explicit image-reference cache phase; preserve every failed attempt."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample
from scripts.image_fresh_source_cache_v39 import VOLUME
from scripts.image_source_rights_v19 import sha, write


def run(command, arguments):
    if command == "prepare" and len(arguments) == 3:
        trial = Path(arguments[1])
        output = trial.parent / (trial.name + "-prepare-profile")
    elif command == "fetch" and len(arguments) == 2 and arguments[1] in {str(n) for n in range(6)}:
        trial = Path(arguments[0])
        output = trial / f"fetch-{int(arguments[1]):03d}-profile"
    elif command == "seal" and len(arguments) == 1:
        trial = Path(arguments[0])
        output = trial / "seal-profile"
    else:
        raise ValueError("One explicit complete source-cache command required")
    output.mkdir(parents=True, exist_ok=False)
    started, child, code, guard = time.monotonic(), None, None, None
    observations, errors, terminal = [], [], []
    with (output / "worker.log").open("x") as log, (output / "resource-samples.jsonl").open("x") as journal:
        def observe(rss, initial=False):
            row = limits.observe(output, started, rss)
            row["violations"] = limits.violations(row, initial=initial)
            mounted = VOLUME.is_mount()
            row["external_volume_mounted"] = mounted
            row["external_free_bytes"] = shutil.disk_usage(VOLUME).free if mounted else 0
            if not mounted:
                row["violations"].append("external_volume_unavailable")
            if row["external_free_bytes"] < (limits.START_DISK if initial else limits.RESERVE):
                row["violations"].append("external_disk_reserve")
            observations.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if observe(0, initial=True):
                guard = "initial_headroom_or_external_volume_unavailable"
            else:
                child = subprocess.Popen([sys.executable, "-m", "scripts.image_fresh_source_cache_v39", command, *arguments], stdout=log, stderr=subprocess.STDOUT)
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    code = child.wait(timeout=0.25)
                    terminal.append({"classification": "verified_terminal_before_first_sample", "exit_code": code})
                    process = None
                while process is not None and child.poll() is None:
                    try:
                        rss, event = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        code = child.wait(timeout=0.25)
                        terminal.append({"classification": "authoritative_terminal_after_missing_process", "exit_code": code, "error_type": type(error).__name__})
                        break
                    if observe(rss):
                        guard = "recorded_resource_limit"
                        break
                    if event:
                        terminal.append(event)
                        break
                    time.sleep(0.25)
        except BaseException as error:
            errors.append({"type": type(error).__name__, "detail": str(error)})
            guard = "monitor_or_unresolved_sampling_error"
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
                if observe(0) and guard is None:
                    guard = "final_resource_limit"
    result = {
        "schema": "chipmunk.fresh-image-cache-resource-profile/39", "command": command, "arguments": arguments,
        "child_exit_code": code, "guard": guard, "sample_errors": len(errors), "errors": errors,
        "terminal_observations": terminal, "duration_s": time.monotonic() - started,
        "sampled_tree_rss_lower_bound_bytes": max(row["tree_rss_bytes"] for row in observations),
        "minimum_observed_available_ram_bytes": min(row["ram_available_bytes"] for row in observations),
        "minimum_observed_free_disk_bytes": min(row["disk_free_bytes"] for row in observations),
        "minimum_observed_external_free_bytes": min(row["external_free_bytes"] for row in observations),
        "resource_limit_observations": [row for row in observations if row["violations"]],
        "resource_journal_sha256": sha(output / "resource-samples.jsonl"),
        "limits": {"seconds": limits.SECONDS, "sampled_tree_rss_bytes": limits.RSS_LIMIT, "ram_internal_external_disk_reserve_bytes": limits.RESERVE, "initial_ram_headroom_bytes": limits.START_RAM, "initial_disk_headroom_bytes": limits.START_DISK},
        "scope": "One explicit data-only cache phase. No model, generation, retry or overwrite; failed and partial cases retained. RSS is sampled, not a guaranteed peak.",
    }
    write(output / "profile.json", result)
    print(json.dumps(result), flush=True)
    if code != 0 or guard or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fetch", "seal"))
    parser.add_argument("arguments", nargs="+")
    args = parser.parse_args()
    run(args.command, args.arguments)
