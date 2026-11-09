"""Bound one explicit model-free source metadata phase, retaining failures."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample
from scripts.image_source_rights_v19 import sha, write


def run(command, folder, number=None):
    if command not in {"plan", "acquire", "summarize"}:
        raise ValueError("Explicit model-free metadata command required")
    arguments = [str(folder)] + ([str(number)] if command == "acquire" else [])
    profile = folder.parent / (folder.name + "-plan-profile") if command == "plan" else folder / (f"phase-{number:03d}-profile" if command == "acquire" else "summary-profile")
    profile.mkdir(parents=True, exist_ok=False)
    started, child, code, guard = time.monotonic(), None, None, None
    observations, errors, terminal = [], [], []
    with (profile / "worker.log").open("x") as log, (profile / "resource-samples.jsonl").open("x") as journal:
        def observe(rss, initial=False):
            row = limits.observe(profile, started, rss)
            row["violations"] = limits.violations(row, initial=initial)
            observations.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if observe(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen([sys.executable, "-m", "scripts.image_fresh_attribution_v37", command, *arguments], stdout=log, stderr=subprocess.STDOUT)
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
        "schema": "chipmunk.fresh-attribution-resource-profile/37", "command": command, "arguments": arguments,
        "child_exit_code": code, "guard": guard, "sample_errors": len(errors), "errors": errors,
        "terminal_observations": terminal, "duration_s": time.monotonic() - started,
        "sampled_tree_rss_lower_bound_bytes": max(row["tree_rss_bytes"] for row in observations),
        "minimum_observed_available_ram_bytes": min(row["ram_available_bytes"] for row in observations),
        "minimum_observed_free_disk_bytes": min(row["disk_free_bytes"] for row in observations),
        "resource_limit_observations": [row for row in observations if row["violations"]],
        "resource_journal_sha256": sha(profile / "resource-samples.jsonl"),
        "limits": {"seconds": limits.SECONDS, "sampled_tree_rss_bytes": limits.RSS_LIMIT, "ram_disk_reserve_bytes": limits.RESERVE, "initial_ram_headroom_bytes": limits.START_RAM, "initial_disk_headroom_bytes": limits.START_DISK},
        "scope": "One metadata-only phase, never neural inference or media download. No restart/overwrite. All failures retained. RSS is sampled, not a guaranteed peak.",
    }
    write(profile / "profile.json", result)
    print(json.dumps(result), flush=True)
    if code != 0 or guard or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "acquire", "summarize"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("phase", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "acquire" and (type(args.phase) is not int or not 0 <= args.phase < 8):
        parser.error("acquire requires one exact phase0..7")
    run(args.command, args.folder.resolve(), args.phase)
