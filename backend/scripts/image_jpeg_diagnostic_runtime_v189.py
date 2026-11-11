"""Run one bounded v189 image diagnostic phase in an isolated local process."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample
from scripts.image_jpeg_diagnostic_v189 import prep_profile, read, require, sha, write
from scripts.image_regional_continuation_runtime_v41 import verified_terminal_transition

MODULE = "scripts.image_jpeg_diagnostic_v189"
LIMITS = {
    "seconds": limits.SECONDS,
    "rss": limits.RSS_LIMIT,
    "reserve": limits.RESERVE,
    "start_ram": limits.START_RAM,
    "start_disk": limits.START_DISK,
}


def assert_profile(folder, command, arguments):
    profile = read(folder / "profile.json")
    require(
        profile["command"] == command
        and profile["worker_arguments"] == arguments
        and profile["worker"] == MODULE
        and profile["limits"] == LIMITS
        and profile["monitor_sha256"] == sha(Path(__file__))
        and profile["resource_sha256"] == sha(folder / "resources.jsonl")
        and profile["exit_code"] == 0
        and profile["failure"] is None
        and 0 < profile["duration_seconds"] <= limits.SECONDS,
        "Complete successful bounded phase required",
    )
    return profile


def phase_paths(command, arguments):
    trial = Path(arguments[0]).resolve()
    if command == "prepare":
        raw_root = Path(arguments[1]).resolve()
        return prep_profile(trial), trial, raw_root, [command, str(trial), str(raw_root)]
    raw_root = Path(read(trial / "inputs.json")["raw_root"])
    worker_args = [command, str(trial), *arguments[1:]]
    if command == "seal":
        return trial / "seal-profile", trial / "summary.json", raw_root, worker_args
    quality, phase = map(int, arguments[-2:])
    label = arguments[2] if command == "score" else "readout"
    name = f"{label}-q{quality}-{phase:02d}"
    return trial / f"{name}-profile", trial / name, raw_root, worker_args


def run(command, arguments):
    folder, output, raw_root, worker_args = phase_paths(command, arguments)
    require(not output.exists() and raw_root.is_dir(), "Use a new phase and existing private raw-map root")
    folder.mkdir()
    started = time.monotonic()
    child = status = failure = None
    maximum_rss = 0
    transitions = []
    resources = folder / "resources.jsonl"
    with resources.open("x") as journal, (folder / "worker.log").open("x") as log:

        def record(rss, *, initial=False):
            row = limits.observe(folder, started, rss)
            row["raw_disk_free_bytes"] = shutil.disk_usage(raw_root).free
            problems = limits.violations(row, initial=initial)
            if row["raw_disk_free_bytes"] < (limits.START_DISK if initial else limits.RESERVE):
                problems.append("raw_disk_reserve")
            journal.write(json.dumps({**row, "violations": problems}, allow_nan=False) + "\n")
            journal.flush()
            return problems

        try:
            failure = record(0, initial=True) or None
            if failure is None:
                child = subprocess.Popen(
                    [sys.executable, "-m", MODULE, *worker_args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    process = psutil.Process(child.pid)
                    while child.poll() is None:
                        rss, transition = sample(process, child)
                        maximum_rss = max(maximum_rss, rss)
                        failure = record(rss) or None
                        if transition:
                            transitions.append(transition)
                        if transition or failure:
                            break
                        time.sleep(0.25)
                except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                    transitions.append(verified_terminal_transition(child, error))
        except Exception as error:
            failure = {"observer_error": type(error).__name__}
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
                    final_violations = record(0)
                    failure = failure or final_violations or None
                except Exception as error:
                    failure = failure or {"final_observer_error": type(error).__name__}
    write(
        folder / "profile.json",
        {
            "schema": "chipmunk.image-jpeg-diagnostic-profile/189",
            "command": command,
            "worker": MODULE,
            "worker_arguments": worker_args,
            "exit_code": status,
            "failure": failure,
            "duration_seconds": time.monotonic() - started,
            "limits": LIMITS,
            "monitor_sha256": sha(Path(__file__)),
            "resource_sha256": sha(resources),
            "sampled_tree_rss_lower_bound_bytes": maximum_rss,
            "terminal_observations": transitions,
        },
    )
    assert_profile(folder, command, worker_args)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    arity = {"prepare": 2, "score": 5, "readout": 4, "seal": 2}
    require(command in arity and len(args) == arity[command], "Explicit v189 diagnostic command required")
    run(command, args)
