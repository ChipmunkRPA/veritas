"""Explicit monitor/authored speech stages; unchanged neural resource policy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress

import psutil

from scripts import local_probe_limits_v24 as limits
from scripts.audio_tree_sampler_v64 import sample_tree
from scripts.av_speech_native_v110 import ROOT, TRIAL, read, require, sha, verify, write
from scripts.video_appearance_native_v108 import ARCHIVE
from scripts.video_appearance_runtime_v108 import violations


def run(command):
    require(command in {"monitor", "authored"}, "One explicit qualification stage required")
    verify()
    folder = TRIAL / command
    require(not folder.exists(), "Preserve completed and partially attempted stages")
    if command == "authored":
        monitor = TRIAL / "monitor"
        if not all(p.is_file() for p in (monitor / "output/result.json", monitor / "profile.json")):
            print(json.dumps({"status": "pending_monitor_no_stage_attempt"}), flush=True)
            return
        result, profile = read(monitor / "output/result.json"), read(monitor / "profile.json")
        require(
            result["status"] == "complete"
            and result["owned_descendant_sampled"]
            and result["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py")
            and profile["child_exit_code"] == 0
            and profile["guard"] is None
            and profile["sample_errors"] == 0
            and profile["worker_receipt_sha256"] == sha(monitor / "output/result.json"),
            "Successful exact owned-process preflight required",
        )
    admission = limits.observe(TRIAL, time.monotonic(), 0)
    rejected = violations(admission, command, initial=True)
    if not ARCHIVE.parent.parent.is_mount() or shutil.disk_usage(ARCHIVE).free < limits.START_DISK:
        rejected.append("archive_headroom_unavailable")
    if rejected:
        print(
            json.dumps(
                {
                    "status": "pending_headroom_no_stage_attempt",
                    "command": command,
                    "reasons": rejected,
                    "available_ram_bytes": admission["ram_available_bytes"],
                }
            ),
            flush=True,
        )
        return
    folder.mkdir(exist_ok=False)
    env = {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"}
    }
    env.update(
        PYTHONPATH=str(ROOT / "backend"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    started, child, code, guard = time.monotonic(), None, None, None
    rows, errors, exits = [], [], []
    with (folder / "resources.jsonl").open("x") as journal, (folder / "worker.log").open("x") as log:

        def record(rss, initial=False):
            row = limits.observe(folder, started, rss)
            row["violations"] = violations(row, command, initial=initial)
            row["archive_free_disk_bytes"] = shutil.disk_usage(ARCHIVE).free
            if not ARCHIVE.parent.parent.is_mount() or row["archive_free_disk_bytes"] < (
                limits.START_DISK if initial else limits.RESERVE
            ):
                row["violations"].append("archive_headroom_unavailable")
            rows.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if record(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "scripts.av_speech_native_v110",
                        command,
                        str(folder / "output"),
                    ],
                    cwd=ROOT / "backend",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    code = child.wait(timeout=0.5)
                    process = None
                    exits.append({"classification": "verified_worker_terminal_transition", "exit_code": code})
                while process is not None and child.poll() is None:
                    try:
                        rss, events, terminal = sample_tree(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
                        code = child.wait(timeout=0.25)
                        exits.append(
                            {
                                "classification": "verified_worker_terminal_transition",
                                "exit_code": code,
                                "type": type(exc).__name__,
                            }
                        )
                        break
                    exits.extend(events)
                    if record(rss):
                        guard = "resource_limit"
                        break
                    if terminal:
                        break
                    time.sleep(0.25)
        except BaseException as exc:
            errors.append(type(exc).__name__)
            guard = "monitor_unavailable"
        finally:
            if child is not None:
                if child.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
                if guard is not None or code != 0:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
            try:
                if record(0) and guard is None:
                    guard = "final_resource_limit"
            except BaseException as exc:
                errors.append(type(exc).__name__)
                guard = guard or "monitor_unavailable"
    result_path = folder / "output/result.json"
    result = read(result_path) if result_path.is_file() else None
    if guard is None and code == 0 and (result is None or result.get("status") != "complete"):
        guard = "missing_completion_receipt"
    profile = {
        "schema": "chipmunk.av-speech-resource-profile/110",
        "command": command,
        "child_started": child is not None,
        "child_exit_code": code,
        "guard": guard,
        "sample_errors": len(errors),
        "error_types": errors,
        "terminal_observations": exits,
        "duration_s": time.monotonic() - started,
        "sampled_tree_rss_lower_bound_bytes": max((r["tree_rss_bytes"] for r in rows), default=0),
        "minimum_available_ram_bytes": min((r["ram_available_bytes"] for r in rows), default=None),
        "resource_limit_observations": [r for r in rows if r["violations"]],
        "resource_journal_sha256": sha(folder / "resources.jsonl"),
        "worker_receipt_sha256": sha(result_path) if result else None,
        "new_neural_calls": result["new_neural_calls"]
        if result
        else (0 if child is None or command == "monitor" else None),
        "scope": "Unchanged 900s/6-GiB sampled descendant RSS/6-GiB initial RAM/3-GiB initial disk/1.5-GiB running neural reserves. Monitor is data-only at 3-GiB initial RAM and 1.5-GiB RSS. No overwrite/retry; samples are not guaranteed peaks.",
    }
    write(folder / "profile.json", profile)
    print(json.dumps(profile), flush=True)
    if guard is not None or errors or code != 0:
        raise RuntimeError("Failed qualification retained without retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("monitor", "authored"))
    run(parser.parse_args().command)
