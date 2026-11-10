"""Explicit sequential data/neural stages with the unchanged neural limits."""

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
from pathlib import Path

import psutil

from scripts import local_probe_limits_v24 as original
from scripts.audio_tree_sampler_v64 import sample_tree
from scripts.video_appearance_native_v108 import (
    ARCHIVE,
    ROOT,
    TRIAL,
    read,
    ready,
    require,
    sha,
    verify,
    write,
)

BACKEND = ROOT / "backend"


def violations(row, command, initial=False):
    result = original.violations(row, initial=initial)
    if command in {"monitor", "bind"}:
        result = [p for p in result if p not in {"tree_rss_limit", "initial_ram_headroom"}]
        if row["tree_rss_bytes"] > 1536 * 1024**2:
            result.append("tree_rss_limit")
        if initial and row["ram_available_bytes"] < 3 * 1024**3:
            result.append("initial_ram_headroom")
    return result


def prerequisites(command, batch):
    if command == "monitor":
        return True
    if not (TRIAL / "monitor/output/result.json").is_file() or not (TRIAL / "monitor/profile.json").is_file():
        return False
    monitor = read(TRIAL / "monitor/output/result.json")
    profile = read(TRIAL / "monitor/profile.json")
    require(
        monitor["status"] == "complete"
        and monitor["owned_descendant_sampled"]
        and monitor["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "Successful unchanged owned-process preflight required",
    )
    require(
        profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
        "Successful monitor resource profile required",
    )
    if command == "bind":
        return ready(batch)
    if command == "extract":
        return all(
            p.is_file()
            for p in (
                TRIAL / "authored/output/result.json",
                TRIAL / "authored/profile.json",
                TRIAL / f"bind-{batch:02d}/output/result.json",
                TRIAL / f"bind-{batch:02d}/profile.json",
            )
        )
    return True


def run(command, batch=None):
    require(command in {"monitor", "bind", "authored", "extract"}, "One fixed stage required")
    if command in {"bind", "extract"}:
        require(type(batch) is int and 0 <= batch < 8, "One explicit prepared batch required")
        name = f"{command}-{batch:02d}"
    else:
        require(batch is None, "No batch for qualification stages")
        name = command
    verify()
    folder = TRIAL / name
    require(not folder.exists(), "Preserve every completed or partially attempted stage")
    if not prerequisites(command, batch):
        print(
            json.dumps(
                {"status": "pending_prerequisite_no_stage_attempt", "command": command, "batch": batch}
            ),
            flush=True,
        )
        return
    # Read-only admission does not create an attempted model stage when RAM is short.
    admission = original.observe(TRIAL, time.monotonic(), 0)
    rejected = violations(admission, command, initial=True)
    if not ARCHIVE.is_dir() or not ARCHIVE.parent.parent.is_mount():
        rejected.append("archive_volume_unavailable")
    elif shutil.disk_usage(ARCHIVE).free < original.START_DISK:
        rejected.append("archive_disk_reserve")
    if rejected:
        print(
            json.dumps(
                {
                    "status": "pending_headroom_no_stage_attempt",
                    "command": command,
                    "batch": batch,
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
        PYTHONPATH=str(BACKEND),
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
    started, child, guard, code = time.monotonic(), None, None, None
    rows, errors, exits = [], [], []
    with (folder / "resources.jsonl").open("x") as journal, (folder / "worker.log").open("x") as log:

        def record(rss, initial=False):
            row = original.observe(folder, started, rss)
            row["violations"] = violations(row, command, initial=initial)
            row["archive_free_disk_bytes"] = shutil.disk_usage(ARCHIVE).free
            if not ARCHIVE.parent.parent.is_mount():
                row["violations"].append("archive_volume_unavailable")
            if row["archive_free_disk_bytes"] < (original.START_DISK if initial else original.RESERVE):
                row["violations"].append("archive_disk_reserve")
            rows.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if record(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                argv = [
                    sys.executable,
                    "-B",
                    "-m",
                    "scripts.video_appearance_native_v108",
                    command,
                    str(folder / "output"),
                ]
                if batch is not None:
                    argv.append(str(batch))
                child = subprocess.Popen(
                    argv, cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
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
    receipt_path = folder / "output/result.json"
    receipt = read(receipt_path) if receipt_path.is_file() else None
    if guard is None and code == 0 and (receipt is None or receipt.get("status") != "complete"):
        guard = "missing_completion_receipt"
    result = {
        "schema": "chipmunk.video-appearance-resource-profile/108",
        "command": command,
        "batch": batch,
        "child_started": child is not None,
        "child_exit_code": code,
        "guard": guard,
        "sample_errors": len(errors),
        "error_types": errors,
        "terminal_observations": exits,
        "resource_limit_observations": [r for r in rows if r["violations"]],
        "duration_s": time.monotonic() - started,
        "sampled_tree_rss_lower_bound_bytes": max((r["tree_rss_bytes"] for r in rows), default=0),
        "minimum_available_ram_bytes": min((r["ram_available_bytes"] for r in rows), default=None),
        "resource_journal_sha256": sha(folder / "resources.jsonl"),
        "runtime_sha256": sha(Path(__file__)),
        "worker_receipt_sha256": sha(receipt_path) if receipt else None,
        "new_neural_calls": receipt["new_neural_calls"]
        if receipt
        else (0 if child is None or command in {"monitor", "bind"} else None),
        "scope": "Unchanged neural limits: 900s, 6-GiB sampled descendant RSS, 6-GiB initial RAM, 3-GiB initial disk, 1.5-GiB running reserves. Monitor/binding are data-only: 1.5-GiB RSS and 3-GiB initial RAM. Sequential explicit stages, no retries or overwrites; samples are not guaranteed peaks.",
    }
    write(folder / "profile.json", result)
    print(json.dumps(result), flush=True)
    if guard is not None or errors or code != 0:
        raise RuntimeError("Failed bounded stage retained; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("monitor", "bind", "authored", "extract"))
    parser.add_argument("batch", nargs="?", type=int)
    args = parser.parse_args()
    run(args.command, args.batch)
