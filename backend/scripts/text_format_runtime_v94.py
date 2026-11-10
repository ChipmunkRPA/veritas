"""Fixed text-format model response batches under the unchanged neural limits."""

from __future__ import annotations

import argparse
import hashlib
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

BACKEND = Path(__file__).resolve().parents[1]
TRIAL = BACKEND / "data/independent-evaluation/text-format-comparison-v94"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")


def run(command, index=None):
    if command != "score" or type(index) is not int or not 0 <= index < 8:
        raise ValueError("One fixed neural response batch required")
    name, args = f"batch-{index:02d}", ["score", str(index)]
    outputs = [TRIAL / f"{name}.jsonl", TRIAL / f"{name}-complete.json"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Preserve existing or partially attempted response comparison")
    folder = TRIAL / f"{name}-profile"
    folder.mkdir(parents=True, exist_ok=False)
    env = {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"}
    }
    env.update(
        PYTHONPATH=str(BACKEND),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        VERITAS_DEVICE="cpu",
    )
    started, child, guard, code = time.monotonic(), None, None, None
    rows, errors, exits = [], [], []
    with (folder / "resources.jsonl").open("x") as journal, (folder / "worker.log").open("x") as log:

        def record(rss, initial=False):
            row = original.observe(folder, started, rss)
            row["violations"] = original.violations(row, initial=initial)
            row["archive_free_disk_bytes"] = shutil.disk_usage(ARCHIVE).free
            if not Path("/Volumes/SEAGATE").is_mount():
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
                child = subprocess.Popen(
                    [sys.executable, "-B", "-m", "scripts.text_format_comparison_v94", *args],
                    cwd=BACKEND,
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
                        # Missing PID is not success: wait for this exact worker.
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
                # A failing worker must not leave a decoder running in its session.
                if guard is not None or code != 0:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
            try:
                if record(0) and guard is None:
                    guard = "final_resource_limit"
            except BaseException as exc:
                errors.append(type(exc).__name__)
                guard = guard or "monitor_unavailable"
    result = {
        "schema": "chipmunk.text-format-resource-profile/94",
        "arguments": args,
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
        "resource_journal_sha256": hashlib.sha256((folder / "resources.jsonl").read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Sequential fixed-model response batches: unchanged 900 seconds, 6-GiB sampled descendant RSS/initial available RAM, 3-GiB initial disk and 1.5-GiB running RAM/disk reserve. Samples are not true peaks. Actual neural calls are counted by the worker. No retry, overwrite, fitting or serving change.",
    }
    with (folder / "profile.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if guard is not None or errors or code != 0:
        raise RuntimeError("Failed response stage retained; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score",))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    run(args.command, args.index)
