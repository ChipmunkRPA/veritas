"""Bound the separate standard-library text-format arithmetic reviewer."""

from __future__ import annotations

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

from scripts.audio_tree_sampler_v64 import sample_tree

BACKEND = Path(__file__).resolve().parents[1]
REVIEW = BACKEND / "data/independent-evaluation/text-format-independent-review-v99"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")


def run():
    folder = REVIEW / "run-profile"
    folder.mkdir(exist_ok=False)
    started, child, code, guard = time.monotonic(), None, None, None
    samples, errors, events = [], [], []
    with (folder / "resources.jsonl").open("x") as journal, (folder / "worker.log").open("x") as log:

        def sample(rss, initial=False):
            row = {
                "elapsed_s": time.monotonic() - started,
                "tree_rss_bytes": rss,
                "available_ram_bytes": psutil.virtual_memory().available,
                "project_free_disk_bytes": shutil.disk_usage(folder).free,
                "archive_free_disk_bytes": shutil.disk_usage(ARCHIVE).free,
            }
            minimum = 3 * 1024**3 if initial else 1536 * 1024**2
            crossed = (
                row["elapsed_s"] > 900
                or rss > 1536 * 1024**2
                or min(
                    row["available_ram_bytes"], row["project_free_disk_bytes"], row["archive_free_disk_bytes"]
                )
                < minimum
                or not Path("/Volumes/SEAGATE").is_mount()
            )
            row["limit_crossed"] = crossed
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            samples.append(row)
            return crossed

        try:
            if sample(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen(
                    [sys.executable, "-B", "-m", "scripts.review_text_format_v99", "review"],
                    cwd=BACKEND,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    rss, observations, terminal = sample_tree(process, child)
                    events.extend(observations)
                    if sample(rss):
                        guard = "data_resource_limit"
                        break
                    if terminal:
                        break
                    time.sleep(0.25)
        except BaseException as error:
            errors.append(type(error).__name__)
            guard = "monitor_failure"
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
    profile = {
        "schema": "chipmunk.text-format-review-resource-profile/99",
        "child_started": child is not None,
        "child_exit_code": code,
        "guard": guard,
        "sample_errors": errors,
        "terminal_observations": events,
        "duration_s": time.monotonic() - started,
        "maximum_sampled_tree_rss_bytes": max((s["tree_rss_bytes"] for s in samples), default=0),
        "minimum_available_ram_bytes": min((s["available_ram_bytes"] for s in samples), default=None),
        "resource_journal_sha256": hashlib.sha256((folder / "resources.jsonl").read_bytes()).hexdigest(),
        "scope": "Separate non-neural arithmetic review: 900s,1.5GiB sampled tree RSS,3GiB initial RAM/disk,1.5GiB running reserves; samples are not true peaks. Existing neural budgets unchanged. No overwrite or retry.",
    }
    with (folder / "profile.json").open("x") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    print(json.dumps(profile), flush=True)
    if guard is not None or errors or code != 0:
        raise RuntimeError("Unsuccessful review retained; no automatic retry")


if __name__ == "__main__":
    run()
