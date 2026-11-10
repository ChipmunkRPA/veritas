"""Explicit continuation of only the never-attempted final audio chunks.

The v84 scientific worker, v88 selection, neural limits and completed outputs
remain unchanged. The owner explicitly requested continuation after freeing RAM.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts.audio_parallel_dispatch_v88 import passed, read, sha

BACKEND = Path(__file__).resolve().parents[1]
BASE = BACKEND / "data/independent-evaluation"
PRIMARY = BASE / "audio-combined-confidence-v84"
PARALLEL = BASE / "audio-common-reference-study-v88"
FULL = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/audio-confidence-extraction-v84")
REMOTE = {"audio-combined-confidence-v84-selection.json": "77f207c382beac7041d417d36ffcecdbf5764220",
          "audio-common-reference-v88-selection.json": "bea8c04e817c977cb9428f086ef25e4058078889"}


def unattempted_suffix(primary):
    completed = []
    for index in range(36, 48):
        files = [primary / f"chunk-{index:02d}{suffix}" for suffix in (".jsonl", "-profile", "-complete.json")]
        if not any(path.exists() for path in files):
            continue
        if not all(path.exists() for path in files):
            raise ValueError("An incomplete or failed attempted chunk cannot be retried")
        passed(files[1] / "profile.json")
        value = read(files[2])
        if value["completed"] != 87 or sha(files[0]) != value["compact_sha256"]:
            raise ValueError("Completed chunk is incomplete or changed")
        completed.append(index)
    if completed != list(range(36, 36 + len(completed))):
        raise ValueError("Only a complete contiguous prefix can precede continuation")
    return completed, list(range(36 + len(completed), 48))


def admission():
    completed, pending = unattempted_suffix(PRIMARY)
    if completed != [36] or pending != list(range(37, 48)):
        raise ValueError("This explicit continuation requires exactly the preserved first final batch")
    stop = read(PARALLEL / "confirmation-dispatch/stopped-before-chunk-37.json")
    if not (stop["observed_dispatcher_exit_code"] == 1 and stop["classification"] == "initial_neural_headroom_wait_expired_before_chunk_attempt"
            and stop["completed_chunks"] == [36] and stop["not_attempted_chunks"] == pending
            and sha(PRIMARY / "chunk-36-profile/profile.json") == stop["completed_profile_sha256"]):
        raise ValueError("Preserved terminal before-launch stop required")
    passed(PRIMARY / "train-profile/profile.json")
    passed(PARALLEL / "fit-profile/profile.json")
    if sha(PRIMARY / "trained.json") != read(PARALLEL / "freeze.json")["primary_trained_sha256"]:
        raise ValueError("Primary trained artifact changed")
    if sha(PARALLEL / "trained.json") != read(PARALLEL / "training-summary.json")["trained_sha256"]:
        raise ValueError("Parallel trained artifact changed")
    for trial, filename in ((PRIMARY, "audio-combined-confidence-v84-selection.json"),
                            (PARALLEL, "audio-common-reference-v88-selection.json")):
        raw = (BACKEND.parent / "docs/evaluation-panels" / filename).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        receipt = read(trial / "confirmation-start.json")
        if not (blob == REMOTE[filename] and receipt["remote_verified_before_confirmation_forwards"] is True
                and sha(trial / "selection.json") == hashlib.sha256(raw).hexdigest() == receipt["selection_sha256"]):
            raise ValueError("Exact remotely reverified frozen choice required")
        passed(trial / "select-profile/profile.json")
    original = read(PRIMARY / "chunk-36-complete.json")
    if sha(FULL / "chunk-36-full.jsonl") != original["full_sha256"]:
        raise ValueError("Preserved complete response journal changed")
    return pending


def run():
    pending = admission()
    folder = PARALLEL / "confirmation-continuation-v90"
    folder.mkdir(exist_ok=False)
    record = {"scope": "Owner-authorized continuation of never-attempted chunks only; no retry or overwrite",
              "completed_prefix": [36], "pending_chunks": pending, "controller_sha256": sha(Path(__file__)),
              "parent_stop_sha256": sha(PARALLEL / "confirmation-dispatch/stopped-before-chunk-37.json"),
              "primary_selection_sha256": sha(PRIMARY / "selection.json"),
              "parallel_selection_sha256": sha(PARALLEL / "selection.json"),
              "reverified_remote_blobs": REMOTE, "scientific_worker_and_neural_limits_unchanged": True}
    with (folder / "start.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    with (folder / "journal.jsonl").open("x") as journal:
        for index in pending:
            started, stable = time.monotonic(), 0
            while stable < 2:
                available = psutil.virtual_memory().available
                stable = stable + 1 if available >= 6 * 1024**3 else 0
                if stable >= 2:
                    break
                print(json.dumps({"awaiting_initial_headroom_for_chunk": index,
                                  "available_ram_bytes": available, "neural_worker_started": False}), flush=True)
                if time.monotonic() - started > 900:
                    raise RuntimeError("Before-launch headroom wait expired; no automatic retry")
                time.sleep(10)
            # Recheck exclusivity immediately before the unchanged worker creates its own profile.
            if any((PRIMARY / f"chunk-{index:02d}{suffix}").exists() for suffix in (".jsonl", "-profile", "-complete.json")):
                raise FileExistsError("An attempted chunk must not be overwritten")
            print(json.dumps({"starting_chunk": index, "preserved_completed_cases": 87}), flush=True)
            with (folder / f"chunk-{index:02d}.log").open("x") as log:
                child = subprocess.run([sys.executable, "-B", "-m", "scripts.audio_combined_runtime_v84", "extract", str(index)],
                                       cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT, check=False)
            journal.write(json.dumps({"chunk": index, "exit_code": child.returncode,
                                      "duration_s": time.monotonic() - started}) + "\n")
            journal.flush()
            if child.returncode:
                raise SystemExit(child.returncode)
            passed(PRIMARY / f"chunk-{index:02d}-profile/profile.json")
            if read(PRIMARY / f"chunk-{index:02d}-complete.json")["completed"] != 87:
                raise ValueError("Complete fixed chunk required")
            print(json.dumps({"completed_chunk": index, "total_final_cases_completed": (index - 35) * 87}), flush=True)
    with (folder / "complete.json").open("x") as stream:
        json.dump({"cases": 1044, "preserved_prefix": [36], "new_chunks": pending}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    run()
