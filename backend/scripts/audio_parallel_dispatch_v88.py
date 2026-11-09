"""Sequential dispatch of unchanged primary extraction with both confirmation guards."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

BACKEND = Path(__file__).resolve().parents[1]
PRIMARY = BACKEND / "data/independent-evaluation/audio-combined-confidence-v84"
PARALLEL = PRIMARY.parent / "audio-common-reference-study-v88"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def passed(path):
    value = read(path)
    if value["child_exit_code"] != 0 or value["guard"] is not None or value["sample_errors"] or value["resource_limit_observations"]:
        raise RuntimeError("Required prior stage failed")


def dispatch(role):
    if role not in {"selection", "confirmation"}:
        raise ValueError("One explicit held role required")
    passed(PRIMARY / "train-profile/profile.json")
    passed(PARALLEL / "fit-profile/profile.json")
    if sha(PRIMARY / "trained.json") != read(PARALLEL / "freeze.json")["primary_trained_sha256"]:
        raise ValueError("Primary model changed before parallel extraction")
    if sha(PARALLEL / "trained.json") != read(PARALLEL / "training-summary.json")["trained_sha256"]:
        raise ValueError("Parallel calibration changed")
    if role == "confirmation":
        public = BACKEND.parent / "docs/evaluation-panels"
        for trial, filename in ((PRIMARY, "audio-combined-confidence-v84-selection.json"),
                                (PARALLEL, "audio-common-reference-v88-selection.json")):
            receipt = read(trial / "confirmation-start.json")
            if not (receipt["remote_verified_before_confirmation_forwards"] is True
                    and sha(trial / "selection.json") == sha(public / filename) == receipt["selection_sha256"]):
                raise ValueError("Both exact published choices must precede final forwards")
            passed(trial / "select-profile/profile.json")
    first, end = (24, 36) if role == "selection" else (36, 48)
    for index in range(first, end):
        if (PRIMARY / f"chunk-{index:02d}.jsonl").exists() or (PRIMARY / f"chunk-{index:02d}-profile").exists():
            raise FileExistsError("No dispatcher retry or overwrite of an attempted chunk")
    folder = PARALLEL / (role + "-dispatch")
    folder.mkdir(exist_ok=False)
    record = {"role": role, "range": [first, end - 1], "dispatcher_sha256": sha(Path(__file__)),
              "primary_trained_sha256": sha(PRIMARY / "trained.json"), "parallel_trained_sha256": sha(PARALLEL / "trained.json"),
              "scope": "Unchanged primary module/commands, sequential, stop on first failure; no fitting or outcome-based choice"}
    with (folder / "start.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    with (folder / "journal.jsonl").open("x") as journal:
        for index in range(first, end):
            started = time.monotonic()
            # Wait before attempting a chunk; no scientific output is retried.
            stable = 0
            while stable < 2:
                available = psutil.virtual_memory().available
                stable = stable + 1 if available >= 6 * 1024 ** 3 else 0
                if stable >= 2:
                    break
                print(json.dumps({"role": role, "awaiting_initial_headroom_for_chunk": index,
                                  "available_ram_bytes": available, "neural_worker_started": False}), flush=True)
                if time.monotonic() - started > 900:
                    raise RuntimeError("Initial neural headroom unavailable; this chunk was not attempted")
                time.sleep(10)
            print(json.dumps({"role": role, "starting_chunk": index}), flush=True)
            with (folder / f"chunk-{index:02d}.log").open("x") as log:
                child = subprocess.run([sys.executable, "-B", "-m", "scripts.audio_combined_runtime_v84", "extract", str(index)],
                                       cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {"chunk": index, "exit_code": child.returncode, "duration_s": time.monotonic() - started}
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            if child.returncode != 0:
                print(json.dumps({"role": role, "stopped_chunk": index}), flush=True)
                raise SystemExit(child.returncode)
            passed(PRIMARY / f"chunk-{index:02d}-profile/profile.json")
            if read(PRIMARY / f"chunk-{index:02d}-complete.json")["completed"] != 87:
                raise RuntimeError("Complete fixed chunk required")
            print(json.dumps({"role": role, "completed_chunk": index, "role_cases_completed": (index - first + 1) * 87}), flush=True)
    with (folder / "complete.json").open("x") as stream:
        json.dump({"role": role, "cases": 1044, "source_chunks": list(range(first, end))}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("selection", "confirmation"))
    dispatch(parser.parse_args().role)
