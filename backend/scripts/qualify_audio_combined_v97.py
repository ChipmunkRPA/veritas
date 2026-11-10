"""Export and replay the selected audio combination; no fitting or neural calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

from app.detectors.audio import combined
from scripts import audio_score_fusion_v83 as reference

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
SOURCE = BASE / "audio-combined-confidence-v84"
TRIAL = BASE / "audio-combined-serving-v97"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
JOURNALS = ARCHIVE / "audio-confidence-extraction-v84"
PUBLIC = ROOT / "docs/evaluation-panels/audio-combined-serving-freeze-v97.json"
BINDINGS = {
    "study_freeze_sha256": "5a6cefd6d09b7a78b5699f2a625aeaff3d0886d57621be916df04be134e31b17",
    "selected_global_sha256": "25aa0c565e003b9b578046aa155be24e37d3667f75d7d8ebe09a94b52aa209e5",
    "trained_fusion_sha256": "0aa21efdad351eebbc2390f241c49f1e40d55d188277db442f092c7b1ef668ae",
    "complement_head_sha256": "23350d709c9dc4c968066e2ab36bb16792f5c18d2513408b5f5c3298a8f83ae9",
}
FILES = {
    "study_freeze_sha256": SOURCE / "freeze.json",
    "selected_global_sha256": SOURCE / "selection.json",
    "trained_fusion_sha256": SOURCE / "trained.json",
    "complement_head_sha256": BASE / "audio-v13-wavlm/trained/head.json",
}
CODE = (
    "backend/app/detectors/audio/combined.py",
    "backend/app/detectors/audio/content.py",
    "backend/scripts/qualify_audio_combined_v97.py",
    "backend/scripts/audio_score_fusion_v83.py",
    "backend/tests/test_audio_combined_serving.py",
    "docs/audio-combined-serving-v97-protocol.md",
)
require, sha = combined.require, combined.sha


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class NumericalBudget:
    """In-process NumPy/JSON stages; no descendant or neural workers are used."""

    def __init__(self, path):
        path.mkdir(exist_ok=False)
        self.path, self.started = path, time.monotonic()
        self.stop, self.rows, self.errors = threading.Event(), [], []
        self.stream = (path / "resources.jsonl").open("x")
        self.thread = None

    def observe(self, initial=False):
        row = {
            "elapsed_s": time.monotonic() - self.started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "project_free_disk_bytes": shutil.disk_usage(BASE).free,
            "archive_free_disk_bytes": shutil.disk_usage(ARCHIVE).free,
        }
        self.rows.append(row)
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        require(
            Path("/Volumes/SEAGATE").is_mount()
            and row["elapsed_s"] <= 900
            and row["rss_bytes"] <= 1536 * 1024**2
            and row["available_ram_bytes"] >= (3 * 1024**3 if initial else 1536 * 1024**2)
            and min(row["project_free_disk_bytes"], row["archive_free_disk_bytes"])
            >= (3 * 1024**3 if initial else 1536 * 1024**2),
            "Numerical serving qualification resource guard",
        )

    def check(self):
        require(not self.errors, "Resource sampling failed or a limit was reached")
        require("torch" not in sys.modules, "Numerical qualification must not load a neural library")

    def __enter__(self):
        try:
            self.observe(initial=True)
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

        def watch():
            try:
                while not self.stop.wait(0.25):
                    self.observe()
            except BaseException as exc:
                self.errors.append(type(exc).__name__)

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        require(not self.thread or not self.thread.is_alive(), "Resource observer did not stop")
        self.stream.close()
        write(
            self.path / "profile.json",
            {
                "status": "complete" if kind is None and not self.errors else "failed",
                "error_type": kind.__name__ if kind else None,
                "sample_errors": self.errors,
                "duration_s": time.monotonic() - self.started,
                "maximum_sampled_rss_bytes": max((r["rss_bytes"] for r in self.rows), default=0),
                "minimum_available_ram_bytes": min(
                    (r["available_ram_bytes"] for r in self.rows), default=None
                ),
                "resource_journal_sha256": sha(self.path / "resources.jsonl"),
                "new_neural_calls": 0,
                "scope": "In-process numerical stage sampled every0.25s; no descendant worker.900s/1.5GiB RSS/3GiB initial RAM and disk/1.5GiB running reserves. Samples are not true peaks. Neural limits unchanged. No retry/overwrite.",
            },
        )
        if kind is None:
            self.check()


def source_artifacts():
    require(all(sha(FILES[k]) == v for k, v in BINDINGS.items()), "Frozen selected head or fusion changed")
    selection = read(FILES["selected_global_sha256"])
    require(
        selection["selected_arm"] == "global"
        and selection["trained_sha256"] == BINDINGS["trained_fusion_sha256"],
        "Previously selected global combination required",
    )
    require(
        read(ROOT / "docs/evaluation-panels/audio-combined-confidence-v84-selection.json") == selection,
        "Published selection differs",
    )
    source = read(FILES["trained_fusion_sha256"])["global"]
    require(
        source["arm"] == "global" and source["branches"]["all"]["status"] == "calibrated",
        "Selected own-calibration map required",
    )
    return read(FILES["complement_head_sha256"])["fitted"]["complement"], source


def prepare(budget):
    head, fusion = source_artifacts()
    require(not TRIAL.exists() and not PUBLIC.exists(), "Never attempted serving qualification required")
    TRIAL.mkdir()
    bundle = {
        "schema": combined.SCHEMA,
        "source_bindings": BINDINGS,
        "complement": {
            **{k: head["transform"][k] for k in ("center", "scale", "variance_factor")},
            "coefficients": head["coefficients"],
        },
        "global": {k: fusion["branches"]["all"][k] for k in ("coefficients", "calibration")},
    }
    bundle_path = TRIAL / "serving-bundle.json"
    write(bundle_path, bundle)
    combined.load_calibration(bundle_path, sha(bundle_path))
    freeze = {
        "schema": "chipmunk.audio-combined-serving-freeze/97",
        "source_bindings": BINDINGS,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "private_inference_bundle_sha256": sha(bundle_path),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "psutil")},
        "replay_chunks": 48,
        "replay_inputs": 4176,
        "maximum_absolute_probability_error": 1e-12,
        "scope": "Inference-only numerical parity on completed controlled research inputs, not new inference, fresh accuracy, full serving integration or a default change. Selected global arm and own calibration remain fixed. Learned arrays and individual records stay private.",
        "new_neural_calls": 0,
        "defaults_changed": False,
    }
    budget.check()
    write(TRIAL / "freeze.json", freeze)
    write(PUBLIC, freeze)
    print(json.dumps({"status": "prepared", "planned_replays": 4176, "new_neural_calls": 0}), flush=True)


def replay(budget):
    _, fusion = source_artifacts()
    frozen = read(TRIAL / "freeze.json")
    require(
        frozen == read(PUBLIC) and frozen["code_sha256"] == {p: sha(ROOT / p) for p in CODE},
        "Exact published serving freeze required",
    )
    require(
        read(TRIAL / "execution-start.json")["remote_freeze_blob_verified"] is True,
        "Remote publication must be checked before replay",
    )
    model = combined.load_calibration(
        TRIAL / "serving-bundle.json", frozen["private_inference_bundle_sha256"]
    )
    counts, role_counts, maximum = Counter(), {}, 0.0
    bindings = []
    for index in range(48):
        budget.check()
        compact, full = SOURCE / f"chunk-{index:02d}.jsonl", JOURNALS / f"chunk-{index:02d}-full.jsonl"
        receipt = read(SOURCE / f"chunk-{index:02d}-complete.json")
        profile = read(SOURCE / f"chunk-{index:02d}-profile/profile.json")
        require(
            profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Completed original neural stage required",
        )
        require(
            receipt == {"completed": 87, "compact_sha256": sha(compact), "full_sha256": sha(full)},
            "Complete original response journals required",
        )
        rows = 0
        with compact.open() as cstream, full.open() as fstream:
            for cline, fline in zip(cstream, fstream, strict=True):
                row, small = json.loads(fline), json.loads(cline)
                require(
                    small["full_record_sha256"] == hashlib.sha256(fline.encode()).hexdigest()
                    and all(v == row[k] for k, v in small.items() if k != "full_record_sha256"),
                    "Compact and full measured records differ",
                )
                expected = reference.predict(row, fusion)
                if row["status"] == "ok":
                    features = np.asarray(row["wavlm"], np.float32)
                    require(
                        features.shape == (combined.WIDTH,)
                        and np.isfinite(features).all()
                        and features.astype(np.float64).tolist() == row["wavlm"]
                        and hashlib.sha256(features.astype("<f4").tobytes()).hexdigest()
                        == row["wavlm_sha256"]
                        and row["input_pcm_sha256"] == row["pcm_sha256"]
                        and row["feature_frames"] == row["valid_feature_frames"] == 199
                        and row["windows"] == [{"start_sample": 0, "end_sample": 64000}],
                        "Exact assessed four-second feature packet required",
                    )
                    p = model.predict(row["raw"], features)
                    error = abs(p - expected["probability_ai"])
                    require(
                        math.isfinite(error) and error <= frozen["maximum_absolute_probability_error"],
                        "Serving calibration numerical parity failed",
                    )
                    direction = "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive"
                    require(
                        direction == expected["decision"],
                        "Serving tendency differs from frozen research rule",
                    )
                    maximum = max(maximum, error)
                    counts["assessed"] += 1
                else:
                    require(
                        row["status"] == "not_applicable"
                        and row["raw"] is None
                        and row["wavlm"] is None
                        and row["windows"] == []
                        and expected["probability_ai"] is None
                        and expected["decision"] == "unassessed",
                        "Skipped response cannot supply original evidence",
                    )
                    counts["unassessed"] += 1
                role_counts.setdefault(row["role"], Counter())[row["status"]] += 1
                rows += 1
                budget.check()
        require(rows == 87, "Every planned original row required")
        bindings.append({"compact_sha256": receipt["compact_sha256"], "full_sha256": receipt["full_sha256"]})
    require(sum(counts.values()) == 4176, "Full numerical comparison required")
    budget.check()
    result = {
        "schema": "chipmunk.audio-combined-serving-replay/97",
        "status": "complete",
        "cases": 4176,
        "assessed_replayed": counts["assessed"],
        "unassessed_preserved": counts["unassessed"],
        "maximum_absolute_probability_error": maximum,
        "roles": role_counts,
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "private_bundle_sha256": model.sha256,
        "complete_input_binding_sha256": hashlib.sha256(
            json.dumps(bindings, sort_keys=True).encode()
        ).hexdigest(),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Completed recorded-feature and score replay, not fresh model execution, full analysis/payment/export qualification, external review or new detector accuracy.",
    }
    write(TRIAL / "replay-result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "replay"))
    command = parser.parse_args().command
    folder = (
        BASE / "audio-combined-serving-v97-preparation-profile"
        if command == "prepare"
        else TRIAL / "replay-profile"
    )
    with NumericalBudget(folder) as budget:
        {"prepare": prepare, "replay": replay}[command](budget)
