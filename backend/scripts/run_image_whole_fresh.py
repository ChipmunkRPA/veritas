# SPDX-License-Identifier: Apache-2.0
"""Measure a frozen whole-image transfer with the reviewed CPU candidate and actual Full pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import os
import time
from pathlib import Path
from types import FunctionType

from scripts import image_whole_fresh_sources as inputs
from scripts import run_image_smaug_v7 as prior

ROOT = Path(__file__).resolve().parents[2]
OLD_LOCK = ROOT / "docs/evaluation-panels/image-v7-smaug-development-freeze.json"
PROCESSOR = ROOT / "docs/evaluation-panels/image-v7-smaug-processor-completion.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepared(directory):
    lock = inputs.read(directory / "freeze.json")
    inputs.require(lock["schema"] == "chipmunk.image-whole-fresh-source-freeze/1", "Unknown input freeze")
    commitments = {directory / "plan.json": lock["plan_sha256"], directory / "intake.json": lock["intake_sha256"],
                   ROOT / inputs.POINT: lock["prior_operating_point_sha256"],
                   ROOT / inputs.WEIGHTS: lock["candidate_weight_record_sha256"]}
    inputs.require(all(sha(path) == expected for path, expected in commitments.items()), "Frozen study changed")
    panel = []
    for item in lock["items"]:
        path = (directory / item["path"]).resolve()
        inputs.require(path.is_relative_to(directory) and sha(path) == item["sha256"], "Prepared image bytes changed")
        panel.append({**item, "id": f"{item['source_index']:03d}-{item['format']}",
                      "input_path": str(path), "input_sha256": item["sha256"]})
    ids = {row["id"] for row in panel}
    inputs.require(len(ids) == len(panel) == lock["criteria"]["minimum_complete_deliveries"], "Frozen deliveries incomplete")
    return lock, panel


def candidate(directory):
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    from scripts.audit_image_smaug_runtime_v7 import check_tensor_file
    from scripts.image_smaug_views_v7 import views

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    lock, panel = prepared(directory)
    old = inputs.read(OLD_LOCK)["record"]
    packages = old["candidate_packages"]
    inputs.require(all(importlib.metadata.version(name) == version for name, version in packages.items())
                   and prior.runtime_files(ROOT, candidate=True) == old["candidate_installed_implementation_sha256"],
                   "Reviewed candidate runtime changed")
    model_dir = ROOT / "models/research/image-v7-smaug"
    assets = {item["name"]: item["sha256"] for item in inputs.read(ROOT / inputs.WEIGHTS)["files"]}
    extra = inputs.read(PROCESSOR)["added_asset"]
    assets[extra["name"]] = extra["sha256"]
    inputs.require(assets == old["candidate_assets"] and all(sha(model_dir / name) == pin for name, pin in assets.items()),
                   "Reviewed model or processor bytes changed")
    journal = directory / "candidate.jsonl"
    with journal.open("x"):
        pass
    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False,
                                config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    model.requires_grad_(False).eval()
    base = check_tensor_file(model[0].auto_model, model_dir / "model.safetensors", 438760472)
    head = check_tensor_file(model[3], model_dir / "3_Dense/model.safetensors", 769)
    processor = model[0].processor.image_processor
    processor.max_soft_tokens = processor.image_seq_length = 140
    point = lock["operating_point"]["raw_mean_logit"]
    for row in panel:
        started = time.monotonic()
        path = Path(row["input_path"])
        inputs.require(sha(path) == row["input_sha256"], "Image changed during inference")
        with Image.open(path) as image:
            crops, traces = views(image, 3)
        with torch.inference_mode():
            logits = model.encode([{"image": crop} for crop in crops], batch_size=3, convert_to_tensor=True,
                                  show_progress_bar=False).float().reshape(-1)
        inputs.require(logits.shape == (3,) and torch.isfinite(logits).all(), "Three finite crop responses required")
        mean = float(logits.mean())
        raw = logits.cpu().tolist()
        inputs.require(abs(mean - math.fsum(raw) / 3) <= 1e-4, "Candidate mean differs from independent arithmetic")
        prior.append(journal, {"id": row["id"], "input_sha256": row["input_sha256"], "status": "ok",
                               "mean_logit": mean, "crop_logits": raw, "flagged": mean > point, "traces": traces,
                               "crop_rgb_sha256": [inputs.digest(crop.tobytes()) for crop in crops],
                               "elapsed_seconds": round(time.monotonic() - started, 4)})
        print(row["id"], "ok", flush=True)
    prepared(directory)
    inputs.write(directory / "candidate-complete.json", {"count": len(panel), "input_freeze_sha256": sha(directory / "freeze.json"),
                 "candidate_journal_sha256": sha(journal), "pinned_assets": assets, "base": base, "head": head, "point": point})


def baseline_sources(directory):
    import app

    snapshot = directory / "checkout/backend"
    inputs.require(Path(app.__file__).resolve().parent == snapshot / "app", "Baseline is not running the frozen application copy")
    inputs.require((snapshot / "alembic/versions").is_dir(), "Frozen database migrations are absent")
    sources = {str(path.relative_to(snapshot)): sha(path) for path in sorted(snapshot.rglob("*"))
               if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".npy", ".j2", ".css", ".ini", ".mako"}}
    return {"commit": (directory / "checkout/commit.txt").read_text().strip(), "application": sources,
            "runner": sha(Path(__file__)), "full_pipeline_runner": sha(Path(prior.__file__))}


def freeze_baseline(directory):
    prepared(directory)
    context = prior.baseline_context(ROOT, directory / "baseline-freeze-context")
    frozen = {"baseline": context, "source_sha256": baseline_sources(directory), "input_freeze_sha256": sha(directory / "freeze.json")}
    inputs.write(directory / "baseline-lock.json", frozen)


def baseline(directory):
    _, panel = prepared(directory)
    frozen = inputs.read(directory / "baseline-lock.json")

    def load_lock(path):
        inputs.require(baseline_sources(path) == frozen["source_sha256"] and sha(path / "freeze.json") == frozen["input_freeze_sha256"],
                       "Full pipeline changed after scoring froze")
        return frozen, {"items": panel}

    # Reuse the existing Full pipeline runner with this study's own lock.
    original = prior.baseline
    bound = FunctionType(original.__code__, {**original.__globals__, "load_lock": load_lock})
    bound.__kwdefaults__ = original.__kwdefaults__
    bound(ROOT, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("candidate", "freeze-baseline", "baseline"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    {"candidate": candidate, "freeze-baseline": freeze_baseline, "baseline": baseline}[args.arm](args.directory.resolve())


if __name__ == "__main__":
    main()
