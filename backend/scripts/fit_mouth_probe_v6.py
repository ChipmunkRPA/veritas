"""Freeze, extract and fit a source-separated experimental mouth head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import BATCH, METHOD, features, mouth_chip, train_head


def implementation():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "backend/scripts/mouth_probe_v6.py",
        "backend/scripts/fit_mouth_probe_v6.py",
        "backend/scripts/compare_mouth_probe_v6.py",
        "docs/mouth-trained-v6-development.md",
        "backend/app/detectors/image/gend.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/video/frame_store.py",
        "backend/app/detectors/video/face_sequence.py",
        "backend/app/detectors/video/adapters/face_sequence.py",
        "backend/app/detectors/video/adapters/loaders.py",
        "backend/app/detectors/video/adapters/tracks.py",
        "backend/app/detectors/video/tracking.py",
        "backend/scripts/compare_mouth_sequence_v4.py",
        "backend/scripts/compare_mouth_artifact_v5.py",
        "backend/app/models.yaml",
    ]
    return {p: digest(root / p) for p in paths}


def freeze(plan, manifest, baseline_manifest, baseline, output):
    m = json.loads(manifest.read_text())
    recipe = json.loads(plan.read_text())
    if (
        m["plan_sha256"] != digest(plan)
        or len(m["items"]) != 800
        or any(r["status"] != "ready" for r in m["items"])
    ):
        raise ValueError("Every one of the 800 predeclared training frames must be available")
    excluded = set(recipe["excluded_development_groups"])
    if {r["group"] for r in m["items"]} & excluded:
        raise ValueError("Fitting inputs overlap development actors")
    body = {
        "schema": "chipmunk.mouth-trained-lock/6",
        "method": METHOD,
        "training_plan_sha256": digest(plan),
        "training_manifest_sha256": digest(manifest),
        "baseline_manifest_sha256": digest(baseline_manifest),
        "baseline_receipt_sha256": digest(baseline),
        "implementation_sha256": implementation(),
        "deployment_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"lock_sha256": digest(output), "planned_frames": 800}))


def check_lock(lock):
    if lock["implementation_sha256"] != implementation():
        raise ValueError("Frozen feature, fit or comparison implementation changed")


def extract(manifest, lock_path, output):
    import torch

    from app.core.config import settings
    from app.detectors.image import gend
    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.image.faces import detect_faces
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    lock = json.loads(lock_path.read_text())
    check_lock(lock)
    if digest(manifest) != lock["training_manifest_sha256"]:
        raise ValueError("Frozen training frames changed")
    if not torch.backends.mps.is_available():
        raise ValueError("This feature lock requires MPS")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    weights = {}
    for mid in ["yunet.2023mar", "dinov2.small"]:
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Commercial checkpoint declaration required")
        weights[mid] = [{"name": p.name, "sha256": digest(p)} for p in ensure_verified(spec, force=True)]
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    dino = load_dinov2_small("dinov2.small", "mps", "float32").obj
    header = {
        "schema": "chipmunk.mouth-training-features/6",
        "lock_sha256": digest(lock_path),
        "manifest_sha256": digest(manifest),
        "weights": weights,
        "device": "mps",
        "torch_version": torch.__version__,
    }
    rows = json.loads(manifest.read_text())["items"]
    complete = []
    if output.exists():
        old, *complete = [json.loads(l) for l in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r["id"] for r in rows[: len(complete)]]:
            raise ValueError("Cannot resume different fitting inputs")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for start in range(len(complete), len(rows), BATCH):
            batch = rows[start : start + BATCH]
            chips = []
            records = []
            for row in batch:
                record = {**row, "status": "failed"}
                try:
                    path = manifest.parent / row["path"]
                    if digest(path) != row["sha256"]:
                        raise ValueError("Prepared training file changed")
                    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["native_rgb_sha256"]:
                        raise ValueError("Prepared RGB bytes changed")
                    faces = detect_faces(
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        detector=detector,
                        min_size=settings.face_min_size,
                    )
                    transform = gend.align_transform(faces[0].landmarks) if faces else None
                    if transform is None:
                        raise ValueError("A training frame has no measurable production face")
                    aligned = gend.warp(rgb, *transform)
                    chip, bounds = mouth_chip(aligned)
                    chips.append(chip)
                    record.update(
                        status="ok",
                        chip_bounds=bounds,
                        alignment_side=transform[1],
                        alignment_matrix_hex=[[float(v).hex() for v in r] for r in transform[0]],
                        aligned_rgb_sha256=hashlib.sha256(aligned.tobytes()).hexdigest(),
                        mouth_chip_rgb_sha256=hashlib.sha256(chip.tobytes()).hexdigest(),
                    )
                except Exception as error:
                    record["reason"] = f"{type(error).__name__}: {error}"
                records.append(record)
            if chips:
                values, hashes = features(chips, dino)
                k = 0
                for record in records:
                    if record["status"] == "ok":
                        record.update(
                            features=values[k].tolist(),
                            features_float32_sha256=hashlib.sha256(values[k].tobytes()).hexdigest(),
                            model_input_float32_sha256=hashes[k],
                        )
                        k += 1
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "complete": start + len(batch),
                        "total": len(rows),
                        "failed_in_batch": sum(r["status"] != "ok" for r in records),
                    }
                ),
                flush=True,
            )


def fit(manifest, receipt, lock_path, output):
    import numpy as np
    import sklearn

    lock = json.loads(lock_path.read_text())
    check_lock(lock)
    header, *rows = [json.loads(l) for l in receipt.read_text().splitlines()]
    expected = json.loads(manifest.read_text())["items"]
    if (
        digest(manifest) != lock["training_manifest_sha256"]
        or header["manifest_sha256"] != digest(manifest)
        or header["lock_sha256"] != digest(lock_path)
        or [r["id"] for r in rows] != [r["id"] for r in expected]
        or any(r["status"] != "ok" for r in rows)
    ):
        raise ValueError("The complete frozen source-separated feature panel is required")
    for row, source in zip(rows, expected, strict=True):
        if any(row[k] != source[k] for k in ("group", "role", "condition", "label", "sha256", "frame_index")):
            raise ValueError("Feature labels or source roles changed")
        values = np.asarray(row["features"], np.float32)
        if (
            values.shape != (768,)
            or hashlib.sha256(values.tobytes()).hexdigest() != row["features_float32_sha256"]
        ):
            raise ValueError("Retained frozen features changed")
    head = train_head(rows)
    if len(head["fit_groups"]) != 16 or len(head["point_groups"]) != 4:
        raise ValueError("Source actor roles changed")
    head.update(
        schema="chipmunk.mouth-trained-head/6",
        lock_sha256=digest(lock_path),
        features_receipt_sha256=digest(receipt),
        weights=header["weights"],
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
    )
    with output.open("x") as stream:
        json.dump(head, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "head_sha256": digest(output),
                "fit_actors": len(head["fit_groups"]),
                "point_actors": len(head["point_groups"]),
                "review_point": head["review_point"],
                "iterations": head["iterations"],
                "response_kind": head["response_kind"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "extract", "fit"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--baseline-manifest", type=Path)
    p.add_argument("--lock", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        if not a.plan or not a.baseline_manifest:
            p.error("freeze requires plan and baseline inputs")
        freeze(a.plan, a.manifest, a.baseline_manifest, a.input, a.output)
    elif a.stage == "extract":
        extract(a.manifest, a.input, a.output)
    else:
        if not a.lock:
            p.error("fit requires --lock")
        fit(a.manifest, a.input, a.lock, a.output)
