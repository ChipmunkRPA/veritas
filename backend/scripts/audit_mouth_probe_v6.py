"""Replay native training provenance and video mouth-chip bindings without classifier execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.image import gend
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import mouth_chip, preprocessed


def check_chip(rgb, binding, detector, training=False):
    faces = detect_faces(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
    )
    transform = gend.align_transform(faces[0].landmarks) if faces else None
    if transform is None:
        raise ValueError("A retained mouth chip has no freshly detected face")
    matrix, side = transform
    if (
        side != binding["alignment_side"]
        or [[float(v).hex() for v in row] for row in matrix] != binding["alignment_matrix_hex"]
    ):
        raise ValueError("Retained alignment differs from fresh native-source geometry")
    aligned = gend.warp(rgb, matrix, side)
    field = "aligned_rgb_sha256" if training else "native_crop_rgb_sha256"
    if hashlib.sha256(aligned.tobytes()).hexdigest() != binding[field]:
        raise ValueError("Retained aligned pixels differ from native source replay")
    chip, bounds = mouth_chip(aligned)
    if (
        bounds != binding["chip_bounds" if training else "mouth_chip_bounds"]
        or hashlib.sha256(chip.tobytes()).hexdigest() != binding["mouth_chip_rgb_sha256"]
    ):
        raise ValueError("Retained mouth region differs from the measured native chip")
    if hashlib.sha256(preprocessed([chip])[0].tobytes()).hexdigest() != binding["model_input_float32_sha256"]:
        raise ValueError("Retained normalized model input differs from independent pixel replay")
    if training:
        values = np.asarray(binding["features"], np.float32)
        if (
            values.shape != (768,)
            or hashlib.sha256(values.tobytes()).hexdigest() != binding["features_float32_sha256"]
        ):
            raise ValueError("Retained feature vector changed")


def detector():
    from app.detectors.image.adapters.loaders import load_yunet
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    return load_yunet("yunet.2023mar", "cpu", "float32").obj


def training(plan_path, manifest_path, receipt, output):
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    if (
        manifest["plan_sha256"] != digest(plan_path)
        or header["manifest_sha256"] != digest(manifest_path)
        or [r["id"] for r in rows] != [r["id"] for r in manifest["items"]]
    ):
        raise ValueError("Missing or changed planned training outcomes")
    source_root = Path(plan["source_root"])
    face_detector = detector()
    records = []
    for source in plan["sources"]:
        path = source_root / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Original source bytes changed")
        table = probe_frames(path)
        originals = {f.index: f for f in decode_frames(path, table, source["selected_frames"])}
        for row in [r for r in rows if r["group"] == source["group"]]:
            record = {"id": row["id"], "status": "failed"}
            try:
                if (
                    row["status"] != "ok"
                    or row["role"] != source["role"]
                    or row["source_sha256"] != source["sha256"]
                ):
                    raise ValueError("Training outcome or source role changed")
                original = originals[row["frame_index"]]
                if (
                    row["pts"] != original.pts
                    or row["time_base"] != table.time_base
                    or hashlib.sha256(original.rgb.tobytes()).hexdigest() != row["original_rgb_sha256"]
                ):
                    raise ValueError("Training label refers to different original pixels or source time")
                image_path = manifest_path.parent / row["path"]
                if digest(image_path) != row["sha256"]:
                    raise ValueError("Training image bytes changed")
                rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["native_rgb_sha256"]:
                    raise ValueError("Training native RGB commitment changed")
                mask = manifest_path.parent / row["mask_path"]
                actual = np.any(rgb != original.rgb, axis=2)
                if (
                    digest(mask) != row["mask_sha256"]
                    or not np.array_equal(actual, cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 0)
                    or int(actual.sum()) != row["changed_pixels"]
                ):
                    raise ValueError("Native changes differ from their exact retained mask")
                if bool(actual.any()) != (row["condition"] != "original") or row["label"] != int(
                    row["condition"].startswith("ai-")
                ):
                    raise ValueError("Declared native edit or AI provenance label changed")
                check_chip(rgb, row, face_detector, training=True)
                record["status"] = "ok"
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            records.append(record)
    write_audit(
        records,
        output,
        {
            "mode": "training",
            "manifest_sha256": digest(manifest_path),
            "feature_receipt_sha256": digest(receipt),
            "planned": len(manifest["items"]),
            "independent_actors": len(plan["sources"]),
        },
    )


def video(manifest_path, baseline_path, receipt, output):
    manifest = json.loads(manifest_path.read_text())
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    _, *originals = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    if (
        header["manifest_sha256"] != digest(manifest_path)
        or header["baseline_receipt_sha256"] != digest(baseline_path)
        or [r["id"] for r in rows] != [r["id"] for r in manifest["items"]]
        or [r["id"] for r in originals] != [r["id"] for r in rows]
    ):
        raise ValueError("Changed or incomplete video comparison inputs")
    face_detector = detector()
    records = []
    for row, original in zip(rows, originals, strict=True):
        record = {"id": row["id"], "status": "failed", "verified_inputs": 0}
        try:
            path = manifest_path.parent / row["path"]
            if (
                row["status"] != "ok"
                or not row.get("baseline_replay_parity")
                or row["base_windows"] != original["base_windows"]
                or digest(path) != row["sha256"]
            ):
                raise ValueError("Incomplete source outcome or changed baseline")
            table = probe_frames(path)
            bindings = row["model_inputs"]
            indices = [r["frame_index"] for r in bindings]
            expected = {
                i + k for i in original["sampling_centers"] if 0 < i < len(table) - 1 for k in (-1, 0, 1)
            }
            if indices != sorted(expected):
                raise ValueError("Retained input ordering differs from the exact baseline source schedule")
            for binding, frame in zip(bindings, decode_frames(path, table, indices), strict=True):
                if (
                    binding["frame_index"] != frame.index
                    or binding["pts"] != frame.pts
                    or binding["time_base"] != table.time_base
                    or binding["time_s"] != table.seconds(frame.index)
                    or binding["decoded_rgb_sha256"] != hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                ):
                    raise ValueError("Classifier input is attached to different source pixels or time")
                check_chip(frame.rgb, binding, face_detector)
                record["verified_inputs"] += 1
            record["status"] = "ok"
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    write_audit(
        records,
        output,
        {
            "mode": "video",
            "manifest_sha256": digest(manifest_path),
            "score_receipt_sha256": digest(receipt),
            "planned": len(manifest["items"]),
            "verified_inputs": sum(r["verified_inputs"] for r in records),
        },
    )


def write_audit(records, output, metadata):
    result = {
        "schema": "chipmunk.mouth-trained-source-audit/6",
        **metadata,
        "implementation_sha256": digest(Path(__file__)),
        "passed": sum(r["status"] == "ok" for r in records),
        "scope": "Local source/mask/region/tensor correspondence; not outside review or classifier accuracy",
        "cases": records,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
    if result["passed"] != result["planned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["training", "video"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path, help="Generation plan for training; baseline receipt for video")
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.stage == "training":
        training(a.input, a.manifest, a.receipt, a.output)
    else:
        video(a.manifest, a.input, a.receipt, a.output)
