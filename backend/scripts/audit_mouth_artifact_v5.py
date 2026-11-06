"""Independently replay every retained source/crop/model-input binding, without the classifier."""

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


def check_binding(binding, frame, table, detector):
    if (
        binding["pts"] != frame.pts
        or binding["time_base"] != table.time_base
        or binding["time_s"] != table.seconds(frame.index)
    ):
        raise ValueError("Model input is bound to the wrong source time")
    if binding["decoded_rgb_sha256"] != hashlib.sha256(frame.rgb.tobytes()).hexdigest():
        raise ValueError("Model input is bound to different native pixels")
    faces = detect_faces(
        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=settings.face_min_size
    )
    transform = gend.align_transform(faces[0].landmarks) if faces else None
    if transform is None:
        raise ValueError("A retained model input has no independently detected face")
    matrix, side = transform
    if (
        side != binding["alignment_side"]
        or [[float(x).hex() for x in row] for row in matrix] != binding["alignment_matrix_hex"]
    ):
        raise ValueError("Input crop differs from the independently detected alignment")
    crop = gend.warp(frame.rgb, matrix, side)
    if hashlib.sha256(crop.tobytes()).hexdigest() != binding["native_crop_rgb_sha256"]:
        raise ValueError("Native aligned pixels differ from the classifier input")
    tensor = gend.to_batch([crop])[0].numpy()
    if (
        tensor.dtype != np.float32
        or hashlib.sha256(tensor.tobytes()).hexdigest() != binding["model_input_float32_sha256"]
    ):
        raise ValueError("Model preprocessing differs from the retained tensor hash")


def audit(manifest_path, baseline_path, receipt, output):
    from app.detectors.image.adapters.loaders import load_yunet
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    manifest = json.loads(manifest_path.read_text())
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    baseline_header, *baseline = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    if (
        header["manifest_sha256"] != digest(manifest_path)
        or baseline_header["manifest_sha256"] != digest(manifest_path)
        or header["baseline_receipt_sha256"] != digest(baseline_path)
        or [r["id"] for r in rows] != [r["id"] for r in manifest["items"]]
        or [r["id"] for r in baseline] != [r["id"] for r in rows]
    ):
        raise ValueError("Incomplete or different frozen comparison inputs")
    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    records = []
    for case, row, original in zip(manifest["items"], rows, baseline, strict=True):
        record = {"id": case["id"], "status": "failed", "verified_frames": 0}
        try:
            if row["status"] != "ok" or row.get("baseline_replay_parity") is not True:
                raise ValueError("This audited panel requires complete assessed outcomes")
            if row["base_windows"] != original["base_windows"]:
                raise ValueError("Baseline windows differ from the immutable comparison")
            path = manifest_path.parent / case["path"]
            if digest(path) != case["sha256"] or row["sha256"] != case["sha256"]:
                raise ValueError("Input file commitment changed")
            table = probe_frames(path)
            bindings = row["model_inputs"]
            indices = [b["frame_index"] for b in bindings]
            if not indices or indices != sorted(set(indices)):
                raise ValueError("Model input indices are empty, duplicated or unordered")
            scheduled = {
                i + k for i in original["sampling_centers"] if 0 < i < len(table) - 1 for k in (-1, 0, 1)
            }
            if set(indices) != scheduled:
                raise ValueError("This panel's candidate inputs differ from the baseline native schedule")
            for binding, frame in zip(bindings, decode_frames(path, table, indices), strict=True):
                if binding["frame_index"] != frame.index:
                    raise ValueError("Native frame decode order differs from the classifier input")
                check_binding(binding, frame, table, detector)
                record["verified_frames"] += 1
            record["status"] = "ok"
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    result = {
        "schema": "chipmunk.mouth-artifact-source-audit/5",
        "manifest_sha256": digest(manifest_path),
        "score_receipt_sha256": digest(receipt),
        "implementation_sha256": digest(Path(__file__)),
        "planned": len(records),
        "passed": sum(r["status"] == "ok" for r in records),
        "verified_frames": sum(r["verified_frames"] for r in records),
        "scope": "Local source/crop/tensor correspondence, not outside review or accuracy validation",
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
    p.add_argument("manifest", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.manifest, a.baseline, a.receipt, a.output)
