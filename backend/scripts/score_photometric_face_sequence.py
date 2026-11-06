"""Compare raw and illumination-normalized consecutive-crop evidence offline.

All crops, schedules, face detections and DINOv2 weights are shared. Only the
crop's robust brightness/contrast normalization differs between the two arms.
No result here enters the customer confidence calculation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402
from app.detectors.image.faces import detect_faces  # noqa: E402
from app.detectors.image.gend import align_transform, warp  # noqa: E402
from app.detectors.video.decode import FrameTable, decode_frames  # noqa: E402
from app.engine.fetchers import ensure_verified  # noqa: E402
from app.engine.models import get_model_manager  # noqa: E402
from app.engine.registry import get_registry  # noqa: E402
from app.evidence.face_intervals import bind_tracks  # noqa: E402
from scripts.prepare_video_upgrade_eval import sha  # noqa: E402
from scripts.score_face_sequence_candidates import (  # noqa: E402
    DINO,
    SIZE,
    YUNET,
    patch_features,
    regions,
    sequence_windows,
)

NORMALIZATION = {"luminance": "RGB channel mean", "center_percentile": 50.,
                 "spread_percentiles": [25., 75.], "minimum_spread": 12.,
                 "output_center": 128., "output_spread": 40., "rounding": "nearest uint8, clip to [0,255]"}


def normalize(rgb: np.ndarray) -> np.ndarray | None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Photometric comparison requires uint8 RGB face crops")
    values = rgb.astype(np.float32)
    luminance = values.mean(axis=2)
    low, center, high = np.percentile(luminance, [25, 50, 75])
    spread = float(high-low)
    if spread < NORMALIZATION["minimum_spread"]:
        return None  # Do not amplify low-contrast noise into a purported identity change.
    normalized = (values-float(center)) * NORMALIZATION["output_spread"] / spread + NORMALIZATION["output_center"]
    return np.rint(np.clip(normalized, 0, 255)).astype(np.uint8)


def score(row: dict, root: Path, dino, yunet) -> dict:
    path = root / row["path"]
    if sha(path) != row["sha256"]:
        raise ValueError("Video bytes changed after source preparation")
    table = FrameTable.from_json(row["table"])
    centers = {p["frame_index"] for p in row["points"] if 0 < p["frame_index"] < len(table)-1}
    scheduled = {i+k for i in centers for k in (-1, 0, 1)}
    raw, normalized, faces = {}, {}, {}
    for frame in decode_frames(path, table, sorted(scheduled)):
        detected = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet,
                                min_size=settings.face_min_size)
        if not detected:
            continue
        face = detected[0]
        transform = align_transform(face.landmarks)
        if transform is None:
            continue
        rgb = cv2.resize(warp(frame.rgb, *transform), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        faces[frame.index], raw[frame.index] = face, rgb
        transformed = normalize(rgb)
        if transformed is not None:
            normalized[frame.index] = transformed
    observed = regions(faces, table)
    raw_tracks = bind_tracks([{"frame_index": i, "score": 0.} for i in raw], [observed, *row["exhibits"]], table)
    normal_tracks = bind_tracks([{"frame_index": i, "score": 0.} for i in normalized],
                               [observed, *row["exhibits"]], table)
    return {"raw_windows": sequence_windows(centers, raw_tracks, patch_features(raw, dino), table),
            "normalized_windows": sequence_windows(centers, normal_tracks, patch_features(normalized, dino), table),
            "scheduled_frames": len(scheduled), "face_frames": len(raw), "normalized_frames": len(normalized),
            "low_contrast_frames": len(raw)-len(normalized)}


def run(manifest: Path, baseline: Path, output: Path, device: str, lock: Path | None = None) -> None:
    body = json.loads(manifest.read_text())
    with baseline.open() as stream:
        header = json.loads(next(stream))
        scores = [json.loads(line) for line in stream]
    expected = [r for r in body["items"] if r["split"] == header["split"]]
    if header["manifest_sha256"] != sha(manifest) or len(scores) != len(expected):
        raise ValueError("Incomplete current-check input receipt")
    if any(any(row.get(k) != value for k, value in item.items()) for item, row in zip(expected, scores, strict=True)):
        raise ValueError("Current-check row differs from the prepared source")
    # Without a decision lock, only previously consumed v5 sources are eligible.
    if lock is None:
        frozen_sources = Path(__file__).resolve().parents[1] / "data/independent-evaluation/video-v5/intervals/manifest.json"
        if manifest.resolve() != frozen_sources.resolve():
            raise ValueError("New-source candidate inference requires frozen development rules")
    else:
        selected = json.loads(lock.read_text())
        if (selected["confirmation_manifest_sha256"] != sha(manifest)
                or selected["candidate_runner_sha256"] != sha(Path(__file__))
                or selected["feature_runtime_sha256"] != sha(Path(__file__).with_name("score_face_sequence_candidates.py"))
                or set(selected["development_groups"]) & {r["group"] for r in expected}):
            raise ValueError("Photometric confirmation differs from the frozen method/sources")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    registry = get_registry()
    pins = {}
    for model_id in (YUNET, DINO):
        entry = registry.manifest.model(model_id)
        if entry is None or entry.licence.class_ != "commercial":
            raise ValueError(f"Missing commercial permission record: {model_id}")
        ensure_verified(entry)
        pins[model_id] = [{"path": f.path, "sha256": f.sha256} for f in entry.source.files]
    if lock is not None and selected["weights"] != pins:
        raise ValueError("Photometric confirmation weights differ from development")
    manager = get_model_manager()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (manager.lease((YUNET,), prefer="cpu") as cpu, manager.lease((DINO,), prefer=device) as gpu,
          output.open("x") as out):
        record = {"schema": "chipmunk.photometric-sequence/1", "manifest_sha256": sha(manifest),
                  "baseline_sha256": sha(baseline), "split": header["split"], "runner_sha256": sha(Path(__file__)),
                  "feature_runtime_sha256": sha(Path(__file__).with_name("score_face_sequence_candidates.py")),
                  "normalization": NORMALIZATION, "device": device, "weights": pins,
                  "role": "consumed development" if lock is None else "confirmation",
                  "lock_sha256": sha(lock) if lock else None}
        out.write(json.dumps(record) + "\n")
        out.flush()
        for row in scores:
            value = {"id": row["id"], "source_sha256": row.get("sha256"), "status": "not_applicable",
                     "reason": "No scored face input in the current sample plan"}
            started = time.perf_counter()
            if row["status"] == "ready" and row.get("points"):
                # An integrity failure terminates the trial, rather than becoming a detector miss.
                if sha(manifest.parent / row["path"]) != row["sha256"]:
                    raise ValueError("Prepared video bytes changed")
                try:
                    value.update(score(row, manifest.parent, gpu.get(DINO).obj, cpu.get(YUNET).obj), status="ok")
                    value.pop("reason", None)
                except Exception as error:
                    value.update(status="failed", reason=f"{type(error).__name__}: {error}")
            value["runtime_s"] = time.perf_counter()-started
            out.write(json.dumps(value, allow_nan=False) + "\n")
            out.flush()
            print(row["id"], value["status"], len(value.get("normalized_windows", [])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.baseline, args.output, args.device, args.lock)
