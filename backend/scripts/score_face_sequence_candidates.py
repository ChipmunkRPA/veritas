"""Run independent denser frame scoring and consecutive face-crop comparisons."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.detectors.image.dfdc import DFDCClassifier
from app.detectors.image.faces import detect_faces
from app.detectors.image.gend import align_transform, warp
from app.detectors.video.decode import FrameTable, decode_frames
from app.detectors.video.tracking import DinoEmbedder
from app.engine.fetchers import ensure_verified
from app.engine.models import get_model_manager
from app.engine.registry import get_registry
from app.evidence.face_intervals import bind_tracks
from scripts.prepare_video_upgrade_eval import sha

YUNET = "yunet.2023mar"
DINO = "dinov2.small"
B7 = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
SIZE = 112
INNER = np.zeros((8, 8), dtype=bool)
INNER[2:6, 2:6] = True
RING = ~INNER.flatten()
CORE = INNER.flatten()


def extra_frames(points: list[dict], table: FrameTable) -> set[int]:
    """Refine three separated moments. The same schedule runs on all inputs."""
    centers: list[float] = []
    for point in sorted(points, key=lambda p: (-p["score"], p["frame_index"])):
        center = table.seconds(point["frame_index"])
        if all(abs(center - chosen) >= 1.0 for chosen in centers):
            centers.append(center)
        if len(centers) == 3:
            break
    indices = {i for i, stamp in enumerate(table.times()) if i % 2 == 0 and any(abs(stamp - t) <= .55 for t in centers)}
    return indices.difference(point["frame_index"] for point in points)


def regions(faces: dict, table: FrameTable) -> dict:
    return {"kind": "regions", "created_by": "vid.faces.yunet", "regions": [
        {"region": {"kind": "bbox", "frame_pts": round(table.seconds(i) * 1000),
                    "bbox": {"x": face.x, "y": face.y, "w": face.w, "h": face.h}}}
        for i, face in faces.items()]}


def patch_features(crops: dict[int, np.ndarray], dino: DinoEmbedder) -> dict[int, np.ndarray]:
    keys = sorted(crops)
    features = {}
    dtype = next(dino.model.parameters()).dtype
    for start in range(0, len(keys), 12):
        indices = keys[start:start+12]
        pixels = np.stack([crops[i] for i in indices]).astype(np.float32) / 255
        values = torch.from_numpy(((pixels - dino.MEAN) / dino.STD).transpose(0, 3, 1, 2).copy()).to(dino.device, dtype)
        with torch.inference_mode():
            tokens = dino.model(pixel_values=values).last_hidden_state[:, 1:].float()
            tokens = torch.nn.functional.normalize(tokens, dim=-1).cpu().numpy()
        if tokens.shape[1] != 64:
            raise ValueError("Unexpected face-feature grid")
        features.update(zip(indices, tokens, strict=True))
    return features


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - np.sum(a * b, axis=-1) / np.maximum(scale, 1e-8)


def sequence_windows(centers: set[int], tracks: dict, patches: dict, table: FrameTable) -> list[dict]:
    windows = []
    for track_id, samples in tracks.items():
        index = {sample.frame_index for sample in samples}
        blocks = []
        for i in sorted(centers):
            group = (i-1, i, i+1)
            if all(k in index and k in patches for k in group) and table.seconds(i+1) - table.seconds(i-1) <= .13:
                blocks.append((i, np.mean([patches[k] for k in group], axis=0)))
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        values = np.stack([value for _, value in blocks])
        for start in range(2, len(blocks)-4):
            for end in range(start+3, len(blocks)-1):
                span = times[end-1] - times[start]
                if span > 2.6:
                    break
                if span < .52 or np.max(np.diff(times[start-2:end+2])) > .55:
                    continue
                left, core, right = (np.mean(part, axis=0) for part in (values[start-2:start], values[start:end], values[end:end+2]))
                left_shift, right_shift, normal = distance(core, left), distance(core, right), distance(left, right)
                inside = min(float(left_shift[CORE].mean()), float(right_shift[CORE].mean()))
                outside = max(float(left_shift[RING].mean()), float(right_shift[RING].mean()))
                response = inside - .60 * outside - .35 * float(normal[CORE].mean())
                windows.append({"track_id": track_id, "start_s": float(times[start]), "end_s": float(times[end-1]),
                                "response": response, "inner_change": inside, "outer_change": outside,
                                "centers": [i for i, _ in blocks[start-2:end+2]]})
    return sorted(windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))[:12]


def score(row: dict, root: Path, clf: DFDCClassifier, dino: DinoEmbedder, yunet) -> dict:
    path = root / row["path"]
    if sha(path) != row["sha256"]:
        raise ValueError(f"Source video bytes changed: {row['id']}")
    table = FrameTable.from_json(row["table"])
    points = row["points"]
    # The sequence arm uses baseline sample positions and native neighbors; dense scores never enter it.
    centers = {p["frame_index"] for p in points if 0 < p["frame_index"] < len(table)-1}
    sequence = {i + k for i in centers for k in (-1, 0, 1)}
    extras = extra_frames(points, table)
    crops, faces, inputs = {}, {}, []
    for frame in decode_frames(path, table, sorted(sequence | extras)):
        detected = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=settings.face_min_size)
        if not detected:
            continue
        i, face = frame.index, detected[0]
        faces[i] = face
        if i in extras:
            inputs.append((i, frame.rgb, face))
        if i in sequence and (aligned := align_transform(face.landmarks)) is not None:
            crops[i] = cv2.resize(warp(frame.rgb, *aligned), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    newer_regions = regions(faces, table)
    started = time.perf_counter()
    responses = clf.predict([(rgb, face) for _, rgb, face in inputs])
    extra_points = [{"frame_index": item[0], "score": value} for item, value in zip(inputs, responses, strict=True)]
    dense_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    sequence_tracks = bind_tracks([{"frame_index": i, "score": 0.0} for i in crops],
                                  [newer_regions, *row["exhibits"]], table)
    windows = sequence_windows(centers, sequence_tracks, patch_features(crops, dino), table)
    return {"dense_points": extra_points, "extra_regions": newer_regions, "scheduled_extra_frames": len(extras),
            "sequence_frames": len(crops), "sequence_windows": windows,
            "dense_inference_s": dense_elapsed, "sequence_inference_s": time.perf_counter() - started}


def receipt_line(stream, *, follow: bool) -> dict:
    deadline = time.monotonic() + 600
    offset = stream.tell()
    while True:
        line = stream.readline()
        if line.endswith("\n"):
            return json.loads(line)
        if not follow or time.monotonic() >= deadline:
            raise ValueError("Baseline receipt is incomplete")
        time.sleep(.5)
        stream.seek(offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="cpu")
    parser.add_argument("--follow", action="store_true", help="Read completed rows while the baseline is still scoring")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text())["items"]
    registry = get_registry()
    for model_id in (YUNET, DINO, *B7):
        model = registry.manifest.model(model_id)
        if model is None or model.licence.class_ != "commercial":
            raise ValueError(f"Missing commercial checkpoint provenance: {model_id}")
        ensure_verified(model)
    manager = get_model_manager()
    with (manager.lease((YUNET,), prefer="cpu") as cpu,
          manager.lease((*B7, DINO), prefer=args.device) as gpu,
          args.baseline.open() as baseline,
          tempfile.TemporaryFile(mode="w+") as pending):
        header = receipt_line(baseline, follow=args.follow)
        if header["manifest_sha256"] != sha(args.manifest):
            raise ValueError("Baseline receipt uses a different input manifest")
        expected = [item for item in manifest if item["split"] == header["split"]]
        clf = DFDCClassifier(members=[gpu.get(model).obj for model in B7], device=gpu.device)
        for item in expected:
            row = receipt_line(baseline, follow=args.follow)
            if any(row.get(key) != value for key, value in item.items()):
                raise ValueError("Baseline row differs from the expected input")
            started = time.perf_counter()
            value = (score(row, args.manifest.parent, clf, gpu.get(DINO).obj, cpu.get(YUNET).obj)
                     if row["status"] == "ready" and row.get("points") else {})
            pending.write(json.dumps({"id": row["id"], "source_sha256": row.get("sha256"),
                                      "seconds": time.perf_counter() - started, **value}, allow_nan=False) + "\n")
            print(row["id"], len(value.get("dense_points", [])), len(value.get("sequence_windows", [])), flush=True)
        if baseline.readline():
            raise ValueError("Unexpected extra baseline rows")
        pending.seek(0)
        with args.output.open("x") as out:
            out.write(json.dumps({"manifest_sha256": sha(args.manifest), "baseline_sha256": sha(args.baseline),
                                  "split": header["split"], "runner_sha256": sha(Path(__file__)), "device": gpu.device}) + "\n")
            shutil.copyfileobj(pending, out)


if __name__ == "__main__":
    main()
