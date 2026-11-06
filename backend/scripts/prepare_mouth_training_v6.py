"""Prepare source-separated native mouth-artifact fitting frames; no customer detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.liveportrait_mouth_benchmark import crop_from_mesh, digest, load
from scripts.prepare_mouth_edit_v4 import composite, mouth_alpha

POLICY = "chipmunk.mouth-trained-frame-preparation/6"
CONDITIONS = ("original", "local-light", "local-resample", "ai-redraw", "ai-retime")
POSITIONS = (0, 3, 5, 8, 11, 14, 16, 19)


def selected_indices(frames):
    if not 50 <= frames <= 100:
        raise ValueError("Training sources require 50–100 complete source frames")
    start = (frames - 20) // 2
    return start, tuple(start + i for i in POSITIONS)


def plan(source_root, previous_plan, output):
    old = json.loads((source_root / "manifest.json").read_text())
    previous = json.loads(previous_plan.read_text())
    excluded = {r["group"] for r in previous["sources"]}
    excluded_hashes = {r["sha256"] for r in previous["sources"]}
    selected = []
    for row in old["items"]:
        if (
            row["category"] != "original"
            or not row["group"].startswith("CREMA-D:")
            or row["group"] in excluded
        ):
            continue
        path = source_root / row["path"]
        if digest(path) != row["sha256"] or row["sha256"] in excluded_hashes:
            raise ValueError("A fitting source changed or overlaps development")
        table = probe_frames(path)
        common = min(100, len(table), len(decode_audio(path, source_sha256=row["sha256"]).pcm) // 640)
        start, indices = selected_indices(common)
        selected.append({**row, "frames": common, "edit_start": start, "selected_frames": indices})
    selected.sort(key=lambda r: hashlib.sha256(("chipmunk.mouth-trained/6:" + r["id"]).encode()).hexdigest())
    if (
        len(selected) != 20
        or len({r["group"] for r in selected}) != 20
        or len({r["sha256"] for r in selected}) != 20
    ):
        raise ValueError("The fixed plan requires all 20 remaining consumed actor groups")
    for i, row in enumerate(selected):
        row["role"] = "fit" if i < 16 else "point"
    root = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/prepare_mouth_training_v6.py",
        "backend/scripts/prepare_mouth_edit_v4.py",
        "backend/scripts/liveportrait_mouth_benchmark.py",
        "docs/mouth-trained-v6-development.md",
    ]
    body = {
        "policy": POLICY,
        "role": "Consumed development training; no fresh confirmation",
        "source_root": str(source_root.resolve()),
        "source_manifest_sha256": digest(source_root / "manifest.json"),
        "previous_plan_sha256": digest(previous_plan),
        "excluded_development_groups": sorted(excluded),
        "sources": selected,
        "conditions": CONDITIONS,
        "planned_frames": 800,
        "implementation_sha256": {p: digest(root / p) for p in pins},
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"plan_sha256": digest(output), "fit_actors": 16, "point_actors": 4, "planned_frames": 800}
        )
    )


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise ValueError("Could not encode a lossless training frame")
    replay = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    if not np.array_equal(replay, rgb):
        raise ValueError("PNG did not preserve exact training pixels")
    return {"sha256": digest(path), "native_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}


def build(plan_path, assets, output):
    import mediapipe as mp
    import torch

    recipe = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[2]
    if any(digest(root / p) != sha for p, sha in recipe["implementation_sha256"].items()):
        raise ValueError("Frozen generation implementation changed")
    source_root = Path(recipe["source_root"])
    if digest(source_root / "manifest.json") != recipe["source_manifest_sha256"]:
        raise ValueError("Consumed source manifest changed")
    if not torch.backends.mps.is_available():
        raise ValueError("Frozen generation uses MPS; no silent device substitution")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    renderer = load(assets, "mps")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(assets / "permissions/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
    )
    records = []
    for source in recipe["sources"]:
        started = time.monotonic()
        folder = output / source["id"]
        folder.mkdir()
        planned = [
            {
                "id": f"{source['id']}:{i}:{c}",
                "group": source["group"],
                "role": source["role"],
                "condition": c,
                "label": int(c.startswith("ai-")),
                "source_sha256": source["sha256"],
                "source_url": source["download_url"],
                "source_license": source["source_license"],
                "frame_index": i,
                "status": "unavailable",
            }
            for i in source["selected_frames"]
            for c in CONDITIONS
        ]
        try:
            path = source_root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError("Selected original bytes changed")
            table = probe_frames(path)
            required = sorted(set(source["selected_frames"]) | {i - 8 for i in source["selected_frames"]})
            frames = {f.index: f.rgb for f in decode_frames(path, table, required)}
            if set(frames) != set(required):
                raise ValueError("A planned native training frame is missing")
            mesh = {}
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                for i in required:
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[i]),
                        round(table.seconds(i) * 1000),
                    )
                    if len(result.face_landmarks) == 1:
                        mesh[i] = result.face_landmarks[0]
            if set(mesh) != set(required):
                raise ValueError("A planned frame lacks unique-face geometry")
            geometry = {i: crop_from_mesh(frames[i], mesh[i]) for i in required}
            for i in source["selected_frames"]:
                crop, matrix, ratio = geometry[i]
                h, w = frames[i].shape[:2]
                inverse = cv2.invertAffineTransform(matrix * 2)
                alpha = mouth_alpha(frames[i].shape, mesh[i])
                offset = i - source["edit_start"]
                opacity = min(1.0, (offset + 1) / 2, (20 - offset) / 2)
                variants = {
                    "original": frames[i],
                    "local-light": (frames[i].astype(np.float32) * 1.12)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8),
                    "local-resample": cv2.warpAffine(
                        cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR),
                        inverse,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                    ),
                }
                for c, target in (("ai-redraw", ratio), ("ai-retime", geometry[i - 8][2])):
                    variants[c] = cv2.warpAffine(
                        renderer.render(crop, ratio, target), inverse, (w, h), flags=cv2.INTER_LINEAR
                    )
                for row in [r for r in planned if r["frame_index"] == i]:
                    c = row["condition"]
                    native = (
                        frames[i] if c == "original" else composite(frames[i], variants[c], alpha, opacity)
                    )
                    changed = np.any(native != frames[i], axis=2)
                    if c != "original" and not changed.any():
                        raise ValueError("A planned edited frame has no changed pixels")
                    image_path = folder / f"{i:04d}-{c}.png"
                    image_commit = save_rgb(image_path, native)
                    mask_path = folder / f"{i:04d}-{c}-mask.png"
                    if not cv2.imwrite(str(mask_path), changed.astype(np.uint8) * 255):
                        raise ValueError("Could not retain a training mask")
                    row.update(
                        status="ready",
                        path=str(image_path.relative_to(output)),
                        **image_commit,
                        pts=table.pts[i],
                        time_base=table.time_base,
                        original_rgb_sha256=hashlib.sha256(frames[i].tobytes()).hexdigest(),
                        mask_path=str(mask_path.relative_to(output)),
                        mask_sha256=digest(mask_path),
                        changed_pixels=int(changed.sum()),
                        outside_support_exact=bool(np.array_equal(native[alpha == 0], frames[i][alpha == 0])),
                    )
        except Exception as error:
            for row in planned:
                if row["status"] != "ready":
                    row["reason"] = f"{type(error).__name__}: {error}"
        records.extend(planned)
        with (output / "partial-receipts.jsonl").open("a") as stream:
            for row in planned:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "role": source["role"],
                    "ready": sum(r["status"] == "ready" for r in planned),
                    "planned": len(planned),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    body = {
        "policy": POLICY,
        "plan_sha256": digest(plan_path),
        "renderer_asset_receipt_sha256": digest(assets / "permissions/liveportrait-verified-assets.json"),
        "items": records,
        "role": "Consumed development training; no fresh confirmation",
        "deployment_eligible": False,
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned": len(records),
                "ready": sum(r["status"] == "ready" for r in records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "build"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--previous-plan", type=Path)
    p.add_argument("--assets", type=Path)
    a = p.parse_args()
    if a.stage == "plan":
        if not a.previous_plan:
            p.error("plan requires --previous-plan")
        plan(a.input, a.previous_plan, a.output)
    else:
        if not a.assets:
            p.error("build requires --assets")
        build(a.input, a.assets, a.output)
