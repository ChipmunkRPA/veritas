"""Bounded engineering smoke on one consumed clip; no accuracy/admission claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import torch

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.liveportrait_mouth_benchmark import crop_from_mesh, load


def run(assets, old, output, device):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((old / "manifest.json").read_text())
    row = next(r for r in manifest["items"] if r["category"] == "original")
    path = old / row["path"]
    table = probe_frames(path)
    index = len(table.pts) // 2
    frame = next(decode_frames(path, table, [index]))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(assets / "permissions/face_landmarker.task"),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb))
    if len(result.face_landmarks) != 1:
        raise ValueError("Consumed smoke frame did not have one measurable face")
    crop, matrix, ratio = crop_from_mesh(frame.rgb, result.face_landmarks[0])
    cv2.imwrite(str(output / "source.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    renderer = load(assets, device)
    print(json.dumps({"stage": "loaded", "seconds": time.monotonic() - started}), flush=True)
    records = []
    for target in (ratio, 0.0, 0.3):
        started = time.monotonic()
        image = renderer.render(crop, ratio, target)
        name = f"render-{len(records)}.png"
        cv2.imwrite(str(output / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        records.append(
            {
                "target_ratio": target,
                "output": name,
                "seconds": time.monotonic() - started,
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    receipt = {
        "purpose": __doc__,
        "source_id": row["id"],
        "source_sha256": row["sha256"],
        "frame_index": index,
        "pts": table.pts[index],
        "time_base": table.time_base,
        "native_rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
        "affine": matrix.tolist(),
        "source_aperture": ratio,
        "device": device,
        "outputs": records,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets", type=Path)
    p.add_argument("consumed_source_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    a = p.parse_args()
    run(a.assets, a.consumed_source_dir, a.output, a.device)
