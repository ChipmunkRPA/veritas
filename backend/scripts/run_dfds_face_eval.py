"""Offline pinned face-crop trial with source-frame receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from prepare_video_upgrade_eval import sha
from torchvision import models, transforms

from app.core.config import settings
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.sampling import uniform_indices

WEIGHT_SHA = "444efd63dd3a312ee3515b608ac238c7b748c75a1e7d9cd3b9c19343694a2279"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SIZE = 260


def load(path: Path, device: str):
    if sha(path) != WEIGHT_SHA:
        raise ValueError("checkpoint does not match the reviewed release")
    weights = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    model = models.efficientnet_b2(weights=None)
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(1408, 2))
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


def score(path, model, yunet, device):
    table = probe_frames(path)
    # Use a deterministic whole-clip sample so evidence is comparable between the paired clips.
    transform = transforms.Compose([transforms.Resize((SIZE, SIZE), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    values, frames = [], []
    for frame in decode_frames(path, table, uniform_indices(len(table), 16)):
        faces = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=50)
        if not faces:
            continue
        face = faces[0]
        h, w = frame.rgb.shape[:2]
        x0, y0 = max(0, face.x - 20), max(0, face.y - 20)
        x1, y1 = min(w, face.x + face.w + 20), min(h, face.y + face.h + 20)
        values.append(transform(Image.fromarray(frame.rgb[y0:y1, x0:x1])))
        frames.append({"frame_index": frame.index, "pts": frame.pts, "time_s": table.seconds(frame.index),
                       "bbox": [x0, y0, x1 - x0, y1 - y0]})
    if not frames:
        return {"status": "not_applicable", "prob": None, "frames": []}
    with torch.inference_mode():
        scores = torch.cat([model(batch.to(device)).softmax(1)[:, 0].cpu() for batch in torch.stack(values).split(8)])
    for frame, raw in zip(frames, scores.tolist(), strict=True):
        frame["raw"] = raw
    return {"status": "ok", "prob": float(np.mean(scores.numpy())), "time_base": table.time_base, "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = load(args.weights, args.device)
    yunet_path = settings.weights_dir / "face_detection_yunet_2023mar.onnx"
    if sha(yunet_path) != YUNET_SHA:
        raise ValueError("unexpected face proposal weights")
    yunet = load_yunet(yunet_path)
    with args.output.open("x") as out:
        out.write(json.dumps({"run": {"manifest_sha256": sha(args.manifest), "weights_sha256": WEIGHT_SHA,
                                      "script_sha256": sha(Path(__file__)), "device": args.device}}) + "\n")
        for item in json.loads(args.manifest.read_text())["items"]:
            row = {key: item[key] for key in ("id", "group", "label", "split", "family")}
            if item["status"] == "ready":
                info = item["matched"]
                path = args.manifest.parent / info["path"]
                if sha(path) != info["sha256"]:
                    raise ValueError(f"source bytes changed: {item['id']}")
                row.update(sha256=info["sha256"], **score(path, model, yunet, args.device))
            else:
                row.update(status=item["status"], prob=None, frames=[])
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            print(row["id"], row["status"], flush=True)


if __name__ == "__main__":
    main()
