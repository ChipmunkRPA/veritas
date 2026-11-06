"""Run the published temporal face checkpoint on the locked RTFS clips."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from prepare_face_swap_upgrade_eval import POLICY
from prepare_video_upgrade_eval import sha

MODEL_FILES = {
    "config.json": "3f022d107fd4b4d2e848189696281f7e7cec54447016bd9632a5b91be2879163",
    "preprocessor_config.json": "4bc9b255ff212041bc28127d4fa0ae1630abca52bb2d882f3e4c0b9e5b763f06",
    "model.safetensors": "293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29",
}


def uniform_frames(path: Path, count: int) -> tuple[list[np.ndarray], list[int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < count:
            raise ValueError(f"only {total} frames")
        indices = np.linspace(0, total - 1, count, dtype=int).tolist()
        required = set(indices)
        frames = []
        for index in range(total):
            decoded, bgr = capture.read()
            if not decoded:
                raise ValueError(f"could not decode frame {index}")
            if index in required:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return frames, indices
    finally:
        capture.release()


def run(manifest: Path, model_dir: Path, output: Path, device: str) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    for name, expected in MODEL_FILES.items():
        if sha(model_dir / name) != expected:
            raise ValueError(f"unexpected model file: {name}")
    source = json.loads(manifest.read_text())
    if source["policy"] != POLICY:
        raise ValueError("unexpected face-swap manifest policy")
    config = VideoMAEConfig.from_pretrained(model_dir, local_files_only=True)
    if config.id2label != {0: "real", 1: "fake"}:
        raise ValueError("unexpected checkpoint label order")
    # Transformers v5.2 passed q_bias/v_bias and a zero key bias directly
    # to F.linear. The installed version reads them from each linear layer.
    model = VideoMAEForVideoClassification(config)
    state = load_file(model_dir / "model.safetensors")
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention"
        state[f"{prefix}.query.bias"] = state.pop(f"{prefix}.q_bias")
        state[f"{prefix}.value.bias"] = state.pop(f"{prefix}.v_bias")
        state[f"{prefix}.key.bias"] = torch.zeros_like(state[f"{prefix}.value.bias"])
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    processor = VideoMAEImageProcessor.from_pretrained(model_dir, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(json.dumps({"run": {
            "system": "candidate-videomae",
            "manifest_sha256": sha(manifest),
            "scorer_sha256": sha(Path(__file__)),
            "model_files": MODEL_FILES,
            "attention_bias_mapping": "query=q_bias,key=zeros_like(v_bias),value=v_bias",
            "variants": ["matched"],
            "torch": torch.__version__,
            "device": device,
        }}) + "\n")
        file.flush()
        for item in source["items"]:
            row = {key: item[key] for key in ("id", "group", "label", "split", "family")}
            meta = item.get("matched", {})
            row.update(variant="matched", sha256=meta.get("sha256"))
            if item["status"] != "ready":
                row.update(status="unavailable", reason=item.get("reason"))
            else:
                path = manifest.parent / meta["path"]
                if sha(path) != row["sha256"]:
                    raise ValueError(f"source bytes changed: {item['id']}")
                started = time.perf_counter()
                try:
                    frames, indices = uniform_frames(path, model.config.num_frames)
                    with torch.inference_mode():
                        tensor = processor(frames, return_tensors="pt")["pixel_values"].to(device)
                        logits = model(pixel_values=tensor).logits[0]
                    logit = float((logits[1] - logits[0]).item())
                    if not math.isfinite(logit):
                        raise ValueError("non-finite detector response")
                    row.update(
                        status="ok", logit=logit, score=float(torch.sigmoid(logits[1] - logits[0]).item()),
                        frame_indices=indices, runtime_s=round(time.perf_counter() - started, 4),
                    )
                except Exception as exc:
                    row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            file.write(json.dumps(row, allow_nan=False) + "\n")
            file.flush()
            print(f"{item['id']}: {row['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model_dir", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    run(args.manifest, args.model_dir, args.output, args.device)
