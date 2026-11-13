# SPDX-License-Identifier: Apache-2.0
"""Run the reviewed whole-image cue in the separate local model environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from scripts import audit_image_smaug_runtime_v7 as assets
from scripts.image_smaug_views_v7 import views

ROOT = Path(__file__).resolve().parents[2]
POINT = ROOT / "docs/evaluation-panels/image-v7-smaug-genuine-operating-point.json"
POINT_SHA256 = "c97b5e02b26a2c8763c9f1c8cb543f5ab63780b65f0f554e0450a42ca470d4bc"
PACKAGES = {"torch": "2.13.0", "Pillow": "12.3.0", "numpy": "2.2.6", "transformers": "5.22.2",
            "sentence-transformers": "6.2.0", "huggingface-hub": "1.32.0", "tokenizers": "0.23.1"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(source, model_dir, output):
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if (sha(assets.ASSET_RECEIPT) != assets.ASSET_SHA or sha(assets.PROCESSOR_RECEIPT) != assets.PROCESSOR_SHA
            or sha(POINT) != POINT_SHA256 or any(version(package) != expected for package, expected in PACKAGES.items())):
        raise ValueError("The reviewed local image runtime is unavailable")
    reviewed = json.loads(assets.ASSET_RECEIPT.read_bytes())["files"]
    reviewed.append(json.loads(assets.PROCESSOR_RECEIPT.read_bytes())["added_asset"])
    if any(sha(model_dir / item["name"]) != item["sha256"] for item in reviewed):
        raise ValueError("The reviewed local model assets are unavailable")
    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False,
                                config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    model.requires_grad_(False).eval()
    assets.check_tensor_file(model[0].auto_model, model_dir / "model.safetensors", 438760472)
    assets.check_tensor_file(model[3], model_dir / "3_Dense/model.safetensors", 769)
    processor = model[0].processor.image_processor
    processor.max_soft_tokens = processor.image_seq_length = 140
    with Image.open(source, formats=["JPEG", "PNG"]) as image:
        crops, traces = views(image, 3)
    with torch.inference_mode():
        responses = model.encode([{"image": crop} for crop in crops], batch_size=3, convert_to_tensor=True,
                                 show_progress_bar=False).float().reshape(-1)
    if responses.shape != (3,) or not torch.isfinite(responses).all():
        raise ValueError("The whole-image response is incomplete")
    point = json.loads(POINT.read_bytes())["gate"]["raw_mean_logit"]
    flagged = float(responses.mean()) > point
    regions = []
    if flagged:
        for number, (crop, trace) in enumerate(zip(crops, traces, strict=True), 1):
            first, second = trace["source_edge_bounds_rational"][:2], trace["source_edge_bounds_rational"][2:]
            x, y = (n // d for n, d in first)
            right, bottom = ((n + d - 1) // d for n, d in second)
            image_path = output / f"crop-{number}.png"
            crop.save(image_path, "PNG")
            regions.append({"region": number, "bounds": [x, y, right - x, bottom - y], "sha256": sha(image_path)})
    with (output / "result.json").open("x") as stream:
        json.dump({"source_sha256": sha(source), "flagged": flagged, "checked_views": len(crops), "regions": regions}, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect(args.source, args.model_dir, args.output)
