"""Community Forensics ViT-S/16 at 384 px (Park and Owens, CVPR 2025): the inference path, written from the upstream code.

- **Model.** timm `vit_small_patch16_384` with a one-logit head, the architecture of upstream `models.ViTClassifier`
  (github.com/JeongsooP/Community-Forensics, MIT, commit ee5b71d4). The pinned safetensors keys carry a `vit.` prefix.
  No upstream code is vendored.
- **Preprocessing (ML N2).** Native-resolution 384 px crops, never resampled: the centre and the four corners when the
  short side is at least 384 px, duplicates dropped (`native_crops`); 8-bit RGB scaled to [0, 1] and normalised with the
  ImageNet mean and std, as upstream `dataloader.get_transform`. Upstream evaluation instead resizes the short side to
  440 px and takes the centre 384 px crop; the slow test reproduces that path against the upstream notebook's outputs.
- **Aggregation.** Mean logit over the crops (N2); the adapter keeps the maximum for the explanation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.detectors.image import native_crops

CROP = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model() -> torch.nn.Module:
    import timm

    return timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)


def load_model(path: Path, device: Any) -> torch.nn.Module:
    """Load the pinned `model.safetensors` strictly (every key present, none left over)."""
    from safetensors.torch import load_file

    state = {key.removeprefix("vit."): value for key, value in load_file(str(path)).items()}
    model = build_model()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def crop_origins(width: int, height: int, crop: int = CROP) -> list[tuple[int, int, str]]:
    return native_crops.crop_origins(width, height, crop)


def to_batch(pixels: np.ndarray, origins: list[tuple[int, int, str]], crop: int = CROP) -> torch.Tensor:
    return native_crops.to_batch(pixels, origins, crop, MEAN, STD)


def logits(model: Any, batch: torch.Tensor, device: Any) -> list[float]:
    with torch.inference_mode():
        out = model(batch.to(device))
    return [float(v) for v in out.detach().to("cpu", torch.float32).reshape(-1)]
