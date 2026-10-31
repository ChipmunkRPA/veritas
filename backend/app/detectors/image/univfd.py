"""UnivFD (Ojha, Li, Lee — CVPR 2023): a linear probe on frozen CLIP ViT-L/14 image features.

Generalises across generator families (GANs, diffusion) because it does not learn
generator-specific artifacts. Weights: WisconsinAIVision/UniversalFakeDetect fc_weights.pth.

Evaluation follows the paper: 224x224 crops at *native* resolution (no resizing — resizing
destroys the low-level statistics the probe relies on). We average five crops (centre + corners)
for stability. Images smaller than 224px are upscaled and flagged as out-of-distribution.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

from app.core.config import settings
from app.core.device import DEVICE
from app.detectors.common.download import fetch
from app.detectors.common.registry import registry

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CROP = 224
CROPS_PER_SCALE = 5
MAX_SCALES = 2
FRAME_BATCH = 16


def native_crops(im: Image.Image, n: int = CROPS_PER_SCALE) -> tuple[list[Image.Image], bool]:
    """Return up to n 224px crops at native resolution and whether upscaling was needed."""
    im = im.convert("RGB")
    w, h = im.size
    upscaled = False
    if min(w, h) < CROP:
        scale = CROP / min(w, h)
        im = im.resize((max(CROP, round(w * scale)), max(CROP, round(h * scale))), Image.BICUBIC)
        w, h = im.size
        upscaled = True
    pts = [((w - CROP) // 2, (h - CROP) // 2), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    # de-duplicate (small images yield identical crops)
    seen, crops = set(), []
    for x, y in pts[:n]:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        crops.append(im.crop((x, y, x + CROP, y + CROP)))
    return crops, upscaled


def load_probe(path: Path, device: Any) -> torch.nn.Linear:
    """The UnivFD linear probe (768 → 1) from `fc_weights.pth`."""
    sd = torch.load(path, map_location="cpu")
    fc = torch.nn.Linear(768, 1)
    fc.load_state_dict({"weight": sd["weight"], "bias": sd["bias"]})
    return fc.to(device).eval()


class UnivFD:
    """v1 loads both parts itself; v2 adapters pass the leased CLIP model, probe and device."""

    def __init__(self, model: Any = None, fc: torch.nn.Module | None = None, device: Any = None) -> None:
        self.device = DEVICE if device is None else device
        if model is None:
            # QuickGELU, the activation the OpenAI weights were trained with (B-4a; open_clip's "ViT-L-14" uses GELU)
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
            model = model.to(self.device).eval()
        self.model = model
        self.fc = fc if fc is not None else load_probe(fetch(settings.univfd_fc_url, "univfd_fc_weights.pth"), self.device)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    @torch.inference_mode()
    def _scores(self, crops: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.norm(c) for c in crops]).to(self.device)
        feats = self.model.encode_image(x).float()
        return torch.sigmoid(self.fc(feats)).squeeze(-1).cpu().numpy()

    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Native-resolution crops; for large photos (> 1024 px) also crops from a 1024 px copy, averaged.

        224 px native crops of a 10 MP photograph are near-featureless texture patches and the probe misreads
        them; the moderate downscale keeps sensor statistics while restoring scene content (field test 2026-08).
        """
        out = []
        for im in images:
            im = ImageOps.exif_transpose(im)
            crops, up = native_crops(im)
            scales = 1
            if min(im.size) > 1024:
                r = 1024 / min(im.size)
                small = im.resize((max(CROP, round(im.width * r)), max(CROP, round(im.height * r))), Image.LANCZOS)
                c2, _ = native_crops(small)
                crops = crops + c2
                scales = MAX_SCALES
            s = self._scores(crops)
            out.append({"p_ai": float(s.mean()), "p_max": float(s.max()), "crops": len(crops), "upscaled": up, "scales": scales})
        return out

    def predict_batch(self, images: list[Image.Image], crops_per_image: int = 1) -> list[float]:
        """Fast path for video frames: one centre crop per frame, batched."""
        crops: list[Image.Image] = []
        for im in images:
            c, _ = native_crops(im, n=crops_per_image)
            crops.append(c[0])
        probs: list[float] = []
        for i in range(0, len(crops), FRAME_BATCH):
            probs += self._scores(crops[i : i + FRAME_BATCH]).tolist()
        return probs


def get_univfd() -> UnivFD:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get("univfd", UnivFD)


def borrow_univfd() -> AbstractContextManager[UnivFD]:
    """Keep the legacy model pinned throughout inference."""
    return registry.borrow("univfd", UnivFD)
