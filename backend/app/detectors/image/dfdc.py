"""DFDC 2020 winning face-swap classifier (Selim Seferbekov): EfficientNet-B7 (NoisyStudent) trained on
the Deepfake Detection Challenge corpus (~100k videos). Weights from the author's GitHub release.

Pre-processing follows the original inference code: face box padded by 1/3 on each side, isotropic
resize to 380 px, centred on a black 380x380 canvas, ImageNet normalisation. Output: sigmoid p(fake).
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
from torch import nn

from app.core.config import settings
from app.core.device import DEVICE
from app.core.logging import get_logger
from app.detectors.common.download import fetch
from app.detectors.common.registry import registry
from app.detectors.image.faces import Face

log = get_logger(__name__)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 380
BATCH = 8


def preprocess(rgb: np.ndarray, face: Face) -> np.ndarray:
    h, w = rgb.shape[:2]
    ph, pw = face.h // 3, face.w // 3
    y0, y1 = max(face.y - ph, 0), min(face.y + face.h + ph, h)
    x0, x1 = max(face.x - pw, 0), min(face.x + face.w + pw, w)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = SIZE / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    y, x = (SIZE - ch) // 2, (SIZE - cw) // 2
    canvas[y : y + ch, x : x + cw] = crop
    arr = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


class _B7(nn.Module):
    def __init__(self, path: Path, device: Any = None) -> None:
        super().__init__()
        self.encoder = timm.create_model("tf_efficientnet_b7", pretrained=False)
        self.fc = nn.Linear(2560, 1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # pickle: the caller verified the pinned sha256
        sd = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
        enc = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        missing = [m for m in missing if not m.startswith("classifier")]
        if missing:
            log.warning("dfdc.missing_keys", n=len(missing), sample=missing[:5])
        self.fc.load_state_dict({"weight": sd["fc.weight"], "bias": sd["fc.bias"]})
        self.eval().to(DEVICE if device is None else device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder.forward_features(x)
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.fc(f)


def load_member(path: Path, device: Any = None) -> nn.Module:
    """One B7 checkpoint on `device` (the v2 adapter loader; the caller verified the file's pinned sha256)."""
    return _B7(path, device)


class DFDCClassifier:
    """Ensemble of one or more DFDC EfficientNet-B7 checkpoints (sigmoid outputs averaged).

    v1 loads the checkpoints in `settings.dfdc_urls`; v2 adapters pass the leased members and their device."""

    def __init__(self, members: list[nn.Module] | None = None, device: Any = None) -> None:
        self.device = DEVICE if device is None else device
        self.members = members if members is not None else [
            _B7(fetch(u, u.rsplit("/", 1)[-1]), self.device) for u in settings.dfdc_urls]

    @torch.inference_mode()
    def predict(self, items: list[tuple[np.ndarray, Face]], batch: int = BATCH) -> list[float]:
        out: list[float] = []
        for i in range(0, len(items), batch):
            x = torch.from_numpy(np.stack([preprocess(rgb, f) for rgb, f in items[i : i + batch]])).to(self.device)
            probs = torch.stack([torch.sigmoid(m(x)).squeeze(-1).float() for m in self.members]).mean(0)
            out += probs.cpu().tolist()
        return out


def get_dfdc() -> DFDCClassifier:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get("dfdc_effnet_b7", DFDCClassifier)


def borrow_dfdc() -> AbstractContextManager[DFDCClassifier]:
    """Keep the legacy model pinned throughout inference."""
    return registry.borrow("dfdc_effnet_b7", DFDCClassifier)
