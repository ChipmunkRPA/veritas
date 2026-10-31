"""D3, the training-free second-order temporal statistic for generated video (V-4; VF §5; survey §2 #2).

Chende Zheng et al., "D3: Training-Free AI-Generated Video Detection Using Second-Order Features", ICCV 2025
(arXiv 2508.00701), github.com/Zig-HS/D3 (MIT). This module follows the upstream code (`models/D3_model.py`,
`data/datasets.py`, `utils/video2frame.py`, read 2026-09-13):

- **Frames:** consecutive frames at 8 fps; 16 per clip, or 8 when fewer than 16 exist. Veritas takes them from the plan's
  clips within one shot (`sampling.shot_clips`), because a cut is a second-order spike of its own (VF §5 item 3).
- **Pre-processing:** 10 % of the longer side is cut from each end, then a 224 × 224 bilinear resize and ImageNet mean and
  std. Upstream reads its JPEG frames with `cv2.imread` and never converts them, so the encoder sees **BGR** channel order;
  that is kept here, because the paper's figures come from that pipeline.
- **Features:** X-CLIP B/16 vision tower `pooler_output` (768-d, not normalised). The tower's layers exchange message tokens
  across each group of `num_frames` (8) consecutive items of the batch, so a clip's frames are encoded together, in order,
  as upstream's `x.reshape(-1, 3, h, w)` of a (1, 16) or (1, 8) clip does.
- **Statistic:** first order d1(k) = ‖F(k+1) − F(k)‖₂; second order d2(k) = d1(k+1) − d1(k); the score is σ, the sample
  standard deviation of d2 (`torch.std`, n − 1). No division by Δt. The paper reads a low σ as generated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

RATE_FPS = 8.0
CLIP_FRAMES = 16
MIN_FRAMES = 8  # upstream scores 8 frames when a video has fewer than 16
SIDE = 224
CROP_FRACTION = 0.1
MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)
STD = np.asarray((0.229, 0.224, 0.225), np.float32)


def clip_length(available: int) -> int:
    """Frames D3 scores from a clip with `available` frames: 16, 8, or 0 (not scored)."""
    return CLIP_FRAMES if available >= CLIP_FRAMES else MIN_FRAMES if available >= MIN_FRAMES else 0


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 frame → (3, 224, 224) float32 in the upstream channel order (BGR)."""
    bgr = rgb[..., ::-1]
    height, width = bgr.shape[:2]
    if width > height:
        cut = int(width * CROP_FRACTION)
        bgr = bgr[:, cut:width - cut]
    else:
        cut = int(height * CROP_FRACTION)
        bgr = bgr[cut:height - cut, :]
    x = cv2.resize(np.ascontiguousarray(bgr), (SIDE, SIDE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


@dataclass(frozen=True)
class ClipStatistic:
    first_order: tuple[float, ...]  # ‖F(k+1) − F(k)‖₂, k = 0 … T − 2
    second_order: tuple[float, ...]  # d1(k+1) − d1(k), k = 0 … T − 3
    mean: float
    sigma: float


def second_order(features: np.ndarray) -> ClipStatistic:
    """The D3 statistic of one clip's (T, D) frame features, T ≥ 4 (two second-order values give a sample σ)."""
    f = np.asarray(features, np.float64)
    if f.ndim != 2 or len(f) < 4:
        raise ValueError(f"D3 needs at least 4 frame features, got shape {f.shape}")
    d1 = np.linalg.norm(f[1:] - f[:-1], axis=1)
    d2 = d1[1:] - d1[:-1]
    return ClipStatistic(tuple(float(v) for v in d1), tuple(float(v) for v in d2), float(d2.mean()), float(d2.std(ddof=1)))


class XClipEncoder:
    """X-CLIP B/16 vision tower: `pooler_output` of one clip's pre-processed consecutive frames, as float32 on the CPU."""

    def __init__(self, model: Any, device: Any):
        self.model, self.device = model, device

    def features(self, frames: np.ndarray) -> np.ndarray:
        import torch

        group = int(self.model.config.num_frames)
        if len(frames) == 0 or len(frames) % group:
            raise ValueError(f"X-CLIP encodes consecutive frames in groups of {group}, got {len(frames)}")
        dtype = next(self.model.parameters()).dtype
        x = torch.from_numpy(np.ascontiguousarray(frames, np.float32)).to(self.device, dtype)
        with torch.inference_mode():
            return self.model(pixel_values=x).pooler_output.float().cpu().numpy()
