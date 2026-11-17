"""Estimate visible speech groups from consecutive, face-aligned mouth patches."""

from __future__ import annotations

import cv2
import numpy as np
from torch import nn

WIDTH, HEIGHT = 48, 32
MOUTH_GROUPS = ("closure", "open", "rounded", "spread", "lower_lip", "other", "quiet")
PHONE_GROUPS = {
    "closure": {"b", "p", "m"}, "open": {"aa", "ae", "ah"},
    "rounded": {"ao", "ow", "uw", "uh", "w"}, "spread": {"iy", "ih", "ey", "eh", "y"},
    "lower_lip": {"f", "v"}, "quiet": {"SIL"},
}
PATCH_POINTS = (1, 33, 133, 263, 362)


def crop(rgb, landmarks):
    """Sample an eye-aligned region beneath the nose at a fixed face scale."""
    height, width = rgb.shape[:2]
    if len(landmarks) != 1 or len(landmarks[0]) != 478:
        return None
    nose, eye_left_a, eye_left_b, eye_right_a, eye_right_b = np.asarray(
        [(landmarks[0][index].x * width, landmarks[0][index].y * height) for index in PATCH_POINTS])
    eye_left = (eye_left_a + eye_left_b) / 2
    eye_right = (eye_right_a + eye_right_b) / 2
    right = eye_right - eye_left
    scale = np.linalg.norm(right)
    if not np.isfinite(scale) or scale < 18:
        return None
    down = np.array([-right[1], right[0]])
    center = nose + .42 * down
    origin = center - .69 * right - .46 * down
    corners = np.array((origin, origin + 1.38 * right, origin + .92 * down), dtype=np.float32)
    far = corners[1] + corners[2] - corners[0]
    extent = np.concatenate((corners, far[None]))
    if (not np.isfinite(extent).all() or (extent < 0).any() or (extent >= [width, height]).any()):
        return None
    target = np.array(((0, 0), (WIDTH - 1, 0), (0, HEIGHT - 1)), dtype=np.float32)
    patch = cv2.warpAffine(rgb, cv2.getAffineTransform(corners, target), (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)


def normalize(patches, valid):
    values = patches.astype(np.float32)
    if valid.any():
        common = np.median(values[valid], axis=0)
        view = (values - common) / 32.
    else:
        view = np.zeros_like(values)
    mean = values.mean(axis=(-1, -2), keepdims=True)
    scale = values.std(axis=(-1, -2), keepdims=True).clip(12.)
    views = np.stack(((values - mean) / scale, view), axis=1).clip(-5., 5.)
    views[~valid] = 0.
    return views


class VisualSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        self.picture = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(), nn.Linear(32 * 2 * 3, 48), nn.GELU(),
        )
        self.context = nn.Sequential(nn.Conv1d(48, 48, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(48, 32, 3, padding=2, dilation=2), nn.GELU(),
                                     nn.Conv1d(32, len(MOUTH_GROUPS), 1))

    def forward(self, crops):
        batch, frames = crops.shape[:2]
        embedded = self.picture(crops.flatten(0, 1)).reshape(batch, frames, -1)
        return self.context(embedded.transpose(1, 2)).transpose(1, 2)
