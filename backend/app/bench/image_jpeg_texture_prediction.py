# SPDX-License-Identifier: Apache-2.0
"""Predict native JPEG detail from intact-photo context and locate residual changes."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.jpeg_band_localizer import CELL, FREQUENCIES
from app.detectors.image.jpeg_boundary_localizer import proposals
from app.detectors.image.jpeg_dq import read_coefficients

LOW_INPUTS = 18
TARGETS = 8
ARMS = ("texture_deficit", "texture_surprise", "neighbor_deficit", "joint_deficit")


def _pool(values, rows, cols):
    return values[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL, -1).mean(axis=(1, 3))


def _frequency_energy(tiles, table, positions):
    planes = [tiles[:, :, y, x].astype(np.float32) * float(table[y, x]) for y, x in positions]
    return np.log1p(np.sqrt(np.mean(np.square(planes), axis=0)) / 8)


def _ring(values, outer, inner):
    support = np.ones(values.shape[:2], np.float32)

    def total(field, width):
        return cv2.boxFilter(field, -1, (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)

    counts = total(support, outer) - total(support, inner)
    return (total(values, outer) - total(values, inner)) / counts[..., None]


def measure(body):
    """Retain coarse content and keep the target detail cells out of the neighbor input."""
    coefficients = read_coefficients(body)
    if coefficients is None or not 384 <= min(coefficients.width, coefficients.height) <= max(
            coefficients.width, coefficients.height) <= 1280:
        raise ValueError("Texture prediction requires one readable, bounded JPEG")
    rows, cols = coefficients.height // CELL, coefficients.width // CELL
    luma = coefficients.y[:rows, :cols]
    table = coefficients.table
    with Image.open(io.BytesIO(body)) as image:
        if image.getexif().get(274, 1) != 1 or image.size != (coefficients.width, coefficients.height):
            raise ValueError("JPEG analysis must preserve the presented pixel grid")
        rgb = np.asarray(image.convert("RGB"))
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255
    level = _pool(ycc, rows, cols)
    coarse = []
    for y, x in FREQUENCIES[0]:
        coarse.append(np.arcsinh(luma[:, :, y, x].astype(np.float32) * float(table[y, x]) / 16))

    other = cv2.GaussianBlur(ycc, (0, 0), 1.4, borderType=cv2.BORDER_REFLECT_101)
    fine_chroma = np.log1p(96 * np.sqrt(np.maximum(0., _pool((ycc[..., 1:] - other[..., 1:]) ** 2, rows, cols))))
    # Split the two target bands by direction. Target energy never enters the same cell's
    # coarse input and is compared separately with detail beyond a two-cell buffer.
    directions = []
    for band in FREQUENCIES[1:]:
        for direction in (-1, 0, 1):
            positions = [(y, x) for y, x in band if np.sign(y - x) == direction]
            directions.append(_frequency_energy(luma, table, positions))
    target = np.dstack((*directions, fine_chroma))
    quantization = np.array([np.log1p(np.mean([table[y, x] for y, x in band])) / 4
                             for band in FREQUENCIES], dtype=np.float32)
    quantization = np.broadcast_to(quantization, (rows, cols, len(quantization)))
    gradient = np.dstack([np.abs(cv2.Sobel(level[..., 0], cv2.CV_32F, x, y, ksize=3))
                          for x, y in ((1, 0), (0, 1))])
    image_profile = np.broadcast_to(np.median(level, axis=(0, 1)), level.shape[:2] + (3,))[..., :1]
    low = np.dstack((level, *coarse, gradient, quantization, image_profile))
    context = np.concatenate((_ring(target, 11, 5), _ring(target, 19, 9)), axis=-1)
    if low.shape[-1] != LOW_INPUTS or target.shape[-1] != TARGETS:
        raise ValueError("Texture prediction features differ from the measurement layout")
    return tuple(np.moveaxis(x.astype(np.float32, copy=False), -1, 0) for x in (low, context, target))


def model():
    from torch import cat, nn

    class TexturePrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse = nn.Sequential(
                nn.Conv2d(LOW_INPUTS, 24, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=2, dilation=2, groups=24),
                nn.Conv2d(24, 24, 1),
                nn.GELU(),
            )
            self.content = nn.Conv2d(24, 2 * TARGETS, 1)
            self.neighbor = nn.Sequential(nn.Conv2d(24 + 2 * TARGETS, 24, 1), nn.GELU(), nn.Conv2d(24, 2 * TARGETS, 1))

        def forward(self, low, context):
            coarse = self.coarse(low)
            return self.content(coarse), self.neighbor(cat((coarse, context), dim=1))

    return TexturePrediction()


def errors(content, neighbor, target):
    def standardized(values):
        prediction, scale = np.split(values, 2, axis=0)
        return np.clip((prediction - target) / np.exp(np.clip(scale, -2.8, 1.4)), -12, 12)

    standalone = standardized(content)
    nearby = standardized(neighbor)
    deficit = np.maximum(standalone, 0).mean(axis=0)
    neighborhood_deficit = np.maximum(nearby, 0).mean(axis=0)
    return {
        "texture_deficit": deficit,
        "texture_surprise": np.abs(standalone).mean(axis=0),
        "neighbor_deficit": neighborhood_deficit,
        "joint_deficit": np.sqrt(deficit * neighborhood_deficit),
    }


def locate(values):
    # The area rule and proposal geometry are the same across all four readings.
    cropped = values.copy()
    cropped[:3] = cropped[-3:] = 0
    cropped[:, :3] = cropped[:, -3:] = 0
    return proposals(cropped)
