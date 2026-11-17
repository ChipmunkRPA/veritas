# SPDX-License-Identifier: Apache-2.0
"""Native JPEG region and boundary measurements for the paired-edit trial."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import jpeg_band_localizer as dct

CELL = dct.CELL
FEATURES = 43
ARMS = ("plain", "paired", "boundary")


def _pool(values, height, width):
    return values[: height * CELL, : width * CELL].reshape(height, CELL, width, CELL, -1).mean(axis=(1, 3))


def features(body):
    """Combine native coefficient summaries and differences across neighboring cells."""
    bands = dct.features(body)
    height, width = bands[0].shape[1:]
    native = []
    for positions, band in zip(dct.FREQUENCIES, bands, strict=True):
        ac = band[: len(positions)]
        native.extend((ac.mean(axis=0), ac.std(axis=0), *band[-3:]))
    native.extend(bands[0][-12:-6])
    with Image.open(io.BytesIO(body)) as decoded:
        rgb = np.asarray(decoded.convert("RGB"))
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(ycc, (0, 0), 1.25)
    residual = ycc - smooth
    medium = smooth - cv2.GaussianBlur(ycc, (0, 0), 3.25)
    for plane in (residual, medium):
        energy = _pool(plane**2, height, width)
        native.extend(np.moveaxis(np.log1p(320 * np.sqrt(energy)), -1, 0))
        cross = _pool(plane * np.roll(plane, -1, axis=-1), height, width)
        correlation = cross / np.sqrt(2e-6 + energy * np.roll(energy, -1, axis=-1))
        native.extend(np.moveaxis(correlation, -1, 0))
    fine = np.log1p(180 * _pool(np.abs(residual), height, width))
    for axis in (0, 1):
        before = np.maximum(0, np.arange(fine.shape[axis]) - 1)
        after = np.minimum(fine.shape[axis] - 1, np.arange(fine.shape[axis]) + 1)
        differences = np.abs(np.take(fine, after, axis=axis) - np.take(fine, before, axis=axis))
        native.extend(np.moveaxis(differences, -1, 0))
        gradient = np.abs(cv2.Sobel(smooth[..., 0], cv2.CV_32F, int(axis == 1), int(axis == 0), ksize=3))
        # Contrast normalized texture with the scene gradient on the same cells.
        strength = _pool(gradient[..., None], height, width)[..., 0]
        native.extend((np.log1p(12 * strength), fine[..., 0] / (1 + 10 * strength)))
    value = np.stack(native).astype(np.float32)
    if len(value) != FEATURES:
        raise ValueError("The JPEG boundary feature layout changed")
    return value


def model():
    from torch import nn

    width = 32
    layers = [nn.Conv2d(FEATURES, width, 1), nn.GELU()]
    for dilation in (1, 2, 4, 1):
        layers.extend(
            (
                nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width),
                nn.Conv2d(width, width, 1),
                nn.GELU(),
            )
        )
    return nn.Sequential(*layers, nn.Conv2d(width, 2, 1))


def _ring(mask):
    binary = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    outside = cv2.dilate(binary, kernel) > 0
    inside = cv2.erode(binary, kernel) > 0
    return outside & ~mask, outside & ~inside


def proposals(occupancy, boundary=None):
    """Score connected regions at several relative levels on one native grid."""
    heat = cv2.GaussianBlur(np.asarray(occupancy, dtype=np.float32), (0, 0), 0.7)
    edge = None if boundary is None else cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), 0.65)
    largest = max(8, round(0.19 * heat.size))
    median = float(np.median(heat))
    maximum = float(heat.max())
    levels = [median + fraction * (maximum - median) for fraction in (0.32, 0.46, 0.60, 0.74)]
    ranked = []
    for level in levels:
        active = (heat > level).astype(np.uint8)
        active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        count, components, stats, _ = cv2.connectedComponentsWithStats(active)
        for i in range(1, count):
            x, y, w, h, size = map(int, stats[i])
            if not 8 <= size <= largest or min(w, h) < 3:
                continue
            mask = components == i
            outer, contour = _ring(mask)
            if int(outer.sum()) < 8:
                continue
            internal = float(np.mean(heat[mask]))
            difference = internal - float(np.mean(heat[outer]))
            score = 0.58 * internal + 0.42 * difference
            if edge is not None:
                # Require a responding contour near the edge of the same proposed area.
                support = float(np.mean(edge[contour]))
                score += 0.50 * support
            ranked.append((round(score, 8), size, -y, -x, mask))
    if not ranked:
        return None, np.zeros_like(heat, dtype=bool)
    selected = max(ranked, key=lambda row: row[:4])
    return selected[0], selected[-1]
