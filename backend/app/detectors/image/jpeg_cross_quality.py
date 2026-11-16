# SPDX-License-Identifier: Apache-2.0
"""Measure local JPEG coefficients and decoded residual texture on the native grid."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.jpeg_dq import read_coefficients
from app.provenance.metadata_rules import ZIGZAG

CELL = 16
AC = tuple(divmod(int(index), 8) for index in ZIGZAG[1:21])
RESIDUAL_SCALES = (.65, 1.3, 2.6)
PADDING = cv2.BORDER_REFLECT_101


def pool(values, height, width, side):
    planes = values.shape[-1]
    return values[:height * side, :width * side].reshape(height, side, width, side, planes)


def features(body):
    coefficients = read_coefficients(body)
    if coefficients is None:
        raise ValueError("Readable quantized JPEG coefficients are required")
    height, width = coefficients.height // CELL, coefficients.width // CELL
    if min(height, width) < 16 or max(coefficients.height, coefficients.width) > 1280:
        raise ValueError("The JPEG is outside the measured native grid")
    ys, xs = zip(*AC, strict=True)
    raw = coefficients.y[:height * 2, :width * 2, ys, xs].astype(np.float32)
    dequantized = np.abs(raw * coefficients.table[ys, xs].astype(np.float32))
    activity = pool(np.log1p(dequantized), height, width, 2)
    occupied = pool((raw != 0).astype(np.float32), height, width, 2)
    dct = [activity.mean(axis=(1, 3)), occupied.mean(axis=(1, 3)), activity.std(axis=(1, 3))]

    with Image.open(io.BytesIO(body)) as image:
        if image.size != (coefficients.width, coefficients.height):
            raise ValueError("JPEG pixels and coefficient geometry differ")
        rgb = np.asarray(image.convert("RGB"))
    colors = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    texture = []
    previous = colors
    for scale in RESIDUAL_SCALES:
        blurred = cv2.GaussianBlur(colors, (0, 0), scale, borderType=PADDING)
        residual = pool(previous - blurred, height, width, CELL)
        texture.append(np.log1p(np.mean(residual**2, axis=(1, 3))))
        previous = blurred
    base = np.concatenate((*dct, *texture), axis=2)
    inside = cv2.blur(base, (3, 3), borderType=PADDING)
    ring = (49 * cv2.blur(base, (7, 7), borderType=PADDING) - 9 * inside) / 40
    return np.ascontiguousarray(np.concatenate((base, inside, inside - ring), axis=2), dtype=np.float32)


def region(feature_grid, model):
    center = np.asarray(model["center"], dtype=np.float32)
    inverse = np.asarray(model["inverse_scale"], dtype=np.float32)
    weight = np.asarray(model["coefficients"], dtype=np.float32)
    heat = ((feature_grid - center) * inverse) @ weight + model["intercept"]
    heat = cv2.GaussianBlur(heat, (0, 0), .65, borderType=PADDING)
    if not np.isfinite(heat).all():
        raise ValueError("The image model produced a non-finite local reading")
    median, peak = float(np.median(heat)), float(heat.max())
    active = (heat > median + .58 * (peak - median)).astype(np.uint8)
    count, parts, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    choices = []
    for index, (x, y, w, h, area) in enumerate(stats[1:count], 1):
        if 3 <= area <= heat.size * .16:
            mask = parts[y:y + h, x:x + w] == index
            values = heat[y:y + h, x:x + w][mask]
            score = float(.65 * values.max() + .35 * values.mean())
            choices.append((round(score, 8), -int(y), -int(x), index))
    if not choices:
        return 0., np.zeros_like(active, dtype=bool)
    score, _, _, index = max(choices)
    return score, parts == index
