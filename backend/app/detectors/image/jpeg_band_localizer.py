# SPDX-License-Identifier: Apache-2.0
"""Localize JPEG inconsistencies using distinct native frequency bands."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.jpeg_dq import read_coefficients

CELL = 8
BANDS = ("coarse", "middle", "fine")
FREQUENCIES = tuple(
    tuple((row, column) for row in range(CELL) for column in range(CELL) if lo <= row + column <= hi)
    for lo, hi in ((1, 3), (4, 6), (7, 14))
)
CONTEXT = 12
ARMS = (*BANDS, "combined")


def _blocks(planes, rows, cols):
    native = planes[:rows * CELL, :cols * CELL]
    return native.reshape(rows, CELL, cols, CELL, planes.shape[-1])


def _neighbor_alignment(signed):
    power = (signed**2).sum(axis=-1)
    comparisons = []
    for axis in (0, 1):
        before = np.maximum(0, np.arange(signed.shape[axis]) - 1)
        nearby = np.take(signed, before, axis=axis)
        variance = power * np.take(power, before, axis=axis)
        agreement = (nearby * signed).sum(axis=-1) / np.sqrt(1 + variance)
        comparisons.append(agreement[..., None])
    return comparisons


def features(body):
    """Measure signed coefficient relationships and decoded color on the same cells."""
    coeff = read_coefficients(body)
    if coeff is None or min(coeff.width, coeff.height) < 384 or max(coeff.width, coeff.height) > 1280:
        raise ValueError("A bounded JPEG with readable native coefficients is required")
    rows, cols = coeff.height // CELL, coeff.width // CELL
    with Image.open(io.BytesIO(body)) as image:
        if image.size != (coeff.width, coeff.height) or image.getexif().get(274, 1) != 1:
            raise ValueError("The stored and presented JPEG grids differ")
        rgb = np.asarray(image.convert("RGB"))
    ycc = (cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) - 128) / 64
    low = cv2.GaussianBlur(ycc, (0, 0), 2.4, borderType=cv2.BORDER_REFLECT_101)
    fine_base = cv2.GaussianBlur(ycc, (0, 0), .85, borderType=cv2.BORDER_REFLECT_101)
    means = _blocks(low, rows, cols).mean(axis=(1, 3))
    base = _blocks(ycc, rows, cols)
    texture = np.log1p(8 * base.std(axis=(1, 3)))
    maps = []
    for positions, residual in zip(FREQUENCIES, (low, fine_base - low, ycc - fine_base), strict=True):
        vertical, horizontal = zip(*positions, strict=True)
        raw = coeff.y[:rows, :cols, vertical, horizontal]
        magnitude = np.arcsinh(raw.astype(np.float32) * coeff.table[vertical, horizontal] / 12)
        band_activity = np.log1p(8 * _blocks(residual, rows, cols).std(axis=(1, 3)))
        active = (raw != 0).mean(axis=-1, keepdims=True, dtype=np.float32)
        stack = np.concatenate((np.abs(magnitude), means, texture, band_activity, active,
                                *_neighbor_alignment(magnitude)), axis=-1)
        maps.append(np.moveaxis(stack.astype(np.float32, copy=False), -1, 0))
    return tuple(maps)


def model():
    from torch import nn

    def branch(channels):
        layers = [nn.Conv2d(channels, 24, 1), nn.GELU()]
        # Each response sees nearby detail and more distant context on its own band.
        for dilation in (1, 2, 4):
            layers.extend((nn.Conv2d(24, 24, 3, padding=dilation, dilation=dilation, groups=24),
                           nn.Conv2d(24, 24, 1), nn.GELU()))
        return nn.Sequential(*layers, nn.Conv2d(24, 1, 1))

    return nn.ModuleList(branch(len(positions) + CONTEXT) for positions in FREQUENCIES)


def strongest(heat):
    """Return the highest response and its actual connected cells."""
    if heat.ndim != 2 or not np.isfinite(heat).all():
        raise ValueError("Expected one finite response per native cell")
    heat = cv2.GaussianBlur(heat, (0, 0), .8, borderType=cv2.BORDER_REFLECT_101)
    median, maximum = np.median(heat), heat.max()
    active = heat > median + .52 * (maximum - median)
    count, components, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), connectivity=8)
    ranked = []
    for index, (x, y, width, height, area) in enumerate(stats[1:count], 1):
        if not 6 <= area <= heat.size * .18:
            continue
        member = components[y:y + height, x:x + width] == index
        values = heat[y:y + height, x:x + width][member]
        region = float(.40 * values.max() + .60 * values.mean())
        ranked.append((round(region, 8), -int(y), -int(x), index))
    if not ranked:
        return None, np.zeros_like(active)
    region, _, _, chosen = max(ranked)
    return region, components == chosen
