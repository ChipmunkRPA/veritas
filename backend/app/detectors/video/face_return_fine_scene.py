# SPDX-License-Identifier: Apache-2.0
"""Finer non-face tiles for a short appearance return with sparse scene texture."""

from __future__ import annotations

import cv2
import numpy as np

from app.evidence.face_return_binding import (
    FINE_CHROMA_MAX,
    FINE_LUMA_MAX,
    FINE_MEDIAN_MIN,
    FINE_QUADRANT_MIN,
    FINE_SECTOR_MIN,
    FINE_TILE_MIN,
)

WIDTH, HEIGHT = 192, 128
ROWS, COLS = 8, 16
ROW, COL = np.divmod(np.arange(ROWS * COLS), COLS)
QUADRANTS = (ROW // 4) * 2 + COL // 8
SECTORS = (ROW // 4) * 4 + COL // 4


def frame(rgb, face_box):
    reduced = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(reduced, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    blocks = yuv.reshape(ROWS, HEIGHT // ROWS, COLS, WIDTH // COLS, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(ROWS * COLS, -1, 3)
    luma = blocks[:, :, 0]
    residual = luma - luma.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(residual, axis=1)
    textured = norm >= 4.0 * np.sqrt(luma.shape[1])
    outside_face = np.ones(ROWS * COLS, dtype=bool)
    if face_box is not None:
        native_height, native_width = rgb.shape[:2]
        x, y, face_width, face_height = face_box
        outside_face &= ~(
            (COL * native_width / COLS < x + 1.1 * face_width)
            & ((COL + 1) * native_width / COLS > x - .1 * face_width)
            & (ROW * native_height / ROWS < y + 1.1 * face_height)
            & ((ROW + 1) * native_height / ROWS > y - .1 * face_height)
        )
    return residual / np.maximum(norm[:, None], 1e-7), textured, outside_face, blocks


def edge(first, second, coarse_tiles):
    left, left_textured, left_spatial, left_pixels = first
    right, right_textured, right_spatial, right_pixels = second
    space = left_spatial & right_spatial
    mask = space & left_textured & right_textured
    count = int(np.count_nonzero(mask))
    if count < FINE_TILE_MIN:
        return None, None
    support = np.sum(left[mask] * right[mask], axis=1)
    median = float(np.median(support))
    sectors = len(set(SECTORS[mask]))
    if median < FINE_MEDIAN_MIN or sectors < FINE_SECTOR_MIN:
        return None, None
    qualified = []
    quadrants = QUADRANTS[mask]
    for quadrant in sorted(set(quadrants)):
        scores = support[quadrants == quadrant]
        if len(scores) >= 3 and float(np.quantile(scores, .25)) >= .82:
            qualified.append(int(quadrant))
    if len(qualified) < FINE_QUADRANT_MIN:
        return None, None
    delta = np.mean(np.abs(left_pixels[space] - right_pixels[space]), axis=1)
    luma = float(np.quantile(delta[:, 0], .9))
    chroma = float(np.quantile(np.max(delta[:, 1:], axis=1), .9))
    if luma > FINE_LUMA_MAX or chroma > FINE_CHROMA_MAX:
        return None, None
    return median, {
        "coarse_tiles": coarse_tiles, "fine_tiles": count, "qualified_quadrants": qualified,
        "sectors": sectors, "luma_p90_hex": luma.hex(), "chroma_p90_hex": chroma.hex(),
    }
