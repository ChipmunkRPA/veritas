"""Propose matches on a bounded image and measure their support at source resolution."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.detectors.image import copy_affine

SEARCH_LONG_SIDE = 1024


def search_size(width: int, height: int) -> tuple[int, int]:
    scale = SEARCH_LONG_SIDE / max(width, height)
    return round(width * scale), round(height * scale)


def _source_coordinates(proposal, size: tuple[int, int], reduced: tuple[int, int]):
    width, height = size
    sx, sy = reduced[0] / width, reduced[1] / height
    # Resize maps sample centers. Transform the fitted working-grid pixel
    # centers back to the source before measuring any evidence.
    to_search = np.array(((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)))
    affine = np.vstack((proposal.affine.matrix, [0.0, 0.0, 1.0]))
    lifted = (np.linalg.inv(to_search) @ affine @ to_search)[:2]
    matrix = np.array([[round(float(value), 6) for value in row] for row in lifted], dtype=np.float64)
    x, y, w, h = proposal.box_a
    seed = (max(0, math.floor(x / sx)), max(0, math.floor(y / sy)),
            min(width, math.ceil((x + w) / sx)), min(height, math.ceil((y + h) / sy)))
    return matrix, seed


def candidates(rgb: np.ndarray):
    height, width = rgb.shape[:2]
    size = search_size(width, height)
    working = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return [_source_coordinates(proposal, (width, height), size) for proposal in copy_affine.candidates(working)]


def luma(rgb: np.ndarray) -> np.ndarray:
    """Use the same smoothing extent as the working-grid luma in source pixels."""
    sigma = .85 * max(rgb.shape[:2]) / SEARCH_LONG_SIDE
    return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), sigma).astype(np.float32)
