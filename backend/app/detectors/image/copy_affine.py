"""Dense pixel agreement for a sparse same-image geometric proposal."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from app.detectors.image import copy_move

MAX_CANDIDATES = 1500
MAX_PAIRS = 3
MIN_AREA = 300


def luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), .85).astype(np.float32)


def _average(value: np.ndarray) -> np.ndarray:
    return cv2.boxFilter(value, cv2.CV_32F, (11, 11), normalize=True)


def _support(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ma, mb = _average(a), _average(b)
    va, vb = np.maximum(0, _average(a * a) - ma * ma), np.maximum(0, _average(b * b) - mb * mb)
    covariance = _average(a * b) - ma * mb
    detail = np.minimum(va, vb) >= 25
    agreement = covariance >= .94 * np.sqrt(va * vb)
    difference = _average(np.abs(a - b)) <= 3 + .25 * np.sqrt(np.maximum(va, vb))
    selected = (valid & detail & agreement & difference).astype(np.uint8)
    return cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & valid


def dense_pair(gray: np.ndarray, matrix: np.ndarray, seed: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return masks in original coordinates for two undirected areas."""
    height, width = gray.shape
    left, top, right, bottom = seed
    x0, y0, x1, y1 = max(0, left - 36), max(0, top - 36), min(width, right + 36), min(height, bottom + 36)
    size = (x1 - x0, y1 - y0)
    if min(size) < 16:
        return None
    local = matrix.copy()
    local[:, 2] += local[:, :2] @ np.array([x0, y0])
    aligned = cv2.warpAffine(gray, local, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    yy, xx = np.indices(aligned.shape)
    bx = local[0, 0] * xx + local[0, 1] * yy + local[0, 2]
    by = local[1, 0] * xx + local[1, 1] * yy + local[1, 2]
    valid = (bx >= 6) & (bx < width - 6) & (by >= 6) & (by < height - 6)
    valid &= (bx - (xx + x0)) ** 2 + (by - (yy + y0)) ** 2 >= 48**2
    selected = _support(gray[y0:y1, x0:x1], aligned, valid)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
    seeds = Counter(labels[top-y0:bottom-y0, left-x0:right-x0].ravel())
    candidates = [label for label, count in seeds.items()
                  if label and count >= 100 and stats[label, cv2.CC_STAT_AREA] >= MIN_AREA]
    if not candidates:
        return None
    label = max(candidates, key=lambda value: (seeds[value], stats[value, cv2.CC_STAT_AREA], -value))
    first = np.zeros(gray.shape, np.uint8)
    first[y0:y1, x0:x1] = labels == label
    second = cv2.warpAffine(first, matrix, (width, height), flags=cv2.INTER_NEAREST)
    if int(second.sum()) < MIN_AREA or np.any(first & second):
        return None
    return first.astype(bool), second.astype(bool)


def candidates(rgb: np.ndarray):
    return copy_move.analyse(rgb, max_candidate_pairs=MAX_CANDIDATES).distinct[:MAX_PAIRS]
