"""Locate two moving ranges with a constant crop between their pictures."""
from __future__ import annotations

import math
from itertools import product
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from app.evidence.picture_sequence import MIN_SPAN

WORK_WIDTH, WORK_HEIGHT = 192, 108
GRID_WIDTH, GRID_HEIGHT = 32, 18
MIN_EDGES = 8
MAX_EDGE_STEP = 3
MIN_SUPPORT_FRACTION = 0.60
MAX_PICTURE_RMS = 3.0
MIN_DIRECT_RMS = 7.0
MIN_MOTION_COSINE = 0.82
MIN_MOTION_RATIO = 0.60
MIN_MOTION_RMS_25FPS = 0.37
CROP_PERCENTAGES = (70, 75, 80, 85, 90)


class CropMatch(NamedTuple):
    ranges: tuple[tuple[int, int], tuple[int, int]]
    wider_range: str
    crop_box: tuple[int, int, int, int]
    support: tuple[int, ...]


def luminance(rgb: np.ndarray) -> np.ndarray:
    small = Image.fromarray(rgb).resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.BOX)
    return np.asarray(small.convert("L"), dtype=np.float32) - 128


def crop_boxes():
    """Return integer boxes on the working grid, in left, top, right, bottom order."""
    for scale in CROP_PERCENTAGES:
        width = round(WORK_WIDTH * scale / 100)
        height = round(WORK_HEIGHT * scale / 100)
        spare_x, spare_y = WORK_WIDTH - width, WORK_HEIGHT - height
        for x, y in product((0, spare_x // 2, spare_x), (0, spare_y // 2, spare_y)):
            yield x, y, x + width, y + height


def _vectors(frames, box):
    left, top, right, bottom = box
    return np.stack([cv2.resize(frame[top:bottom, left:right], (GRID_WIDTH, GRID_HEIGHT),
                                interpolation=cv2.INTER_AREA).ravel() for frame in frames])


def _motion(vectors):
    difference = np.diff(vectors, axis=0)
    difference -= difference.mean(axis=1, keepdims=True)
    return difference, np.sqrt(np.square(difference).mean(axis=1))


def _distance(left, right, left_power, right_power):
    return left_power[:, None] + right_power[None, :] - 2 * (left @ right.T) / left.shape[1]


def _chains(linked, moving, minimum_steps):
    for lag in range(minimum_steps + 1, len(linked) - minimum_steps + 1):
        connected = linked.diagonal(lag)
        offsets = np.flatnonzero(connected & moving.diagonal(lag))
        if len(offsets) < MIN_EDGES:
            continue
        # Similarity must hold between supporting changes. A sequence may
        # include near-static pictures, but it cannot cross a broken match.
        gaps = np.diff(offsets) > MAX_EDGE_STEP
        losses = np.diff(np.cumsum(~connected)[offsets]) > 0
        for chain in np.split(offsets, np.flatnonzero(gaps | losses) + 1):
            for first in range(len(chain) - MIN_EDGES + 1):
                start = int(chain[first])
                stop = int(np.searchsorted(chain, start + lag - 1, side="left"))
                if stop - first < MIN_EDGES:
                    continue
                end = int(chain[stop - 1]) + 1
                if end - start >= minimum_steps and stop - first >= MIN_SUPPORT_FRACTION * (end - start):
                    yield start, end, lag, tuple(int(i) - start for i in chain[first:stop])


def find(frames: list[np.ndarray], frame_rate: float) -> CropMatch | None:
    if len(frames) < 2 * (math.ceil(frame_rate * MIN_SPAN) + 1):
        return None
    minimum_steps = math.ceil(frame_rate * MIN_SPAN)
    full = _vectors(frames, (0, 0, WORK_WIDTH, WORK_HEIGHT))
    power = np.square(full).mean(axis=1)
    distinct = _distance(full, full, power, power) >= MIN_DIRECT_RMS ** 2
    motion, rms = _motion(full)
    minimum_rms = MIN_MOTION_RMS_25FPS * 25 / frame_rate
    best_rank, best_match = None, None
    for box in crop_boxes():
        cropped = _vectors(frames, box)
        crop_power = np.square(cropped).mean(axis=1)
        distances = _distance(cropped, full, crop_power, power)
        close = (distances <= MAX_PICTURE_RMS ** 2) & distinct
        linked = close[:-1, :-1] & close[1:, 1:]
        if np.count_nonzero(linked) < MIN_EDGES:
            continue
        crop_motion, crop_rms = _motion(cropped)
        weaker = np.minimum(crop_rms[:, None], rms[None, :])
        stronger = np.maximum(crop_rms[:, None], rms[None, :])
        correlation = (crop_motion @ motion.T) / full.shape[1]
        moving = ((weaker >= minimum_rms) & (weaker >= MIN_MOTION_RATIO * stronger)
                  & (correlation >= MIN_MOTION_COSINE * weaker * stronger))
        for wider, connections, changes, errors in (
            ("first", linked, moving, distances), ("second", linked.T, moving.T, distances.T),
        ):
            for start, end, lag, support in _chains(connections, changes, minimum_steps):
                indices = np.arange(start, end + 1)
                error = float(np.mean(errors[indices, indices + lag]))
                rank = end - start, len(support), -error, -start, -lag
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_match = CropMatch(((start, end), (start + lag, end + lag)), wider, box, support)
    return best_match
