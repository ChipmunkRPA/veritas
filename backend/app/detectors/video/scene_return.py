"""Locate a brief, spatially distributed picture change that then returns."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from app.detectors.video.decode import DecodedFrame, FrameTable

FRAME_COUNT = 120
ROWS, COLUMNS, TILE = 6, 10, 16
THUMB_WIDTH = COLUMNS * TILE
THUMB_HEIGHT = ROWS * TILE
NEIGHBORS = 3
MIN_DURATION, MAX_DURATION = 6, 24
MIN_SUPPORT = 24
MIN_QUADRANT_SUPPORT = 4
MIN_TEXTURE = 4.0
EDGE_CHANGE = 0.36
STABLE_CHANGE = 0.20
MAX_PAN_PIXELS = 6.0


@dataclass(frozen=True)
class SceneReturn:
    start: int
    stop: int
    tiles: tuple[int, ...]

    @property
    def anchors(self) -> tuple[int, int, int]:
        return self.start - 1, (self.start + self.stop) // 2, self.stop


def supported(table: FrameTable) -> bool:
    tb = table.tb
    return (
        len(table) == FRAME_COUNT
        and (table.width, table.height, table.rotation) == (1280, 720, 0)
        and not table.truncated and not table.interpolated
        and all((b - a) * tb == Fraction(1, 30) for a, b in zip(table.pts, table.pts[1:]))
    )


def _picture(rgb: np.ndarray) -> np.ndarray:
    thumb = cv2.resize(rgb, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _descriptors(pictures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tiles = pictures.reshape(-1, ROWS, TILE, COLUMNS, TILE).transpose(0, 1, 3, 2, 4)
    tiles = tiles.reshape(len(pictures), ROWS * COLUMNS, TILE * TILE)
    centered = tiles - tiles.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.maximum(norms, 1.0), norms[..., 0] >= MIN_TEXTURE * TILE


def _pan(first: np.ndarray, second: np.ndarray) -> bool:
    (dx, dy), strength = cv2.phaseCorrelate(first, second)
    return strength >= .15 and (dx * dx + dy * dy) >= MAX_PAN_PIXELS ** 2


def detect(frames: Iterable[DecodedFrame], table: FrameTable) -> SceneReturn | None:
    """Report the strongest exact-clock return, if the complete profile was checked."""
    if not supported(table):
        return None
    pictures = []
    for frame in frames:
        if frame.index != len(pictures) or frame.index >= FRAME_COUNT or not frame.pts_verified or frame.pts != table.pts[frame.index]:
            return None
        pictures.append(_picture(frame.rgb))
    if len(pictures) != FRAME_COUNT:
        return None
    images = np.stack(pictures)
    features, textured = _descriptors(images)
    edges = 1.0 - np.einsum("ijk,ijk->ij", features[1:], features[:-1])
    abrupt = (edges >= EDGE_CHANGE) & textured[1:] & textured[:-1]
    proposals = np.flatnonzero(abrupt.sum(axis=1) >= MIN_SUPPORT) + 1
    edge_tiles = {}
    for n in proposals:
        if not NEIGHBORS + 1 <= n < FRAME_COUNT - NEIGHBORS or _pan(images[n - 1], images[n]):
            continue
        neighboring = np.r_[n - NEIGHBORS - 1:n - 1, n:n + NEIGHBORS]
        stable = np.max(edges[neighboring], axis=0) <= STABLE_CHANGE
        edge_tiles[int(n)] = abrupt[n - 1] & stable
    strongest = None
    for start, before in edge_tiles.items():
        for stop in range(start + MIN_DURATION, start + MAX_DURATION + 1):
            if stop not in edge_tiles:
                continue
            outside = 1.0 - np.sum(features[start - 1] * features[stop], axis=-1)
            inside = 1.0 - np.sum(features[start] * features[stop - 1], axis=-1)
            mask = before & edge_tiles[stop] & (outside <= STABLE_CHANGE) & (inside <= STABLE_CHANGE)
            grid = mask.reshape(ROWS, COLUMNS)
            quadrants = grid.reshape(2, ROWS // 2, 2, COLUMNS // 2).sum(axis=(1, 3))
            indices = tuple(int(i) for i in np.flatnonzero(mask))
            if (len(indices) >= MIN_SUPPORT and min(quadrants.flat) >= MIN_QUADRANT_SUPPORT
                    and (strongest is None or len(indices) > len(strongest.tiles))):
                strongest = SceneReturn(start, stop, indices)
    return strongest
