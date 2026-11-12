# SPDX-License-Identifier: Apache-2.0
"""Development candidate for paired scene changes around a tracked face range."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames, probe_frames

WIDTH, HEIGHT = 640, 360
FRAME_COUNT = 120
STEP = Fraction(1, 25)
ROWS, COLS = 6, 8
TILE_H, TILE_W = 16, 20
FACE_PAD = .35
RADIUS = 9
CONTEXT = 2
MIN_CUT_FRAMES, MAX_CUT_FRAMES = 5, 12
MAX_COARSE_FRAMES = 24
MIN_VALID_TILES = 16
MIN_SUPPORT_FRACTION = .35
MIN_TEXTURE_WITH_COLOR = 1.2
MIN_TEXTURE_WITHOUT_COLOR = 2.5
MIN_COLOR = .85
CONTEXT_MULTIPLE = 3.
RETURN_MULTIPLE = .55
METHOD = "paired-background-edge/1"


@dataclass(frozen=True)
class Tiles:
    texture: np.ndarray
    contrast: np.ndarray
    chroma: np.ndarray
    rgb_sha256: str


def _scope(table: FrameTable, coarse: dict) -> tuple[int, int] | None:
    if ((table.width, table.height) != (WIDTH, HEIGHT) or len(table) != FRAME_COUNT
            or table.interpolated or table.truncated or table.rotation or table.time_base != coarse.get("time_base")
            or any((b - a) * table.tb != STEP for a, b in zip(table.pts, table.pts[1:]))):
        return None
    try:
        start, end = (table.pts.index(coarse[key]) for key in ("start_pts", "end_pts"))
    except (KeyError, ValueError):
        return None
    return (start, end) if 0 < end - start <= MAX_COARSE_FRAMES else None


def _eligible_tiles(observations: list[dict]) -> np.ndarray | None:
    if not observations:
        return None
    row, column = np.indices((ROWS, COLS)).reshape(2, -1)
    left, top = column * WIDTH / COLS, row * HEIGHT / ROWS
    mask = np.ones(ROWS * COLS, dtype=bool)
    for item in observations:
        x, y, w, h = (item["bbox"][key] for key in ("x", "y", "w", "h"))
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            return None
        mask &= ~((left + WIDTH / COLS > x - FACE_PAD * w) & (left < x + (1. + FACE_PAD) * w)
                  & (top + HEIGHT / ROWS > y - FACE_PAD * h) & (top < y + (1. + FACE_PAD) * h))
    return mask if np.count_nonzero(mask) >= MIN_VALID_TILES else None


def _tiles(rgb: np.ndarray, eligible: np.ndarray) -> Tiles:
    thumbnail = cv2.resize(rgb, (COLS * TILE_W, ROWS * TILE_H), interpolation=cv2.INTER_AREA)
    colors = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    tiles = colors.reshape(ROWS, TILE_H, COLS, TILE_W, 3).transpose(0, 2, 1, 3, 4)
    pixels = tiles.reshape(ROWS * COLS, TILE_H * TILE_W, 3)[eligible]
    luminance = pixels[:, :, 0]
    centered = luminance - luminance.mean(axis=1, keepdims=True)
    contrast = np.maximum(np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True)), 2.)
    return Tiles(centered / contrast, contrast, pixels[:, :, 1:], hashlib.sha256(rgb.tobytes()).hexdigest())


def _distance(first: Tiles, second: Tiles) -> np.ndarray:
    texture = np.mean(np.abs(first.texture - second.texture) * np.minimum(first.contrast, second.contrast), axis=1)
    chroma = np.mean(np.abs(first.chroma - second.chroma), axis=(1, 2))
    return np.column_stack((texture, chroma))


def _quadrants(tiles: np.ndarray) -> int:
    rows, columns = np.divmod(tiles, COLS)
    counts = np.bincount(2 * (rows >= ROWS // 2) + (columns >= COLS // 2), minlength=4)
    return int(np.count_nonzero(counts >= 2))


def _support(left: int, right: int, frames: dict[int, Tiles], adjacent: dict[int, np.ndarray]) -> np.ndarray:
    edges = np.minimum(adjacent[left], adjacent[right])
    baseline = np.median([adjacent[left - 1], adjacent[left + 1], adjacent[right - 1], adjacent[right + 1]], axis=0)
    within = np.maximum(_distance(frames[left - 1], frames[right]), _distance(frames[left], frames[right - 1]))
    stable = within <= RETURN_MULTIPLE * edges
    normalized = edges >= CONTEXT_MULTIPLE * baseline
    texture = stable[:, 0] & normalized[:, 0]
    color = stable[:, 1] & normalized[:, 1] & (edges[:, 1] >= MIN_COLOR)
    return texture & ((edges[:, 0] >= MIN_TEXTURE_WITHOUT_COLOR) | ((edges[:, 0] >= MIN_TEXTURE_WITH_COLOR) & color))


def examine(path: Path, coarse: dict, *, table: FrameTable | None = None) -> dict | None:
    """Return native edge indices and source bindings for one distinct background departure and return."""
    table = probe_frames(path) if table is None else table
    interval = _scope(table, coarse)
    eligible = _eligible_tiles(coarse.get("frame_inputs", []))
    if interval is None or eligible is None:
        return None
    start, end = interval
    first = max(0, start - RADIUS - CONTEXT)
    last = min(len(table) - 1, end + RADIUS + CONTEXT)
    needed = range(first, last + 1)
    frames = {}
    for picture in decode_frames(path, table, needed, max_side=None, timeout_s=4):
        if picture.pts_verified:
            frames[picture.index] = _tiles(picture.rgb, eligible)
    if len(frames) != len(needed):
        return None
    adjacent = {i: _distance(frames[i - 1], frames[i]) for i in range(first + 1, last + 1)}
    strongest = set(i for i, distance in adjacent.items() if np.count_nonzero(distance[:, 0] >= MIN_TEXTURE_WITH_COLOR)
                    >= MIN_SUPPORT_FRACTION * np.count_nonzero(eligible))
    findings = []
    for left in sorted(strongest):
        if abs(left - start) > RADIUS or left < first + CONTEXT:
            continue
        for right in range(left + MIN_CUT_FRAMES, min(left + MAX_CUT_FRAMES, last - 1) + 1):
            if right not in strongest or abs(right - end) > RADIUS:
                continue
            overlap = min(right, end) - max(left, start)
            if 2 * overlap < min(end - start, right - left):
                continue
            supported = np.flatnonzero(eligible)[_support(left, right, frames, adjacent)]
            if len(supported) >= MIN_SUPPORT_FRACTION * np.count_nonzero(eligible) and _quadrants(supported) >= 3:
                findings.append((left, right, supported.tolist()))
    if len(findings) != 1:
        return None
    left, right, tiles = findings[0]
    return {"method": METHOD, "time_base": table.time_base, "coarse_frames": [start, end],
            "edge_frames": [left, right], "edge_pts": [table.pts[left], table.pts[right]],
            "eligible_tiles": np.flatnonzero(eligible).tolist(), "support_tiles": tiles,
            "pictures": [{"index": i, "pts": table.pts[i], "rgb_sha256": frames[i].rgb_sha256}
                         for i in (left - 1, left, right - 1, right)]}
