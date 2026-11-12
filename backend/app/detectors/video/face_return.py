# SPDX-License-Identifier: Apache-2.0
"""A bounded same-shot appearance return for a separate inspection range."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from app.detectors.image.faces import Face, iou
from app.detectors.video.decode import FrameTable

METHOD = "face-short-return/1"
POINT = .10
CADENCE = 8
MAX_RANGE = Fraction(1, 2)
DEDUPE_IOU = Fraction(1, 2)
ROLES = ("before", "during", "after")
CORE_SIZES = (3, 4, 6)
FLANK_SIZE = 4
MINIMUM_FLANK = 3
MINIMUM_TILE_COUNT = 8
SCENE_CORRELATION_POINT = .55
MAX_GAP = .26
MINIMUM_FACE_IOU = .25


@dataclass(frozen=True)
class Sample:
    face: Face | None
    vector: np.ndarray | None
    scene: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Range:
    start: int
    end: int
    response: float
    scene_correlations: tuple[float, float]
    support: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    scene_frames: tuple[int, int, int, int]

    def anchors(self) -> dict[str, int]:
        before, during, after = self.support
        return {"before": before[-1], "during": during[len(during) // 2], "after": after[0]}


def schedule(table: FrameTable) -> tuple[tuple[Fraction, ...], tuple[int, ...]] | None:
    """The pilot uses observed upright 1080p clocks at 25 or 30 Hz and a bounded scan."""
    if table.rotation or table.interpolated or table.truncated or (table.width, table.height) != (1920, 1080):
        return None
    if not 50 <= len(table) <= 240:
        return None
    tb = table.tb
    times = tuple((pts - table.pts[0]) * tb for pts in table.pts)
    step = times[1] - times[0]
    if step not in (Fraction(1, 25), Fraction(1, 30)) or any(t != i * step for i, t in enumerate(times)):
        return None
    if not 2 <= times[-1] + step <= 8:
        return None
    targets = (Fraction(i, CADENCE) for i in range(int(times[-1] * CADENCE) + 1))
    indices = tuple(dict.fromkeys(min(range(len(times)), key=lambda i: (abs(times[i] - target), i)) for target in targets))
    return times, indices


def scene(rgb: np.ndarray, face: Face | None) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(cv2.resize(rgb, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    tiles = gray.reshape(4, 16, 8, 12).transpose(0, 2, 1, 3).reshape(32, 192).astype(np.float32)
    tiles -= tiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(tiles, axis=1)
    mask = norms >= 4. * math.sqrt(192)
    if face is not None:
        height, width = rgb.shape[:2]
        x, y, w, h = face.box()
        for k in range(len(tiles)):
            row, col = divmod(k, 8)
            if ((col + 1) * width / 8 > x - .1 * w and col * width / 8 < x + 1.1 * w
                    and (row + 1) * height / 4 > y - .1 * h and row * height / 4 < y + 1.1 * h):
                mask[k] = False
    return tiles / np.maximum(norms[:, None], 1e-6), mask


def scene_correlation(a: Sample, b: Sample) -> float | None:
    first, valid_a = a.scene
    second, valid_b = b.scene
    valid = valid_a & valid_b
    if np.count_nonzero(valid) < MINIMUM_TILE_COUNT:
        return None
    return float(np.median(np.sum(first[valid] * second[valid], axis=1)))


def distribution_margin(distances: np.ndarray, before: list[int], core: list[int], after: list[int]) -> float:
    def cross(a, b):
        return distances[np.ix_(a, b)]

    lower = np.minimum(np.quantile(cross(core, before), .25, axis=1), np.quantile(cross(core, after), .25, axis=1))
    reference = [np.quantile(cross(before, after), .75)]
    for group in (before, core, after):
        samples = cross(group, group)
        reference.append(np.quantile(samples[np.triu_indices(len(group), k=1)], .75))
    return float(np.quantile(lower, .25) - max(reference))


def highest(times: tuple[Fraction, ...], indices: tuple[int, ...], features: dict[int, Sample]) -> Range | None:
    """Keep the top guarded interval across all core sizes before applying the duration bound."""
    visible = {i: features[i] for i in indices if features[i].vector is not None and features[i].face is not None}
    if len(visible) < 9:
        return None
    order = {i: k for k, i in enumerate(visible)}
    vectors = np.asarray([sample.vector for sample in visible.values()])
    distances = 1. - np.clip(vectors @ vectors.T, -1., 1.)
    adjacent = {(a, b): scene_correlation(features[a], features[b]) for a, b in zip(indices, indices[1:])}
    best = None
    for size in CORE_SIZES:
        needed = max(3, math.ceil(size * .66))
        for start in range(FLANK_SIZE, len(indices) - size - FLANK_SIZE + 1):
            stop = start + size
            before, core, after = (tuple(i for i in group if i in visible) for group in (
                indices[start - FLANK_SIZE:start], indices[start:stop], indices[stop:stop + FLANK_SIZE]))
            if min(len(before), len(after)) < MINIMUM_FLANK or len(core) < needed:
                continue
            joined = before + core + after
            if any(times[b] - times[a] > MAX_GAP or iou(visible[a].face, visible[b].face) < MINIMUM_FACE_IOU
                   for a, b in zip(joined, joined[1:])):
                continue
            scene_frames = (indices[start - 1], indices[start], indices[stop - 1], indices[stop])
            entry, exit_ = adjacent[scene_frames[:2]], adjacent[scene_frames[2:]]
            if entry is None or exit_ is None or min(entry, exit_) < SCENE_CORRELATION_POINT:
                continue
            left, middle, right = ([order[i] for i in group] for group in (before, core, after))
            response = distribution_margin(distances, left, middle, right)
            if best is None or response > best.response:
                best = Range(indices[start], indices[stop], response, (entry, exit_), (before, core, after), scene_frames)
    return best


def distinct(candidate: Range, table: FrameTable, incumbent: dict | None) -> bool:
    start, end = table.pts[candidate.start], table.pts[candidate.end]
    if candidate.response <= POINT or not 0 < (end - start) * table.tb < MAX_RANGE:
        return False
    if incumbent is None:
        return True
    before, after = incumbent["start_pts"], incumbent["end_pts"]
    overlap = max(0, min(end, after) - max(start, before))
    return Fraction(overlap, max(end, after) - min(start, before)) < DEDUPE_IOU
