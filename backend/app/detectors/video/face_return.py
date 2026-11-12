# SPDX-License-Identifier: Apache-2.0
"""A bounded same-shot appearance return for a separate inspection range."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import Face, iou
from app.detectors.video.decode import FrameTable
from app.evidence.face_return_binding import (
    COARSE_TILE_MIN,
    DENSE_DIMENSIONS,
    DENSE_ENCODING,
    DENSE_FRAME_PROFILES,
    GUARDED_CODEC_CARRIERS,
    GUARDED_CODEC_FRAME_COUNT,
    PROFILE_STEPS,
)
from app.provenance.c2pa import unsupported_container

METHOD = "face-short-return/1"
POINT = .10
CADENCE = 8
MAX_RANGE = Fraction(1, 2)
DEDUPE_IOU = Fraction(1, 2)
ROLES = ("before", "during", "after")
CORE_SIZES = (3, 4, 6)
FLANK_SIZE = 4
MINIMUM_FLANK = 3
MINIMUM_TILE_COUNT = COARSE_TILE_MIN
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
    """Use a bounded scan on exact upright 1080p25/30, 720p25, and 360p25 clocks."""
    allowed_steps = PROFILE_STEPS.get((table.width, table.height))
    if table.rotation or table.interpolated or table.truncated or allowed_steps is None:
        return None
    if not 50 <= len(table) <= 240:
        return None
    tb = table.tb
    times = tuple((pts - table.pts[0]) * tb for pts in table.pts)
    step = times[1] - times[0]
    if step not in allowed_steps or any(t != i * step for i, t in enumerate(times)):
        return None
    if not 2 <= times[-1] + step <= 8:
        return None
    targets = (Fraction(i, CADENCE) for i in range(int(times[-1] * CADENCE) + 1))
    indices = tuple(dict.fromkeys(min(range(len(times)), key=lambda i: (abs(times[i] - target), i)) for target in targets))
    return times, indices


def dense_schedule(table: FrameTable, encoding: tuple[str | None, str | None]):
    """Use every second native picture on the qualified H.264 deliveries."""
    if encoding != DENSE_ENCODING or (table.width, table.height) != DENSE_DIMENSIONS or len(table) not in DENSE_FRAME_PROFILES:
        return None
    planned = schedule(table)
    return None if planned is None else (planned[0], tuple(range(0, len(table), 2)))


def source_carrier(path: Path) -> str | None:
    """Identify the two qualified carriers from a bounded native file header."""
    try:
        with path.open("rb") as stream:
            head = stream.read(4096)
    except OSError:
        return None
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "webm" if unsupported_container(head) == "webm" else None
    size = int.from_bytes(head[:4], "big")
    if head[4:8] == b"ftyp" and 16 <= size <= len(head) and size % 4 == 0:
        brands = {head[i:i + 4] for i in range(16, size, 4)}
        if head[8:12] in {b"isom", b"mp41", b"mp42"} and brands & {b"mp41", b"mp42"}:
            return "mp4"
    return None


def guarded_dense_schedule(table: FrameTable, encoding: tuple[str | None, str | None], path: Path):
    carrier = GUARDED_CODEC_CARRIERS.get(encoding)
    if carrier is None or (table.width, table.height) != DENSE_DIMENSIONS or len(table) != GUARDED_CODEC_FRAME_COUNT:
        return None
    planned = schedule(table)
    if planned is None or source_carrier(path) != carrier:
        return None
    return (planned[0], tuple(range(0, len(table), 2))), carrier


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


def _scene_correlations(a: Sample, b: Sample) -> np.ndarray | None:
    first, valid_a = a.scene
    second, valid_b = b.scene
    valid = valid_a & valid_b
    if np.count_nonzero(valid) < MINIMUM_TILE_COUNT:
        return None
    return np.sum(first[valid] * second[valid], axis=1)


def scene_correlation(a: Sample, b: Sample) -> float | None:
    values = _scene_correlations(a, b)
    return None if values is None else float(np.median(values))


def scene_lower_quartile(a: Sample, b: Sample) -> float | None:
    values = _scene_correlations(a, b)
    return None if values is None else float(np.quantile(values, .25))


def distribution_margin(distances: np.ndarray, before: list[int], core: list[int], after: list[int]) -> float:
    def cross(a, b):
        return distances[np.ix_(a, b)]

    lower = np.minimum(np.quantile(cross(core, before), .25, axis=1), np.quantile(cross(core, after), .25, axis=1))
    reference = [np.quantile(cross(before, after), .75)]
    for group in (before, core, after):
        samples = cross(group, group)
        reference.append(np.quantile(samples[np.triu_indices(len(group), k=1)], .75))
    return float(np.quantile(lower, .25) - max(reference))


def highest(times: tuple[Fraction, ...], indices: tuple[int, ...], features: dict[int, Sample], *,
            compare_scenes=None) -> Range | None:
    """Keep the top guarded interval across all core sizes before applying the duration bound."""
    visible = {i: features[i] for i in indices if features[i].vector is not None and features[i].face is not None}
    if len(visible) < 9:
        return None
    order = {i: k for k, i in enumerate(visible)}
    vectors = np.asarray([sample.vector for sample in visible.values()])
    distances = 1. - np.clip(vectors @ vectors.T, -1., 1.)
    if compare_scenes is None:
        def compare_scenes(a, b):
            return scene_correlation(features[a], features[b])
    adjacent = {(a, b): compare_scenes(a, b) for a, b in zip(indices, indices[1:])}
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
