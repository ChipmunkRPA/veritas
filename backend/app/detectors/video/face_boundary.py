# SPDX-License-Identifier: Apache-2.0
"""Native timing estimates for the selected short face appearance range."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from app.detectors.image.faces import Face, iou
from app.detectors.video import face_appearance
from app.detectors.video.decode import FrameTable
from app.evidence.face_boundary_clock import CADENCE as CADENCE
from app.evidence.face_boundary_clock import MAX_FRAMES as MAX_FRAMES
from app.evidence.face_boundary_clock import SCHEDULE_MARGIN as PROTOTYPE_FAR
from app.evidence.face_boundary_clock import native_schedule

METHOD = "face-appearance-boundary/1"
SEARCH = Fraction(3, 8)
PROTOTYPE_NEAR = Fraction(1, 4)
CORE_MARGIN = Fraction(1, 8)
MAX_GAP = Fraction(1, 8)


@dataclass(frozen=True)
class Edge:
    frame_indices: tuple[int, ...]
    signed_distances: tuple[float, ...]

    @property
    def first_new(self) -> int:
        return self.frame_indices[2]


@dataclass(frozen=True)
class Estimate:
    entry: Edge
    exit: Edge
    during: int


def schedule(table: FrameTable, start_pts: int, end_pts: int) -> dict[int, Fraction]:
    return native_schedule(table.pts, table.tb, start_pts, end_pts)


def _prototype(pictures: Sequence[int], vectors: Mapping[int, np.ndarray], times: Mapping[int, Fraction],
               low: Fraction, high: Fraction) -> np.ndarray | None:
    candidates = [vectors[i] for i in pictures if low <= times[i] <= high and i in vectors]
    if len(candidates) < 3:
        return None
    value = np.median(np.array(candidates), axis=0)
    return value if np.linalg.norm(value) > 1e-12 else None


def _edge(pictures: Sequence[int], vectors: Mapping[int, np.ndarray], faces: Mapping[int, Face],
          times: Mapping[int, Fraction], near: Fraction, before: np.ndarray, after: np.ndarray) -> Edge | None:
    distance = face_appearance.distance
    if distance(before, after) < .025:
        return None
    local = [i for i in pictures if abs(times[i] - near) <= SEARCH + 2 * MAX_GAP]
    ranked = []
    for slot in range(2, len(local) - 1):
        split = local[slot]
        offset = abs(times[split] - near)
        if offset > SEARCH:
            continue
        support = tuple(local[slot - 2:slot + 2])
        if any(i not in vectors for i in support):
            continue
        if any(times[b] - times[a] > MAX_GAP or iou(faces[a], faces[b]) < .35 for a, b in zip(support, support[1:])):
            continue
        signed = tuple(distance(vectors[i], before) - distance(vectors[i], after) for i in support)
        if max(signed[:2]) < 0 < min(signed[2:]):
            strength = min(abs(signed[1]), abs(signed[2]))
            ranked.append((-strength, offset, split, Edge(support, signed)))
    return min(ranked, key=lambda row: row[:3])[-1] if ranked else None


def estimate(times: Mapping[int, Fraction], start: Fraction, end: Fraction,
             features: Mapping[int, np.ndarray], faces: Mapping[int, Face]) -> Estimate | None:
    """Find both observed state transitions with corroborating same-track samples."""
    pictures = sorted(times)
    vectors = {i: face_appearance.unit(features[i]) for i in pictures if i in features}
    before = _prototype(pictures, vectors, times, start - PROTOTYPE_FAR, start - PROTOTYPE_NEAR)
    inside = _prototype(pictures, vectors, times, start + CORE_MARGIN, end - CORE_MARGIN)
    after = _prototype(pictures, vectors, times, end + PROTOTYPE_NEAR, end + PROTOTYPE_FAR)
    if before is None or inside is None or after is None:
        return None
    distance = face_appearance.distance
    if distance(before, after) >= min(distance(before, inside), distance(after, inside)):
        return None
    entry = _edge(pictures, vectors, faces, times, start, before, inside)
    exit_edge = _edge(pictures, vectors, faces, times, end, inside, after)
    if entry is None or exit_edge is None or times[exit_edge.first_new] - times[entry.first_new] < Fraction(3, 8):
        return None
    core = [i for i in pictures if entry.first_new <= i < exit_edge.first_new and i in vectors]
    midpoint = (times[entry.first_new] + times[exit_edge.first_new]) / 2
    during = min(core, key=lambda i: (abs(times[i] - midpoint), i))
    return Estimate(entry, exit_edge, during)
