# SPDX-License-Identifier: Apache-2.0
"""Detect a local shift in interpolation phase in a narrowly qualified PNG."""
from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

CELL = 32
SCOPE = (1536, 2048)
POINT = .25
FOLD_CLOSURE = .58
PHASE_SEPARATION = .18
MIN_SAMPLES = 36
CLOSURE_TOLERANCE = 12
ENDPOINT_SEPARATION = 48
REGION_SIDES = (2, 3, 4, 5, 6)
MAX_BODY_BYTES = 25 * 1024 * 1024
MAX_CHUNKS = 4096
PROFILE = b"\x10\x02\x00\x00\x00"
PNG = b"\x89PNG\r\n\x1a\n"

Status = Literal["unsupported_png", "no_repeatable_stencil", "assessed"]


@dataclass(frozen=True)
class InterpolationReview:
    status: Status
    phase: int | None = None
    score: float = 0.
    proposal: tuple[int, int, int, int] | None = None
    supported_cells: int = 0
    reference_closure: float | None = None
    decoded_rgb_sha256: str | None = None

    @property
    def region(self) -> tuple[int, int, int, int] | None:
        return self.proposal if self.score > POINT else None


def allowed_png(body: bytes | memoryview) -> bool:
    if not 33 <= len(body) <= MAX_BODY_BYTES or body[:8] != PNG:
        return False
    height, width = SCOPE
    if body[8:29] != struct.pack(">I4sII", 13, b"IHDR", width, height) + PROFILE:
        return False
    view = memoryview(body)
    offset = 8
    idat = False
    for index in range(MAX_CHUNKS):
        if offset + 12 > len(body):
            return False
        size, kind = struct.unpack_from(">I4s", body, offset)
        end = offset + 8 + size
        if end + 4 > len(body) or zlib.crc32(view[offset+4:end]) != struct.unpack_from(">I", body, end)[0]:
            return False
        if kind == b"IHDR":
            if index != 0:
                return False
        elif kind == b"IDAT":
            idat = True
        elif kind == b"IEND":
            return idat and size == 0 and end + 4 == len(body)
        else:
            return False
        offset = end + 4
    return False


def decoded_rgb(body: bytes | memoryview) -> np.ndarray | None:
    if not allowed_png(body):
        return None
    try:
        bgr = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None
    if bgr is None or bgr.dtype != np.uint16 or bgr.shape != (*SCOPE, 3):
        return None
    return np.ascontiguousarray(bgr[..., ::-1])


def cell_sums(matrix):
    height, width = matrix.shape
    return matrix.reshape(height // CELL, CELL, width // CELL, CELL).sum(axis=(1, 3))


def stencil(rgb):
    values = rgb.astype(np.int32)
    height, width = rgb.shape[:2]
    channels = []
    for index in (0, 2):
        channel = values[..., index]
        orientations = []
        for axis in (1, 0):
            left = np.roll(channel, 1, axis=axis)
            right = np.roll(channel, -1, axis=axis)
            diff = np.abs(2 * channel - left - right)
            eligible = (np.abs(left - right) >= ENDPOINT_SEPARATION)
            eligible &= (np.minimum(left, right) > 1024) & (np.maximum(left, right) < 64511)
            eligible[[0, -1], :] = False
            eligible[:, [0, -1]] = False
            rates = []
            for parity in range(4):
                p = np.s_[parity // 2:height:2, parity % 2:width:2]
                evaluated = np.zeros_like(eligible)
                evaluated[p] = eligible[p]
                count = cell_sums(evaluated)
                matched = cell_sums(evaluated & (diff <= CLOSURE_TOLERANCE))
                rates.append(np.divide(matched, np.maximum(count, 1), where=count >= MIN_SAMPLES,
                                       out=np.full_like(count, np.nan, dtype=np.float64)))
            orientations.append(rates)
        channels.append(orientations)
    red, blue = channels
    return np.stack([np.mean([red[0][p ^ 1], red[1][p ^ 2], blue[0][p ^ 2], blue[1][p ^ 1]], axis=0)
                     for p in range(4)], axis=-1)


def learn(rates):
    cells = np.isfinite(rates).all(axis=-1)
    y, x = np.indices(cells.shape)
    folds = [rates[cells & ((x + y) % 2 == n)] for n in range(2)]
    if min(map(len, folds)) < 24:
        return None
    profiles = np.stack([np.median(fold, axis=0) for fold in folds])
    ranked = np.sort(profiles, axis=1)
    winners = np.argmax(profiles, axis=1)
    if winners[0] != winners[1] or ranked[:, -1].min() < FOLD_CLOSURE or (ranked[:, -1] - ranked[:, -2]).min() < PHASE_SEPARATION:
        return None
    return int(winners[0]), float(ranked[:, -1].min()), cells


def best_region(score, usable):
    score = np.where(usable, np.maximum(score, 0), 0)
    if not np.any(score):
        return {"score": 0., "box": None}
    rect = cv2.integral(score)
    support = cv2.integral(usable.astype(np.uint8))
    result = {"score": 0., "box": None}
    for h in REGION_SIDES:
        for w in REGION_SIDES:
            if max(h, w) > 2 * min(h, w):
                continue
            summed = rect[h:, w:] + rect[:-h, :-w] - rect[h:, :-w] - rect[:-h, w:]
            enough = support[h:, w:] + support[:-h, :-w] - support[h:, :-w] - support[:-h, w:]
            ranking = np.where(enough >= .65 * h * w, summed / np.sqrt(h * w), 0.)
            y, x = np.unravel_index(np.argmax(ranking), ranking.shape)
            if ranking[y, x] > result["score"]:
                result = {"score": float(ranking[y, x]), "box": [int(i) * CELL for i in (x, y, x+w, y+h)]}
    return result


def measure(rgb: np.ndarray) -> InterpolationReview:
    if rgb.dtype != np.uint16 or rgb.shape != (*SCOPE, 3):
        return InterpolationReview("unsupported_png")
    rates = stencil(rgb)
    reference = learn(rates)
    if reference is None:
        return InterpolationReview("no_repeatable_stencil")
    phase, closure, eligible = reference
    observed = rates[..., phase]
    strongest_other = np.max(rates[..., [p for p in range(4) if p != phase]], axis=-1)
    proposal = best_region(strongest_other - observed - .20, eligible & (strongest_other >= .55))
    box = tuple(proposal["box"]) if proposal["box"] else None
    selected_digest = hashlib.sha256(memoryview(rgb)).hexdigest() if box and proposal["score"] > POINT else None
    return InterpolationReview("assessed", phase, proposal["score"], box, int(eligible.sum()), closure, selected_digest)


def inspect_png(body: bytes | memoryview) -> InterpolationReview:
    rgb = decoded_rgb(body)
    if rgb is None:
        return InterpolationReview("unsupported_png")
    return measure(rgb)
