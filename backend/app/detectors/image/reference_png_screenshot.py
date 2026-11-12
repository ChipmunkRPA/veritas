# SPDX-License-Identifier: Apache-2.0
"""Locate an earlier photo at native scale inside a PNG screenshot."""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
from PIL import Image

from app.detectors.image import reference_compare

SCHEMA = "chipmunk.image-reference-comparison/9"
MIN_SIDE = 160
MAX_REFERENCE_SIDE = 1024
MAX_SCREEN_SIDE = 1280
SCREEN_FRACTION = (.16, .73)
MIN_SCREEN_MARGIN = 8
ANCHOR_STRIDE = 12
ANCHOR_POINTS = ((0, 0), (5, 12), (12, 2))
MAX_SAMPLE_OFFSET = 12
WITNESS_GRID = 5
MIN_WITNESSES = 3
MIN_CELLS = 12
MIN_AXES = 4
MAX_OFFSETS = 16
AXIS_AGREEMENT = .30
MIN_EDGE_RUN = .18
MAX_OUTSIDE_AGREEMENT = .16
MIN_INTERIOR_AGREEMENT = .75
MAX_DIFFERENCE_RECTANGLE = .40


def _fingerprints(rgb):
    packed = rgb.astype(np.uint64)
    packed = (packed[:, :, 0] << 16) | (packed[:, :, 1] << 8) | packed[:, :, 2]
    h, w = (dimension - MAX_SAMPLE_OFFSET for dimension in packed.shape)
    a, b, c = (packed[y:y+h, x:x+w] for y, x in ANCHOR_POINTS)
    return (a * np.uint64(0x9E3779B185EBCA87)) ^ (b * np.uint64(0xC2B2AE3D27D4EB4F)) ^ c


def _proposals(reference, screen):
    anchors = _fingerprints(reference)[::ANCHOR_STRIDE, ::ANCHOR_STRIDE]
    keys, indices, counts = np.unique(anchors, return_index=True, return_counts=True)
    indices, keys = indices[counts == 1], keys[counts == 1]
    if len(keys) < MIN_CELLS * MIN_WITNESSES:
        return ()
    target = _fingerprints(screen)
    choices = np.searchsorted(keys, target).clip(max=len(keys) - 1)
    sy, sx = np.nonzero(keys[choices] == target)
    ry, rx = np.unravel_index(indices[choices[sy, sx]], anchors.shape)
    offsets = np.column_stack((sx - rx * ANCHOR_STRIDE, sy - ry * ANCHOR_STRIDE))
    translations, votes = np.unique(offsets, axis=0, return_counts=True)
    supported = translations[votes >= MIN_CELLS * MIN_WITNESSES]
    if len(supported) > MAX_OFFSETS:
        return ()
    proposals = []
    for dx, dy in supported:
        match = (offsets == (dx, dy)).all(axis=1)
        proposals.append((int(dx), int(dy), sy[match], sx[match]))
    return proposals


def _span(fractions):
    good = np.flatnonzero(fractions >= AXIS_AGREEMENT)
    return (int(good[0]), int(good[-1] + 1)) if len(good) else (0, 0)


def _edge_run(edge):
    breaks = np.flatnonzero(~np.pad(edge, 1))
    return (np.diff(breaks).max() - 1) / edge.size


def _bounds(equal, anchor_y):
    left, right = _span(equal[anchor_y.min():anchor_y.max() + MAX_SAMPLE_OFFSET + 1].mean(axis=0))
    if right - left < MIN_SIDE:
        return None
    top, bottom = _span(equal[:, left:right].mean(axis=1))
    if bottom - top < MIN_SIDE:
        return None
    left, right = _span(equal[top:bottom].mean(axis=0))
    if right - left < MIN_SIDE:
        return None
    inside = equal[top:bottom, left:right]
    edges = (inside[0], inside[-1], inside[:, 0], inside[:, -1])
    outside = (equal[max(0, top - 4):top, left:right], equal[bottom:bottom + 4, left:right],
               equal[top:bottom, max(0, left - 4):left], equal[top:bottom, right:right + 4])
    if (inside.mean() < MIN_INTERIOR_AGREEMENT or any(_edge_run(edge) < MIN_EDGE_RUN for edge in edges)
            or any(edge.size and edge.mean() > MAX_OUTSIDE_AGREEMENT for edge in outside)):
        return None
    return left, top, right, bottom


def _witnesses(equal, ys, xs):
    h, w = equal.shape
    inside = (xs >= 0) & (ys >= 0) & (xs + MAX_SAMPLE_OFFSET < w) & (ys + MAX_SAMPLE_OFFSET < h)
    ys, xs = ys[inside], xs[inside]
    # Confirm all three RGB samples so a fingerprint collision cannot count.
    exact = np.logical_and.reduce([equal[ys + y, xs + x] for y, x in ANCHOR_POINTS])
    rows = ys[exact] * WITNESS_GRID // h
    columns = xs[exact] * WITNESS_GRID // w
    counts = np.bincount(rows * WITNESS_GRID + columns, minlength=WITNESS_GRID ** 2).reshape(WITNESS_GRID, WITNESS_GRID)
    cells = [(int(row), int(column), int(counts[row, column])) for row, column in np.argwhere(counts >= MIN_WITNESSES)]
    middle = WITNESS_GRID // 2
    quadrants = {(row > middle, column > middle) for row, column, _ in cells if row != middle and column != middle}
    if (len(cells) < MIN_CELLS or len({row for row, _, _ in cells}) < MIN_AXES
            or len({column for _, column, _ in cells}) < MIN_AXES
            or quadrants != set(product(range(2), repeat=2))):
        return ()
    return cells


def _verify(reference, screen, proposal):
    dx, dy, ys, xs = proposal
    rh, rw = reference.shape[:2]
    sh, sw = screen.shape[:2]
    sx, sy, right, bottom = max(0, dx), max(0, dy), min(sw, rw + dx), min(sh, rh + dy)
    if min(right - sx, bottom - sy) < MIN_SIDE:
        return None
    earlier = reference[sy-dy:bottom-dy, sx-dx:right-dx]
    later = screen[sy:bottom, sx:right]
    all_equal = np.all(earlier == later, axis=2)
    match = _bounds(all_equal, ys - sy)
    if match is None:
        return None
    left, top, right, bottom = match
    w, h = right - left, bottom - top
    x, y = sx + left, sy + top
    if (not SCREEN_FRACTION[0] <= w * h / (sw * sh) <= SCREEN_FRACTION[1]
            or min(x, y, sw - x - w, sh - y - h) < MIN_SCREEN_MARGIN):
        return None
    equal = all_equal[top:bottom, left:right]
    cells = _witnesses(equal, ys - y, xs - x)
    if not cells:
        return None
    py, px = np.nonzero(~equal)
    if len(px) and (int(np.ptp(px)) + 1) * (int(np.ptp(py)) + 1) > MAX_DIFFERENCE_RECTANGLE * equal.size:
        return None
    return {"x": x, "y": y, "w": w, "h": h}, dx, dy, equal, cells


def compare(reference: np.ndarray, screen: np.ndarray, carrier: Image.Image):
    """Require one verified native placement and report compact differences within it."""
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1
            or carrier.getexif().get(274, 1) != 1 or min(*reference.shape[:2], *screen.shape[:2]) < MIN_SIDE
            or max(reference.shape[:2]) > MAX_REFERENCE_SIDE or max(screen.shape[:2]) > MAX_SCREEN_SIDE
            or reference.shape == screen.shape):
        return None, None
    accepted = None
    for proposal in _proposals(reference, screen):
        match = _verify(reference, screen, proposal)
        if match is not None:
            if accepted is not None:
                return None, None
            accepted = match
    if accepted is None:
        return None, None

    bounds, dx, dy, equal, cells = accepted
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    regions, cropped_mask, omitted = reference_compare._regions(~equal, max(48, round(equal.size * .0004)))
    mask = np.zeros(screen.shape[:2], dtype=bool)
    mask[y:y+h, x:x+w] = cropped_mask
    for region in regions:
        region["x"] += x
        region["y"] += y
        region["reference_region"] = {**{key: region[key] for key in ("w", "h")},
                                      "x": region["x"] - dx, "y": region["y"] - dy}
    return {"status": "comparable", "regions": regions, "additional_regions": omitted,
            "embedded_native_photo": {"submitted_bounds": bounds,
                                      "reference_bounds": {**bounds, "x": x - dx, "y": y - dy},
                                      "exact_pixels": int(equal.sum()),
                                      "witness_cells": [list(cell) for cell in cells],
                                      "native_difference_sha256": hashlib.sha256((~equal).tobytes()).hexdigest()}}, mask
