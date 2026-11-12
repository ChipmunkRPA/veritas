# SPDX-License-Identifier: Apache-2.0
"""Verify an earlier image within a larger, lossless, native-scale PNG screenshot."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from app.detectors.image import reference_compare
from app.detectors.image import reference_png_exact_resize as exact

MIN_SIDE = 160
MAX_REFERENCE_SIDE = 1024
MAX_SCREEN_SIDE = 1280
MIN_MARGIN = 8
SCREEN_FRACTION = (.16, .73)
ANCHOR_GRID = 12
ANCHOR_VOTES = 16
MAX_OFFSETS = 16
SAMPLE_SPAN = 12
AXIS_AGREEMENT = .66
STRICT_INTERIOR_AGREEMENT = .90
CORROBORATED_INTERIOR_AGREEMENT = .75
EDGE_AGREEMENT = .80
MAX_OUTSIDE_AGREEMENT = .16


def _fingerprints(rgb):
    packed = rgb.astype(np.uint64)
    packed = (packed[:, :, 0] << 16) | (packed[:, :, 1] << 8) | packed[:, :, 2]
    n = SAMPLE_SPAN
    a, b, c = packed[:-n, :-n], packed[5:5 + packed.shape[0] - n, n:], packed[n:, 2:2 + packed.shape[1] - n]
    return (a * np.uint64(0x9E3779B185EBCA87)) ^ (b * np.uint64(0xC2B2AE3D27D4EB4F)) ^ c


def _proposals(reference, screen):
    prior, later = _fingerprints(reference), _fingerprints(screen)
    anchors = prior[::ANCHOR_GRID, ::ANCHOR_GRID]
    keys, indices, counts = np.unique(anchors, return_index=True, return_counts=True)
    indices, keys = indices[counts == 1], keys[counts == 1]
    if len(keys) < ANCHOR_VOTES:
        return ()
    choices = np.searchsorted(keys, later).clip(max=len(keys) - 1)
    sy, sx = np.nonzero(keys[choices] == later)
    ry, rx = np.unravel_index(indices[choices[sy, sx]], anchors.shape)
    offsets = np.column_stack((sx - rx * ANCHOR_GRID, sy - ry * ANCHOR_GRID))
    translations, votes = np.unique(offsets, axis=0, return_counts=True)
    supported = translations[votes >= ANCHOR_VOTES]
    if len(supported) > MAX_OFFSETS:
        return ()
    return [(int(dx), int(dy), sy[(offsets == (dx, dy)).all(axis=1)])
            for dx, dy in supported]


def _span(fractions):
    good = np.flatnonzero(fractions >= AXIS_AGREEMENT)
    return (int(good[0]), int(good[-1] + 1)) if len(good) else (0, 0)


def _bounds(equal, ay, minimum_interior):
    left, right = _span(equal[ay.min():ay.max() + SAMPLE_SPAN + 1].mean(axis=0))
    if right - left < MIN_SIDE:
        return None
    top, bottom = _span(equal[:, left:right].mean(axis=1))
    if bottom - top < MIN_SIDE:
        return None
    left, right = _span(equal[top:bottom].mean(axis=0))
    if right - left < MIN_SIDE:
        return None
    interior = equal[top:bottom, left:right]
    inside_edges = (interior[0], interior[-1], interior[:, 0], interior[:, -1])
    outside_edges = (equal[max(0, top - 4):top, left:right], equal[bottom:bottom + 4, left:right],
                     equal[top:bottom, max(0, left - 4):left], equal[top:bottom, right:right + 4])
    if (interior.mean() < minimum_interior or any(edge.mean() < EDGE_AGREEMENT for edge in inside_edges)
            or any(edge.size and edge.mean() > MAX_OUTSIDE_AGREEMENT for edge in outside_edges)):
        return None
    return left, top, right, bottom


def _verify(reference, screen, proposal, minimum_interior):
    dx, dy, ay = proposal
    rh, rw = reference.shape[:2]
    sh, sw = screen.shape[:2]
    sx, sy, right, bottom = max(0, dx), max(0, dy), min(sw, rw + dx), min(sh, rh + dy)
    if min(right - sx, bottom - sy) < MIN_SIDE:
        return None
    earlier = reference[sy-dy:bottom-dy, sx-dx:right-dx]
    later = screen[sy:bottom, sx:right]
    match = _bounds(np.all(earlier == later, axis=2), ay - sy, minimum_interior)
    if match is None:
        return None
    x, y, right, bottom = match
    w, h = right - x, bottom - y
    if (not SCREEN_FRACTION[0] <= w * h / (sw * sh) <= SCREEN_FRACTION[1]
            or min(sx + x, sy + y, sw - sx - right, sh - sy - bottom) < MIN_MARGIN):
        return None
    earlier, later = earlier[y:bottom, x:right], later[y:bottom, x:right]
    support = exact._supported(earlier, later, exact._textured_tiles(later))
    if support is None:
        return None
    equal, witnesses = support
    bounds = {"x": sx + x, "y": sy + y, "w": w, "h": h}
    return bounds, dx, dy, equal, witnesses


def compare(reference: np.ndarray, screen: np.ndarray, carrier: Image.Image,
            *, minimum_interior: float = CORROBORATED_INTERIOR_AGREEMENT):
    """Return one exact native placement and report compact differences inside it."""
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1
            or int(carrier.getexif().get(274, 1)) != 1 or min(*reference.shape[:2], *screen.shape[:2]) < MIN_SIDE
            or max(reference.shape[:2]) > MAX_REFERENCE_SIDE or max(screen.shape[:2]) > MAX_SCREEN_SIDE):
        return None, None
    accepted = None
    for proposal in _proposals(reference, screen):
        match = _verify(reference, screen, proposal, minimum_interior)
        if match is None:
            continue
        if accepted is not None:
            return None, None
        accepted = match
    if accepted is None:
        return None, None

    bounds, dx, dy, equal, witnesses = accepted
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
                                      "exact_tiles": [list(cell) for cell in witnesses],
                                      "native_difference_sha256": hashlib.sha256((~equal).tobytes()).hexdigest()}}, mask
