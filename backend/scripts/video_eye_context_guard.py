# SPDX-License-Identifier: MIT
"""Measure source texture just outside a proposed eye hold."""
from __future__ import annotations

import cv2
import numpy as np

from scripts import video_eye_hold_compare as eye
from scripts import video_eye_texture_hold as texture


def collar(box, shape):
    x, y, w, h = box
    dx, dy = max(3, round(w * .49)), max(3, round(h * .58))
    gap_x, gap_y = round(w * .23), round(h * .26)
    bounds = [
        [x - gap_x - dx, y, dx, h],
        [x + w + gap_x, y, dx, h],
        [x, y - gap_y - dy, w, dy],
        [x, y + h + gap_y, w, dy],
    ]
    if any(a < 0 or b < 0 or a + c > shape[1] or b + d > shape[0] for a, b, c, d in bounds):
        return None
    return bounds


def _difference(residual, box):
    vectors = texture._texture_sequence(residual, box)
    return np.clip(1. - np.sum(vectors[1:] * vectors[:-1], axis=1), 0., 2.)


def _edge_peak(values, start, stop):
    return tuple(float(values[index - 2:index + 1].max()) for index in (start, stop))


def measure(frames, detector, proposal):
    if proposal is None:
        return None
    geometry = eye.geometry(frames, detector)
    if geometry is None:
        return None
    eyes, _ = geometry
    side = proposal["eye"]
    x, y, w, h = proposal["box"]
    shape = frames[0].shape[:2]
    neighbors = collar((x, y, w, h), shape)
    shift_x, shift_y = eyes[1 - side] - eyes[side]
    other = eye.centered_box(x + w / 2 + shift_x, y + h / 2 + shift_y, w, h, shape)
    if neighbors is None or other is None:
        return None

    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in frames]
    residual = np.stack([frame - cv2.GaussianBlur(frame, (3, 3), .8) for frame in gray])
    start, stop = proposal["frames"]
    interior = slice(start, stop - 1)
    core = _difference(residual, proposal["box"])
    opposite = _difference(residual, other)
    adjacent = [_difference(residual, region) for region in neighbors]
    typical = [float(np.median(values[interior])) for values in adjacent]
    inside = float(np.median(core[interior]))
    collar_gap = min(max(typical[:2]), max(typical[2:])) - inside
    opposite_gap = min(a - b for a, b in zip(_edge_peak(core, start, stop), _edge_peak(opposite, start, stop), strict=True))
    return {"collar_gap": collar_gap, "opposite_gap": opposite_gap, "inside_median": inside,
            "collar_medians": typical, "collar_boxes": neighbors, "opposite_box": list(other)}
