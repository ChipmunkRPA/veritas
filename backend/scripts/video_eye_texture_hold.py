# SPDX-License-Identifier: MIT
"""Compare eye-texture continuity independently of local brightness and contrast."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.detectors.video import mouth_local_hold as mouth
from scripts import video_eye_hold_compare as eye

POINTS = (.045, .075, .115)
MIN_TRANSITION = .105
MAX_RELATIVE = .40


def _texture_sequence(residual, box):
    x, y, w, h = box
    patches = np.stack([frame[y:y + h, x:x + w].ravel() for frame in residual])
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    return patches / np.maximum(norms, .01)


def _continuous(whole, start, stop):
    section = whole[max(0, start - 6):stop + 5]
    if np.any(np.all(sliding_window_view(section < .22, 4), axis=1)):
        return False
    interior = whole[start:stop - 1]
    flank_motion = max(whole[start - 2:start + 1].max(), whole[stop - 2:stop + 1].max())
    return flank_motion <= max(2.4, 2.65 * np.median(interior))


def _candidates(texture, context, whole, point, guard):
    size = len(texture)
    change = np.maximum(0., 1. - np.sum(texture[:-1] * texture[1:], axis=1))
    for length in range(8, 19):
        starts = np.arange(4, size - length - 3)
        stops = starts + length
        internal = sliding_window_view(change, length - 1)[starts]
        quiet = np.quantile(internal, .8, axis=1)
        local_motion = np.mean(sliding_window_view(context, length - 1)[starts] >= mouth.CONTEXT_MAE, axis=1)
        wide_motion = np.max(sliding_window_view(whole, length + 1)[starts - 1], axis=1)
        at_boundary = sliding_window_view(change, 3).max(axis=1)
        flank = np.minimum(at_boundary[starts - 2], at_boundary[stops - 2])
        candidates = ((quiet <= point) & (flank >= MIN_TRANSITION) & (quiet <= MAX_RELATIVE * flank)
                      & (local_motion >= float(mouth.CONTEXT_MIN_FRACTION)) & (wide_motion <= mouth.WHOLE_MAX))
        for index in np.flatnonzero(candidates):
            a, b = int(starts[index]), int(stops[index])
            middle = (a + b) // 2
            # A long window of small steps can still accumulate large motion.
            agreement = np.maximum(0., 1 - texture[a:b] @ texture[middle])
            accumulated = float(np.quantile(agreement, .8))
            if accumulated <= point and (not guard or _continuous(whole, a, b)):
                yield a, b, float(quiet[index]), float(flank[index]), accumulated, float(local_motion[index])


def _rank(row):
    moving = row["minimum_flank_texture_change"]
    stable = max(row["internal_texture_change"], row["midpoint_texture_change"])
    start, stop = row["frames"]
    return -(stop - start), stable / moving, -row["box"][2] * row["box"][3], start, row["box"]


def observe(frames, detector, point, *, guard=False):
    stable = eye.geometry(frames, detector)
    if stable is None:
        return "no_stable_face", None
    eyes, scale = stable
    choices, covered = [], False
    shape = frames[0].shape[:2]
    for side in (0, 1):
        boxes, nearby = eye.candidates(eyes, scale, side, shape), eye.context(eyes, scale, side, shape)
        if len(boxes) == 0 or len(nearby) < 3:
            continue
        covered = True
        gray, _, _, motion, whole = mouth.transitions(frames, boxes, nearby)
        floating = [frame.astype(np.float32) for frame in gray]
        residual = np.stack([frame - cv2.GaussianBlur(frame, (3, 3), .8) for frame in floating])
        for bounds in boxes.tolist():
            x, y, w, h = bounds
            texture = _texture_sequence(residual, bounds)
            for start, stop, quiet, flank, accumulated, support in _candidates(texture, motion, whole, point, guard):
                center = (start + stop) // 2
                if np.std(gray[center][y:y + h, x:x + w]) < mouth.TEXTURE_MIN_STD:
                    continue
                choices.append({"frames": [start, stop], "box": bounds, "anchors": [start - 1, center, stop],
                                "eye": side, "context_boxes": nearby.tolist(),
                                "internal_texture_change": quiet, "midpoint_texture_change": accumulated,
                                "minimum_flank_texture_change": flank, "moving_context_fraction": support})
    known = {}
    for row in sorted(choices, key=_rank)[:8]:
        for index in row["anchors"]:
            if index not in known:
                known[index], _ = eye.landmarks(frames[index], detector)
        if all(eye._eye_at_anchor(row["box"], known[key], row["eye"]) for key in row["anchors"]):
            return "matched", row
    return "no_match" if covered else "no_stable_face", None
