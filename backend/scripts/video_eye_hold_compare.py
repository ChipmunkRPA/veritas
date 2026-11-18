# SPDX-License-Identifier: MIT
"""Measure a temporarily stable textured eye patch while nearby pixels change."""
from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces
from app.detectors.video import mouth_local_hold as mouth


def landmarks(frame, detector):
    faces = detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detector=detector, min_size=mouth.FACE_MINIMUM)
    if len(faces) != 1:
        return None, len(faces)
    eyes = sorted(faces[0].landmarks[:2])
    a, b = eyes
    scale = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if not 23 <= scale <= 176 or abs(a[1] - b[1]) > .4 * scale:
        return None, 1
    return (eyes, scale), 1


def geometry(frames, detector):
    sampled = [landmarks(frames[index], detector) for index in mouth.ANCHORS]
    points = [point for point, _ in sampled if point]
    if len(points) < 4 or any(count > 1 for _, count in sampled):
        return None
    eyes = np.asarray([point[0] for point in points])
    pivot = np.median(eyes, axis=0)
    scale = float(np.median([point[1] for point in points]))
    if np.max(np.abs(eyes - pivot)) > .48 * scale:
        return None
    return pivot, scale


def centered_box(x, y, width, height, shape):
    w, h = round(width), round(height)
    left, top = round(x - w / 2), round(y - h / 2)
    if w >= 8 and h >= 8 and 0 <= left <= shape[1] - w and 0 <= top <= shape[0] - h:
        return left, top, w, h
    return None


def candidates(eyes, scale, side, shape):
    x, y = eyes[side]
    offset = max(2, round(.105 * scale))
    selection = [
        centered_box(x + dx, y + dy, w * scale, h * scale, shape)
        for w, h in ((.35, .25), (.43, .31))
        for dx in (-offset, 0, offset)
        for dy in (-offset, 0, offset)
    ]
    return np.asarray([box for box in selection if box], dtype=np.int32)


def context(eyes, scale, side, shape):
    left, right = eyes
    other_x, other_y = eyes[1 - side]
    points = [(other_x, other_y), ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 + .49 * scale),
              (left[0], left[1] + .77 * scale), (right[0], right[1] + .77 * scale)]
    return np.asarray([box for x, y in points
                       if (box := centered_box(x, y, .40 * scale, .31 * scale, shape))], dtype=np.int32)


def _eye_at_anchor(box, landmarks_value, side):
    if landmarks_value is None:
        return False
    eyes, scale = landmarks_value
    x, y, w, h = box
    return abs(x + w / 2 - eyes[side][0]) <= .31 * scale and abs(y + h / 2 - eyes[side][1]) <= .26 * scale


def observe(frames, detector):
    stable = geometry(frames, detector)
    if stable is None:
        return "no_stable_face", None
    eyes, scale = stable
    choices, reviewed = [], False
    shape = frames[0].shape[:2]
    for side in (0, 1):
        boxes, nearby = candidates(eyes, scale, side, shape), context(eyes, scale, side, shape)
        if len(boxes) == 0 or len(nearby) < 3:
            continue
        reviewed = True
        texture, core, quiet, neighbors, whole = mouth.transitions(frames, boxes, nearby)
        choices.extend(row | {"eye": side, "context_boxes": nearby.tolist()}
                       for row in mouth._candidates(texture, boxes, core, quiet, neighbors, whole))
    known = {}
    for row in sorted(choices, key=mouth._rank)[:8]:
        for index in row["anchors"]:
            if index not in known:
                known[index], _ = landmarks(frames[index], detector)
        if all(_eye_at_anchor(row["box"], known[key], row["eye"]) for key in row["anchors"]):
            return "matched", row
    return "no_match" if reviewed else "no_stable_face", None
