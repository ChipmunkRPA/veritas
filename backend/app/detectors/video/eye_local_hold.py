# SPDX-License-Identifier: MIT
"""Observe a stable eye texture with independently moving nearby picture regions."""
from __future__ import annotations

import time
from contextlib import suppress
from itertools import pairwise

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import mouth_local_hold as mouth
from app.evidence import eye_local_hold_binding as binding

GEOMETRY_ANCHORS = (5, 14, 24, 34, 43, 53)
SHORTLIST = 8


def _landmarks(frame, detector):
    faces = detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detector=detector, min_size=mouth.FACE_MINIMUM)
    if len(faces) != 1:
        return None, len(faces)
    left, right = sorted(faces[0].landmarks[:2])
    separation = float(np.hypot(right[0] - left[0], right[1] - left[1]))
    if not 23 <= separation <= 176 or abs(left[1] - right[1]) > .4 * separation:
        return None, 1
    return ([left, right], separation), 1


def _stable_geometry(frames, detector):
    sampled = [_landmarks(frames[index], detector) for index in GEOMETRY_ANCHORS]
    points = [point for point, _ in sampled if point is not None]
    if len(points) < 4 or any(count > 1 for _, count in sampled):
        return None
    eyes = np.asarray([point[0] for point in points])
    centers = np.median(eyes, axis=0)
    scale = float(np.median([point[1] for point in points]))
    if np.max(np.abs(eyes - centers)) > .48 * scale:
        return None
    return centers, scale


def _centered_box(x, y, width, height, shape):
    w, h = round(width), round(height)
    left, top = round(x - w / 2), round(y - h / 2)
    if min(w, h) >= 8 and 0 <= left <= shape[1] - w and 0 <= top <= shape[0] - h:
        return [left, top, w, h]
    return None


def _core_boxes(eyes, scale, side, shape):
    x, y = eyes[side]
    offset = max(2, round(.105 * scale))
    return [box for w, h in ((.35, .25), (.43, .31))
            for dx in (-offset, 0, offset) for dy in (-offset, 0, offset)
            if (box := _centered_box(x + dx, y + dy, w * scale, h * scale, shape)) is not None]


def _face_context(eyes, scale, side, shape):
    left, right = eyes
    points = [eyes[1 - side], ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 + .49 * scale),
              (left[0], left[1] + .77 * scale), (right[0], right[1] + .77 * scale)]
    return [box for x, y in points if (box := _centered_box(x, y, .40 * scale, .31 * scale, shape)) is not None]


def side_boxes(box, shape):
    x, y, w, h = box
    side_w, side_h = max(3, round(w * .49)), max(3, round(h * .58))
    x_gap, y_gap = round(w * .23), round(h * .26)
    boxes = [[x - x_gap - side_w, y, side_w, h], [x + w + x_gap, y, side_w, h],
             [x, y - y_gap - side_h, w, side_h], [x, y + h + y_gap, w, side_h]]
    if any(a < 0 or b < 0 or a + c > shape[1] or b + d > shape[0] for a, b, c, d in boxes):
        return None
    return boxes


def _within_eye(box, geometry, side):
    if geometry is None:
        return False
    eyes, scale = geometry
    x, y, w, h = box
    return (abs(x + w / 2 - eyes[side][0]) <= .31 * scale
            and abs(y + h / 2 - eyes[side][1]) <= .26 * scale)


def _texture(residual, box):
    x, y, w, h = box
    patches = np.stack([frame[y:y+h, x:x+w].ravel() for frame in residual])
    return patches / np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), .01)


def _allowed_picture_motion(whole, start, stop):
    nearby = whole[max(0, start - 7):stop + 6]
    if np.any(np.all(sliding_window_view(nearby < .22, 5), axis=1)):
        return False
    interior = whole[start:stop - 1]
    strongest_flank = max(whole[start-3:start+1].max(), whole[stop-3:stop+1].max())
    return strongest_flank <= max(2.4, 2.65 * np.median(interior))


def _prepared(frames):
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    floating = [frame.astype(np.float32) for frame in gray]
    residual = np.stack([frame - cv2.GaussianBlur(frame, (3, 3), .8) for frame in floating])
    changes = [cv2.absdiff(a, b) for a, b in pairwise(gray)]
    return gray, residual, changes, np.array([float(change.mean()) for change in changes])


def _context_motion(changes, boxes):
    measurements = [[change[y:y+h, x:x+w].mean() for x, y, w, h in boxes] for change in changes]
    return np.sort(np.array(measurements), axis=1)[:, -2]


def _regions(gray, residual, box, motion, whole):
    x, y, w, h = box
    texture = _texture(residual, box)
    change = np.maximum(0., 1. - np.sum(texture[:-1] * texture[1:], axis=1))
    edge_peaks = sliding_window_view(change, 4).max(axis=1)
    for length in binding.LENGTHS:
        starts = np.arange(5, len(texture) - length - 4)
        stops = starts + length
        calm = np.quantile(sliding_window_view(change, length - 1)[starts], .8, axis=1)
        context = np.mean(sliding_window_view(motion, length - 1)[starts] >= mouth.CONTEXT_MAE, axis=1)
        whole_flank = np.max(sliding_window_view(whole, length + 1)[starts - 1], axis=1)
        edge = np.minimum(edge_peaks[starts - 3], edge_peaks[stops - 3])
        qualified = ((calm <= binding.TEXTURE_POINT) & (edge >= binding.MIN_EDGE_CHANGE)
                     & (calm <= binding.MAX_RELATIVE_CHANGE * edge)
                     & (context >= float(binding.MIN_CONTEXT_FRACTION)) & (whole_flank <= mouth.WHOLE_MAX))
        for index in np.flatnonzero(qualified):
            start, stop = int(starts[index]), int(stops[index])
            center = (start + stop) // 2
            agreement = np.maximum(0., 1 - texture[start:stop] @ texture[center])
            accumulated = float(np.quantile(agreement, .8))
            if (accumulated <= binding.TEXTURE_POINT and _allowed_picture_motion(whole, start, stop)
                    and np.std(gray[center][y:y+h, x:x+w]) >= mouth.TEXTURE_MIN_STD):
                yield {"frames": [start, stop], "box": box, "anchors": [start - 1, center, stop],
                       "internal_texture_change": float(calm[index]), "midpoint_texture_change": accumulated,
                       "minimum_flank_texture_change": float(edge[index]), "moving_context_fraction": float(context[index])}


def _nearby_values(residual, start, stop, box, neighbors):
    medians = []
    for region in [box, *neighbors]:
        vectors = _texture(residual, region)
        change = np.clip(1. - np.sum(vectors[1:] * vectors[:-1], axis=1), 0., 2.)
        medians.append(float(np.median(change[start:stop - 1])))
    inside, left, right, above, below = medians
    return {"box_median": inside, "nearby_medians": medians[1:],
            "nearby_gap": min(max(left, right), max(above, below)) - inside}


def _rank(candidate):
    start, stop = candidate["frames"]
    _, _, w, h = candidate["box"]
    calm = max(candidate["internal_texture_change"], candidate["midpoint_texture_change"])
    return (start - stop, calm / candidate["minimum_flank_texture_change"], -w * h, start, candidate["box"])


def inspect(frames, detector, *, check=None):
    """Select the first of eight ranked regions with verified eye and surrounding motion."""
    if check is not None:
        check()
    geometry = _stable_geometry(frames, detector)
    if geometry is None:
        return "no_stable_face", None
    eyes, scale = geometry
    gray, residual, changes, whole = _prepared(frames)
    shape = gray[0].shape
    candidates = []
    covered = False
    for side in (0, 1):
        boxes = _core_boxes(eyes, scale, side, shape)
        neighbors = _face_context(eyes, scale, side, shape)
        if not boxes or len(neighbors) < 3:
            continue
        covered = True
        motion = _context_motion(changes, neighbors)
        for box in boxes:
            if check is not None:
                check()
            candidates.extend(row | {"eye": side, "context_boxes": neighbors}
                              for row in _regions(gray, residual, box, motion, whole))
    known = {}
    for row in sorted(candidates, key=_rank)[:SHORTLIST]:
        if check is not None:
            check()
        side = row["eye"]
        for index in row["anchors"]:
            if index not in known:
                known[index], _ = _landmarks(frames[index], detector)
        if not all(_within_eye(row["box"], known[index], side) for index in row["anchors"]):
            continue
        x, y, w, h = row["box"]
        neighbors = side_boxes(row["box"], shape)
        dx, dy = eyes[1 - side] - eyes[side]
        if neighbors is None or _centered_box(x + w/2 + dx, y + h/2 + dy, w, h, shape) is None:
            continue
        nearby = _nearby_values(residual, *row["frames"], row["box"], neighbors)
        if nearby["nearby_gap"] >= binding.NEARBY_POINT:
            return "matched", row | nearby
    return ("no_match" if covered else "no_stable_face"), None


def record(frames, pts, hashes, *, budget_s, check=None):
    """Run a bounded local measurement with the existing pinned face asset."""
    end = time.monotonic() + budget_s

    def check_budget():
        if time.monotonic() >= end:
            raise TimeoutError("Eye-region observation budget expired")
        if check is not None:
            check()

    with suppress(cv2.error, TimeoutError):
        path = mouth.local_detector_path()
        if path is not None:
            check_budget()
            status, match = inspect(frames, load_yunet(path), check=check_budget)
            return binding.record(status, pts=pts, hashes=hashes, match=match)
    return binding.record("unavailable")


def verify_measurements(frames, match):
    """Recompute a recorded region from the exact original RGB pictures."""
    gray, residual, changes, whole = _prepared(frames)
    box, (start, stop) = match["box"], match["frames"]
    motion = _context_motion(changes, match["context_boxes"])
    candidate = next((row for row in _regions(gray, residual, box, motion, whole)
                      if row["frames"] == [start, stop]), None)
    if candidate is None:
        raise ValueError("Eye region fails the recorded picture-motion and texture requirements")
    nearby = side_boxes(box, gray[0].shape)
    if nearby is None:
        raise ValueError("Eye region lacks its four nearby picture measurements")
    confirmed = candidate | _nearby_values(residual, start, stop, box, nearby)
    if confirmed["nearby_gap"] < binding.NEARBY_POINT or any(
            match[key] != expected if isinstance(expected, list) and key != "nearby_medians"
            else not np.allclose(match[key], expected, rtol=0, atol=.000002)
            for key, expected in confirmed.items()):
        raise ValueError("Eye-region measurements differ from the submitted video")
