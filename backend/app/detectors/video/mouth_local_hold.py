"""Single-upload measurement of a temporarily stable native mouth patch."""

from __future__ import annotations

from collections import namedtuple
from contextlib import suppress
from fractions import Fraction
from hashlib import file_digest
from itertools import pairwise

import cv2
import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import mouth_local_hold_binding as binding

STEP, PROFILE, FRAME_COUNT = binding.STEP, binding.PROFILE, binding.FRAME_COUNT
ANCHORS = (4, 12, 20, 28, 36, 44)
FACE_MINIMUM = 56
MIN_SPAN = 7
MAX_SPAN = 20
FLANK = 4
CORE_MAX_MAE = .42
CORE_MIN_FRACTION = .94
CONTEXT_MAE = .75
CONTEXT_MIN_FRACTION = Fraction(2, 3)
FLANK_PEAK = 2.4
WHOLE_MAX = 19
TEXTURE_MIN_STD = 8
FACE_WEIGHT = "face_detection_yunet_2023mar.onnx"
FACE_WEIGHT_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
Geometry = namedtuple("Geometry", "x y width")


def source_geometry(rgb, detector):
    faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=FACE_MINIMUM)
    if len(faces) != 1:
        return None, len(faces)
    (x0, y0), (x1, y1) = faces[0].landmarks[3:5]
    return Geometry((x0 + x1) / 2, (y0 + y1) / 2, float(np.hypot(x1 - x0, y1 - y0))), 1


def stable_geometry(frames, detector):
    sampled = [source_geometry(frames[i], detector) for i in ANCHORS]
    values = np.asarray([geometry for geometry, count in sampled if count == 1])
    if len(values) < 4 or any(count > 1 for _, count in sampled):
        return None
    mouth = Geometry(*np.median(values, axis=0))
    if not 18 <= mouth.width <= 96 or np.max(np.abs(values[:, :2] - [mouth.x, mouth.y])) > .70 * mouth.width:
        return None
    return mouth


def boxes(mouth, shape):
    height, width = shape
    rectangles = []
    radius_x, radius_y = (max(4, round(mouth.width * ratio / 4) * 4) for ratio in (.36, .24))
    for w_ratio, h_ratio in ((.50, .35), (.68, .46)):
        w, h = (max(8, round(mouth.width * ratio / 2) * 2) for ratio in (w_ratio, h_ratio))
        for y in range(round(mouth.y) - radius_y, round(mouth.y) + radius_y + 1, 4):
            for x in range(round(mouth.x) - radius_x, round(mouth.x) + radius_x + 1, 4):
                if x - w >= 0 and x + w <= width and y - h >= 0 and y + h <= height:
                    rectangles.append((x - w // 2, y - h // 2, w, h))
    return np.asarray(rectangles, dtype=np.int32)


def _box_sum(integral, x, y, w, h):
    return integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]


def context_regions(mouth, shape):
    height, width = shape
    w, h = round(.23 * mouth.width), round(.24 * mouth.width)
    regions = []
    for dx, dy in ((-1.05, -.35), (1.05, -.35), (-.55, -1.03), (.55, -1.03), (0, .84)):
        x, y = round(mouth.x + dx * mouth.width), round(mouth.y + dy * mouth.width)
        if x - w >= 0 and x + w <= width and y - h >= 0 and y + h <= height:
            regions.append((x - w, y - h, 2 * w, 2 * h))
    return np.asarray(regions, dtype=np.int32)


def transitions(frames, rectangles, regions):
    x, y, w, h = rectangles.T
    area = w * h
    cx, cy, cw, ch = regions.T
    texture = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    core, quiet, context, whole = [], [], [], []
    for first, second in pairwise(texture):
        change = cv2.absdiff(first, second)
        integral = cv2.integral(change, sdepth=cv2.CV_32S)
        inner = _box_sum(integral, x, y, w, h)
        face_regions = _box_sum(integral, cx, cy, cw, ch) / (cw * ch)
        equal = cv2.integral(np.uint8(change <= 1), sdepth=cv2.CV_32S)
        core.append(inner / area)
        quiet.append(_box_sum(equal, x, y, w, h) / area)
        context.append(np.sort(face_regions)[-2])
        whole.append(float(change.mean()))
    return texture, np.array(core), np.array(quiet), np.array(context), np.array(whole)


def _candidates(texture, rectangles, core, quiet, context, whole):
    steady = (core <= CORE_MAX_MAE) & (quiet >= CORE_MIN_FRACTION)
    for i, rect in enumerate(rectangles):
        x, y, w, h = (int(value) for value in rect)
        edges = np.flatnonzero(np.diff(np.r_[False, steady[:, i], False]))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            end = stop + 1
            length = end - start
            if not MIN_SPAN <= length <= MAX_SPAN or start < FLANK or end + FLANK > len(texture):
                continue
            if whole[start - 1:stop + 1].max() > WHOLE_MAX:
                continue
            moving = context[start:stop]
            active = int(np.count_nonzero(moving >= CONTEXT_MAE))
            if Fraction(active, len(moving)) < CONTEXT_MIN_FRACTION:
                continue
            before, after = core[start - FLANK:start, i], core[stop:stop + FLANK, i]
            if min(before.max(), after.max()) < FLANK_PEAK:
                continue
            middle = (start + end) // 2
            if np.std(texture[middle][y:y + h, x:x + w]) < TEXTURE_MIN_STD:
                continue
            yield {"frames": [int(start), int(end)], "box": [x, y, w, h], "anchors": [int(start - 1), int(middle), int(end)],
                   "maximum_patch_mean_difference": float(core[start:stop, i].max()),
                   "minimum_patch_near_equal_fraction": float(quiet[start:stop, i].min()),
                   "moving_context_edges": active, "total_edges": int(len(moving)),
                   "minimum_flank_peak": float(min(before.max(), after.max()))}


def _mouth_at_anchor(rect, frame, detector):
    mouth, count = source_geometry(frame, detector)
    if count != 1:
        return False
    x, y, w, h = rect
    return (abs(x + w / 2 - mouth.x) <= .53 * mouth.width
            and abs(y + h / 2 - mouth.y) <= .33 * mouth.width)


def inspect(frames, detector):
    mouth = stable_geometry(frames, detector)
    if mouth is None:
        return "no_stable_face", None
    rectangles = boxes(mouth, frames[0].shape[:2])
    regions = context_regions(mouth, frames[0].shape[:2])
    if rectangles.size == 0 or len(regions) < 2:
        return "no_stable_face", None
    texture, core, quiet, context, whole = transitions(frames, rectangles, regions)
    candidates = sorted(_candidates(texture, rectangles, core, quiet, context, whole),
                        key=lambda row: (-row["total_edges"], -row["box"][2] * row["box"][3], row["frames"][0], row["box"]))
    for row in candidates[:8]:
        if all(_mouth_at_anchor(row["box"], frames[index], detector) for index in row["anchors"]):
            return "matched", row | {"context_boxes": regions.tolist()}
    return "no_match", None


def native(path):
    table = probe_frames(path, cap=FRAME_COUNT, timeout_s=8)
    if (len(table) != FRAME_COUNT or table.interpolated or table.truncated or table.rotation
            or (table.width, table.height) not in PROFILE or any((b - a) * table.tb != STEP for a, b in pairwise(table.pts))):
        raise ValueError("The delivery differs from the frozen native picture profile")
    delivered = list(decode_frames(path, table, None, timeout_s=8))
    if len(delivered) != FRAME_COUNT or any(
            frame.index != i or frame.pts != table.pts[i] or not frame.pts_verified for i, frame in enumerate(delivered)):
        raise ValueError("Not all native pictures and timestamps could be verified")
    return table, [frame.rgb for frame in delivered]


def record(frames, pts, hashes, *, detector_path=None):
    status, match = "unavailable", None
    if detector_path is not None:
        with suppress(cv2.error):  # This auxiliary measurement cannot fail the analysis.
            status, match = inspect(frames, load_yunet(detector_path))
    return {"schema": binding.SCHEMA, "status": status, "frames_scanned": FRAME_COUNT,
            "encoding": list(binding.ENCODING), "pts": list(pts), "frame_hashes": hashes,
            "input_sha256": canonical_hash(hashes), "packet_id": binding.PACKET_ID if match else None,
            "affects_overall_confidence": False, "match": match}


def local_detector_path():
    from app.core.config import settings

    path = settings.weights_dir / FACE_WEIGHT
    try:
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() == FACE_WEIGHT_SHA256:
                return path
    except OSError:
        pass
    return None


def verify_measurements(frames, match):
    """Recompute the recorded pixel measurements without the face detector."""
    if match is None:
        return
    rectangles = np.asarray([match["box"]])
    regions = np.asarray(match["context_boxes"])
    texture, core, quiet, context, whole = transitions(frames, rectangles, regions)
    expected = canonical_hash({key: value for key, value in match.items() if key != "context_boxes"})
    if not any(canonical_hash(candidate) == expected for candidate in _candidates(texture, rectangles, core, quiet, context, whole)):
        raise ValueError("The native pictures do not reproduce the held mouth-region measurements")
