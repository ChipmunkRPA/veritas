"""Compare brightness on the three displayed, source-bound face review frames."""

from __future__ import annotations

import math
from itertools import combinations

from app.evidence.face_interval_binding import APPEARANCE_METHOD

METHOD = "display-frame-brightness/1"
ROLES = ("before", "during", "after")
FIELDS = {"face": "Face", "scene": "Scene outside the face"}
LABELS = {f"{name} brightness change from {role}" for name in FIELDS.values() for role in ("before", "after")}
MAX_GRID_SIDE = 384
MIN_FACE_SITES, MIN_SCENE_SITES = 9, 64
VISIBLE_SHIFT = 100 * 12 / 255


def _extent(packet, source):
    width, height = source["width"], source["height"]
    frames = packet["frames"]
    if [frame.get("interval_role") for frame in frames] != list(ROLES):
        raise ValueError("Brightness context requires the displayed before, during and after frames")
    rectangles = []
    for frame in frames:
        box = frame["context_region"]
        x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
        if not all(type(value) is int for value in (x, y, w, h)) or not (
            0 <= x < x + w <= width and 0 <= y < y + h <= height
        ):
            raise ValueError("Brightness context requires a valid displayed face region")
        rectangles.append((x, y, x + w, y + h))
    return frames, rectangles, max(1, (max(width, height) + MAX_GRID_SIDE - 1) // MAX_GRID_SIDE)


def _site_count(rectangles, stride):
    left = max(rect[0] for rect in rectangles)
    top = max(rect[1] for rect in rectangles)
    right = min(rect[2] for rect in rectangles)
    bottom = min(rect[3] for rect in rectangles)
    return max(0, (right - 1) // stride - (left - 1) // stride) * max(
        0, (bottom - 1) // stride - (top - 1) // stride
    )


def _binding(packet, source):
    frames, rectangles, stride = _extent(packet, source)
    excluded = sum(
        (-1) ** (size + 1) * sum(_site_count(group, stride) for group in combinations(rectangles, size))
        for size in range(1, len(rectangles) + 1)
    )
    scene = _site_count([(0, 0, source["width"], source["height"])], stride) - excluded
    rows = []
    for frame, rect in zip(frames, rectangles, strict=True):
        anchor = frame["anchor"]
        rows.append(
            {
                "role": frame["interval_role"],
                "frame_index": anchor["frame_index"],
                "decoded_rgb_sha256": anchor["decoded_rgb_sha256"],
                "face_sites": _site_count([rect], stride),
                "scene_sites": scene,
            }
        )
    status = (
        "measured"
        if scene >= MIN_SCENE_SITES and all(row["face_sites"] >= MIN_FACE_SITES for row in rows)
        else "insufficient_coverage"
    )
    return {"method": METHOD, "status": status, "stride": stride, "frames": rows}, rectangles


def record(packet, source, rgb_by_index):
    import numpy as np

    binding, rectangles = _binding(packet, source)
    if binding["status"] != "measured":
        return binding
    width, height, step = source["width"], source["height"], binding["stride"]
    yy, xx = np.ogrid[:height:step, :width:step]
    faces = [
        (left <= xx) & (xx < right) & (top <= yy) & (yy < bottom) for left, top, right, bottom in rectangles
    ]
    scene = ~np.logical_or.reduce(faces)
    for row, face in zip(binding["frames"], faces, strict=True):
        rgb = np.asarray(rgb_by_index[row["frame_index"]])
        if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise ValueError("Brightness context requires native RGB pixels")
        red, green, blue = rgb[::step, ::step].astype(np.uint16).transpose(2, 0, 1)
        luma = 54 * red + 183 * green + 19 * blue
        for key, mask in (("face", face), ("scene", scene)):
            if int(mask.sum()) != row[f"{key}_sites"]:
                raise ValueError("Brightness sample locations differ from their source binding")
            row[key] = round(100 * float(luma[mask].mean()) / (256 * 255), 2)
    return binding


def measurements(binding):
    if binding["status"] != "measured":
        return []
    before, during, after = binding["frames"]
    return [
        {
            "label": f"{name} brightness change from {flank['role']}",
            "value": round(during[key] - flank[key], 2),
            "unit": "points on a 0–100 scale",
        }
        for flank in (before, after)
        for key, name in FIELDS.items()
    ]


def explanation(binding):
    values = [row["value"] for row in measurements(binding)]
    if not values:
        return (
            " The displayed frames had insufficient face or surrounding pixels for a brightness comparison."
        )
    if all(value >= VISIBLE_SHIFT for value in values):
        return " The face and surrounding scene both brightened in the displayed middle frame."
    if all(value <= -VISIBLE_SHIFT for value in values):
        return " The face and surrounding scene both darkened in the displayed middle frame."
    return " The brightness measurements compare the face and surrounding scene across the three displayed frames."


def validate(packet, source):
    recorded = packet.get("face_lighting")
    expected_measurements = []
    if (packet.get("face_sequence") or {}).get("method") == APPEARANCE_METHOD:
        expected, _ = _binding(packet, source)
        if (
            not isinstance(recorded, dict)
            or set(recorded) != set(expected)
            or len(recorded["frames"]) != len(ROLES)
        ):
            raise ValueError("Missing or invalid face brightness context")
        rows = []
        for row in recorded["frames"]:
            copied = dict(row)
            if expected["status"] == "measured":
                for field in FIELDS:
                    value = copied.pop(field, None)
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError("Invalid measured brightness")
            rows.append(copied)
        if {**recorded, "frames": rows} != expected:
            raise ValueError("Brightness context differs from the displayed frame binding")
        expected_measurements = measurements(recorded)
    elif recorded is not None:
        raise ValueError("Unexpected face brightness context")
    actual = [row for row in packet.get("measurements", []) if row.get("label") in LABELS]
    if actual != expected_measurements:
        raise ValueError("Brightness measurements differ from the recorded values")
