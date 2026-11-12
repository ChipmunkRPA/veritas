# SPDX-License-Identifier: Apache-2.0
"""Measure face travel against verified fixed-scene reference pixels."""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames
from app.evidence.face_interval_binding import validate_sequence_record

ROLES = ("before", "during", "after")
SCENE_WIDTH = 320
POLICY = {
    "maximum_scene_displacement_px_at_320": 6.0,
    "minimum_face_travel_box_widths": .65,
    "maximum_flank_gap_box_widths": .25,
    "minimum_roll_excursion_degrees": 10.0,
    "maximum_flank_roll_gap_degrees": 5.0,
    "minimum_eye_fraction": .20,
    "minimum_middle_frame_fraction": .60,
    "minimum_middle_roll_fraction": .50,
    "face_exclusion_box_margins": [.18, .12, .18, .45],
    "search_radius_at_320": 5,
    "background_tile_side_at_320": 22,
    "minimum_background_tile_gradient": 1.6,
    "minimum_background_tile_correlation": .92,
    "minimum_background_distinct_peak_gap": .02,
    "minimum_background_tile_count": 4,
    "minimum_background_width_fraction": .15,
    "minimum_background_height_fraction": .25,
}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _mask(rows, shape, scale):
    h, w = shape
    mask = np.full(shape, 255, np.uint8)
    left, top, right, bottom = POLICY["face_exclusion_box_margins"]
    for row in rows:
        x, y, bw, bh = (row["bbox"][k] for k in ("x", "y", "w", "h"))
        x0, x1 = max(0, math.floor((x - left * bw) * scale[0])), min(w, math.ceil((x + (1 + right) * bw) * scale[0]))
        y0, y1 = max(0, math.floor((y - top * bh) * scale[1])), min(h, math.ceil((y + (1 + bottom) * bh) * scale[1]))
        mask[y0:y1, x0:x1] = 0
    mask[round(h * .68):] = 0
    return mask


def _templates(reference, mask):
    side, margin = POLICY["background_tile_side_at_320"], POLICY["search_radius_at_320"]
    h, w = reference.shape
    output = []
    for y in range(margin, h - side - margin + 1, side + margin):
        for x in range(margin, w - side - margin + 1, side + margin):
            if not mask[y - margin:y + side + margin, x - margin:x + side + margin].all():
                continue
            tile = reference[y:y + side, x:x + side]
            values = tile.astype(np.float32)
            if min(float(np.diff(values, axis=axis).std()) for axis in (0, 1)) >= POLICY["minimum_background_tile_gradient"]:
                output.append((x, y, tile))
    return output


def _background_transform(templates, moving):
    side, margin = POLICY["background_tile_side_at_320"], POLICY["search_radius_at_320"]
    minimum = POLICY["minimum_background_tile_count"]
    base, match = [], []
    for x, y, template in templates:
        search = moving[y - margin:y + side + margin, x - margin:x + side + margin]
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, point = cv2.minMaxLoc(scores)
        ys, xs = np.ogrid[:scores.shape[0], :scores.shape[1]]
        distractors = ((xs - point[0])**2 + (ys - point[1])**2) >= 9
        if peak < POLICY["minimum_background_tile_correlation"] or peak - scores[distractors].max() < POLICY["minimum_background_distinct_peak_gap"]:
            continue
        base.append((x, y))
        match.append((x + point[0] - margin, y + point[1] - margin))
    if len(base) < minimum or len(base) < math.ceil(len(templates) * .5):
        return None
    base = np.asarray(base, dtype=np.float64)
    matrix, inliers = cv2.estimateAffinePartial2D(np.asarray(match, dtype=np.float64), base, method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.0, maxIters=500, confidence=.99)
    if matrix is None or inliers is None:
        return None
    supported = base[inliers.ravel() == 1]
    h, w = moving.shape
    if (len(supported) < minimum or len(supported) < math.ceil(len(base) * .75)
            or np.ptp(supported[:, 0]) < POLICY["minimum_background_width_fraction"] * w
            or np.ptp(supported[:, 1]) < POLICY["minimum_background_height_fraction"] * h):
        return None
    return matrix


def _roll_and_center(row, scale, matrix):
    x, y, width, height = (row["bbox"][k] for k in ("x", "y", "w", "h"))
    landmarks = np.asarray(row["landmarks"], dtype=np.float64)
    eye = landmarks[1] - landmarks[0]
    nose = landmarks[2] - landmarks[:2].mean(0)
    if np.linalg.norm(eye) < POLICY["minimum_eye_fraction"] * width or eye[0] <= 0 or np.linalg.det([eye, nose]) <= 0:
        return None
    direction = (matrix[:, :2] @ (eye * scale)) / scale
    center = matrix[:, :2] @ (np.asarray((x + .5 * width, y + .5 * height)) * scale) + matrix[:, 2]
    return float(center[0]), math.degrees(math.atan2(direction[1], direction[0]))


def _sustained(values, ref_a, ref_b, minimum):
    plus = np.count_nonzero((values >= ref_a + minimum) & (values >= ref_b + minimum))
    minus = np.count_nonzero((values <= ref_a - minimum) & (values <= ref_b - minimum))
    return max(int(plus), int(minus))


def compare(record, rgb_by_index):
    cv2.setRNGSeed(205)
    frame_records = {row["frame_index"]: row for row in record["frame_inputs"]}
    groups = {role: record["centers"][role] for role in ROLES}
    indices = [i for role in ROLES for i in groups[role]]
    _need(set(rgb_by_index) == set(indices) and len(indices) == len(set(indices)), "Incomplete scene-motion source pictures")
    rgb = next(iter(rgb_by_index.values()))
    height, width = rgb.shape[:2]
    dim = (SCENE_WIDTH, max(1, round(SCENE_WIDTH * height / width)))
    scale = np.asarray((dim[0] / width, dim[1] / height), dtype=np.float64)
    mask = _mask([frame_records[i] for i in indices], (dim[1], dim[0]), scale)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frames = {i: clahe.apply(cv2.cvtColor(cv2.resize(image, dim, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY))
              for i, image in rgb_by_index.items()}
    ref_index = groups["before"][-1]
    basis = frames[ref_index]
    templates = _templates(basis, mask)
    corners = np.asarray(((0, 0), (dim[0] - 1, 0), (0, dim[1] - 1), (dim[0] - 1, dim[1] - 1)), dtype=np.float64)
    results = {}
    for i in indices:
        transform = np.eye(2, 3) if i == ref_index else _background_transform(templates, frames[i])
        if transform is None:
            continue
        shift = float(np.linalg.norm(corners @ transform[:, :2].T + transform[:, 2] - corners, axis=1).max())
        pose = _roll_and_center(frame_records[i], scale, transform)
        results[i] = {"scene_shift": shift, "pose": pose}
    base_result = {"registered_frames": len(results), "scheduled_frames": len(indices), "background_templates": len(templates),
                   "face_pose_frames": sum(row["pose"] is not None for row in results.values())}
    if len(results) != len(indices):
        return {"status": "background_registration_unassessed", **base_result}
    scene_max = max(row["scene_shift"] for row in results.values())
    base_result["scene_max_px_at_320"] = scene_max
    if scene_max > POLICY["maximum_scene_displacement_px_at_320"]:
        return {"status": "background_motion_measured", "translation_context": False, "translation_roll_context": False, **base_result}
    usable = {role: [results[i]["pose"] for i in group if results[i]["pose"] is not None] for role, group in groups.items()}
    needed = math.ceil(POLICY["minimum_middle_frame_fraction"] * len(groups["during"]))
    if len(usable["before"]) != 2 or len(usable["after"]) != 2 or len(usable["during"]) < needed:
        return {"status": "face_pose_unassessed", **base_result}
    face_width = float(np.median([frame_records[i]["bbox"]["w"] for i in indices]) * scale[0])
    flank_before, flank_after = (np.median(usable[role], axis=0) for role in ("before", "after"))
    middle = np.asarray(usable["during"])
    side_gap = abs(float(flank_after[0] - flank_before[0])) / face_width
    roll_gap = abs(float(flank_after[1] - flank_before[1]))
    travel = _sustained(middle[:, 0], flank_before[0], flank_after[0], POLICY["minimum_face_travel_box_widths"] * face_width)
    roll = _sustained(middle[:, 1], flank_before[1], flank_after[1], POLICY["minimum_roll_excursion_degrees"])
    qualified = side_gap <= POLICY["maximum_flank_gap_box_widths"] and travel >= needed
    representative = {role: group[len(group) // 2] for role, group in groups.items()}
    roll_needed = math.ceil(POLICY["minimum_middle_roll_fraction"] * len(groups["during"]))
    return {
        "status": "measured",
        "translation_context": qualified,
        "translation_roll_context": qualified and roll_gap <= POLICY["maximum_flank_roll_gap_degrees"] and roll >= roll_needed,
        "translation_support": travel, "roll_support": roll, "minimum_support": needed, "minimum_roll_support": roll_needed,
        "translation_flank_gap_box_widths": side_gap, "roll_flank_gap_degrees": roll_gap,
        "role_poses": {role: np.median(values, axis=0).tolist() for role, values in usable.items()},
        "role_support": {role: len(values) for role, values in usable.items()},
        "representative_inputs": {role: {key: frame_records[i][key] for key in ("frame_index", "pts", "decoded_rgb_sha256")}
                                  for role, i in representative.items()},
        **base_result,
    }


def native_context(path, record, table_doc, expected_rgb=None):
    table = FrameTable.from_json(table_doc)
    source = {**table_doc, "frame_count": len(table)}
    _need(not (table.rotation or table.truncated or table.interpolated), "Complete upright native clock required")
    validate_sequence_record(record, source, table_pts=table.pts)
    indices = [i for role in ROLES for i in record["centers"][role]]
    rows = {item["frame_index"]: item for item in record["frame_inputs"]}
    pixels = {}
    for frame in decode_frames(path, table, sorted(indices), timeout_s=60):
        digest = hashlib.sha256(frame.rgb).hexdigest()
        item = rows[frame.index]
        _need(frame.pts_verified and frame.pts == item["pts"] and digest == item["decoded_rgb_sha256"]
              and (expected_rgb is None or digest == expected_rgb[frame.index]), "Source scene-motion input differs from recorded pixels")
        pixels[frame.index] = frame.rgb
    return compare(record, pixels)
