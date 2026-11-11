"""Descriptive landmark support for retained face-appearance windows."""

from __future__ import annotations

from statistics import mean

import numpy as np

from scripts.video_face_confound_v222 import MIN_ROLE_FRAMES, ROLES, Unavailable, geometry, require


def measure(record):
    """Measure the caller's already source-bound before, during and after frames."""
    try:
        groups = {role: [i + k for i in record["centers"][role] for k in (-1, 0, 1)] for role in ROLES}
        indices = [i for role in ROLES for i in groups[role]]
        require(all(len(group) >= MIN_ROLE_FRAMES for group in groups.values()), "insufficient_role_frames")
        require(indices == sorted(set(indices)), "overlapping_or_unordered_role_frames")
        rows = {row["frame_index"]: row for row in record["frame_inputs"]}
        require(
            len(rows) == len(record["frame_inputs"]) and set(indices) == set(rows), "inexact_role_coverage"
        )

        support, shapes = {}, {}
        for role, group in groups.items():
            ratios, normalized, outside = [], [], 0
            for index in group:
                row = rows[index]
                marks = np.asarray(row["landmarks"], dtype=np.float64)
                x, y, width, height = (float(row["bbox"][key]) for key in ("x", "y", "w", "h"))
                require(marks.shape == (5, 2) and np.isfinite(marks).all(), "invalid_landmarks")
                require(
                    np.isfinite([x, y, width, height]).all() and min(width, height) > 0, "invalid_face_box"
                )
                ratios.append(float(np.linalg.norm(marks[1] - marks[0])) / width)
                normalized.append(geometry(marks))
                outside += int(np.any((marks < (x, y)) | (marks > (x + width, y + height))))
            require(np.isfinite(ratios).all() and np.isfinite(normalized).all(), "nonfinite_landmark_support")
            support[role] = {
                "median": float(np.median(ratios)),
                "minimum": min(ratios),
                "outside_box_frames": outside,
            }
            shapes[role] = np.median(normalized, axis=0)

        pairs = (("before", "during"), ("after", "during"), ("before", "after"))
        distances = {
            f"{first}_to_{second}": float(
                np.sqrt(np.mean(np.sum((shapes[second] - shapes[first]) ** 2, axis=1)))
            )
            for first, second in pairs
        }
        require(np.isfinite(list(distances.values())).all(), "nonfinite_landmark_support")
        return {
            "status": "measured",
            "role_frame_counts": {role: len(group) for role, group in groups.items()},
            "eye_box_width": support,
            "during_minus_flank_eye_box_width": support["during"]["median"]
            - mean(support[role]["median"] for role in ("before", "after")),
            "geometry_rms_eye_units": distances,
        }
    except Unavailable as exc:
        return {"status": "unavailable", "reason": str(exc)}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": "unavailable", "reason": "invalid_landmark_support"}
