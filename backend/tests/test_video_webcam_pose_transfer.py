"""Regression coverage for the frozen within-track geometry comparison."""

from scripts import video_webcam_pose_transfer as study


def case(middle):
    groups = {"before": [3, 7], "during": [11, 15, 19], "after": [23, 27]}
    normal = [[30, 30], [70, 30], [50, 45], [33, 65], [67, 65]]
    rows = [{"frame_index": i, "landmarks": (middle if role == "during" else normal),
             "bbox": {"w": 100}} for role, centers in groups.items() for center in centers for i in (center - 1, center, center + 1)]
    return {"centers": groups, "frame_inputs": rows}


def test_reversed_landmarks_need_persistent_source_support():
    reversed_points = [[70, 30], [30, 30], [50, 45], [33, 65], [67, 65]]
    record = case(reversed_points)
    assert study.score(record)["context"] == "landmark_inversion"
    for row in record["frame_inputs"]:
        if row["frame_index"] in (14, 15, 16, 18, 19, 20):
            row["landmarks"] = [[30, 30], [70, 30], [50, 45], [33, 65], [67, 65]]
    assert study.score(record)["context"] is None


def test_eye_narrowing_needs_consistent_nose_displacement():
    centered = [[40, 30], [60, 30], [50, 45], [37, 65], [63, 65]]
    turned = [[40, 30], [60, 30], [34, 45], [37, 65], [63, 65]]
    both = study.score(case(turned))
    assert both["context"] == "bilateral_support_change"
    old_only = study.score(case(centered))
    assert old_only["context"] is None
    assert old_only["old_eye_width_rule"]


def test_exact_native_intersection_boundary():
    from fractions import Fraction

    assert study.iou(["9/5", "19/5"], ["2", "4"]) == Fraction(9, 11)
    assert study.iou(["9/5", "2"], ["2", "4"]) == 0
