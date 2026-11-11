import pytest

from scripts.video_landmark_support import measure


def window():
    roles = {"before": [2, 6], "during": [10, 14, 18], "after": [22, 26]}
    frames = []
    for role, centers in roles.items():
        separation = 6 if role == "during" else 40
        for center in centers:
            for index in range(center - 1, center + 2):
                frames.append(
                    {
                        "frame_index": index,
                        "bbox": {"x": 0, "y": 0, "w": 100, "h": 100},
                        "landmarks": [[20, 25], [20 + separation, 25], [30, 45], [25, 60], [55, 60]],
                    }
                )
    return {"centers": roles, "frame_inputs": frames}


def test_role_support_detects_eye_separation_collapse_and_preserves_counts():
    result = measure(window())
    assert result["status"] == "measured"
    assert result["role_frame_counts"] == {"before": 6, "during": 9, "after": 6}
    assert result["eye_box_width"]["during"] == {"median": 0.06, "minimum": 0.06, "outside_box_frames": 0}
    assert result["during_minus_flank_eye_box_width"] == pytest.approx(-0.34)
    geometry = result["geometry_rms_eye_units"]
    assert geometry["before_to_during"] == geometry["after_to_during"] > 1
    assert geometry["before_to_after"] == 0


def test_unavailable_geometry_preserves_degeneracy_and_overlap_reasons():
    record = window()
    record["frame_inputs"][0]["landmarks"][1] = record["frame_inputs"][0]["landmarks"][0]
    assert measure(record) == {"status": "unavailable", "reason": "degenerate_eye_landmarks"}

    record["centers"]["before"][1] = 9
    assert measure(record) == {"status": "unavailable", "reason": "overlapping_or_unordered_role_frames"}
