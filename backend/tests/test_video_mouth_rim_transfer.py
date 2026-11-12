from fractions import Fraction

import cv2
import numpy as np

from scripts.video_mouth_rim_transfer import rim_measures


def test_local_color_boundary_has_positive_contrast_at_the_matching_radius():
    side = 96
    mouth = np.array((47, 57))
    yy, xx = np.ogrid[:side, :side]
    ellipse = ((xx - mouth[0]) / 17) ** 2 + ((yy - mouth[1]) / 10) ** 2 <= 1
    rgb = np.full((side, side, 3), (140, 130, 120), np.uint8)
    flat = rgb.copy()
    rgb[ellipse] = (130, 120, 170)
    geometry = (
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np.eye(2, 3), mouth, rgb.shape[:2]
    )
    ordinary = rim_measures(flat, geometry)
    localized = rim_measures(rgb, geometry)
    assert all(abs(measure[0]) < 1e-6 for measure in ordinary)
    assert localized[0][0] > 0.5
    assert localized[0][0] > max(values[0] for values in localized[1:])
    assert localized[0][2] == (localized[0][0] + localized[0][1]) / 2


def test_missing_native_rim_support_is_unassessed():
    rgb = np.zeros((96, 96, 3), np.uint8)
    geometry = rgb[..., 0], np.eye(2, 3), np.array((60, 73)), rgb.shape[:2]
    assert rim_measures(rgb, geometry) is None


def test_exact_threshold_and_half_open_iou_are_reviewed_separately():
    from scripts.review_video_mouth_rim_transfer import outcome

    best = {"score": 0.5, "medians": [0.25, 0.75, 0.0], "source_clock_s": ["1/2", "3/2"]}
    case = {"arms": {"joint_rim": {"best": best}}, "truth_clock_s": ["1", "2"]}
    assert outcome(case, "joint_rim", 0.5) == "measured_no_range"
    assert outcome(case, "joint_rim", 0.25) == "wrong_range"
    best["source_clock_s"] = [str(Fraction(1)), str(Fraction(3, 2))]
    assert outcome(case, "joint_rim", 0.25) == "correct_range"
