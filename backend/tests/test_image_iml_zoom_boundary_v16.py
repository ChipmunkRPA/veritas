"""Rounding diagnostics must expose discontinuities without clearing strict failure."""

import numpy as np
import pytest

from scripts.review_image_iml_zoom_boundaries_v16 import boundary_difference, zero_metric


def test_one_ulp_change_at_half_level_is_counted_as_discrete_difference():
    a = np.full((3, 5), 0.5)
    b = a.copy()
    b[1, 2] = np.nextafter(0.5, 0.0)
    error, changed = boundary_difference(a, b)
    assert 0 < error < 1e-15 and changed == 1


def test_small_continuous_difference_away_from_half_level_changes_no_pixel():
    a = np.full((3, 5), 0.4)
    b = np.nextafter(a, np.inf)
    assert boundary_difference(a, b)[1] == 0


def test_original_continuous_tolerance_is_not_relaxed():
    with pytest.raises(ValueError, match="original tolerance"):
        boundary_difference(np.zeros((2, 3)), np.full((2, 3), 1e-8))


def test_ungated_zero_keeps_every_true_edited_pixel_as_a_miss():
    truth = np.zeros((7, 11), bool)
    truth[2:5, 3:9] = True
    metric = zero_metric(truth)
    assert metric["false_negative_pixels"] == metric["reference_positive_pixels"] == 18
    assert metric["pixels"] == 77 and metric["predicted_positive_pixels"] == 0
    assert metric["iou"] == metric["pixel_f1"] == metric["recall"] == 0
