from __future__ import annotations

import cv2
import numpy as np

from scripts import image_edge_spread_transfer as study


def test_width_is_contrast_normalized_and_declines_complex_profiles():
    broad = np.r_[np.zeros(3), np.linspace(0, 1, 7), np.ones(3)]
    narrow = np.r_[np.zeros(6), .5, np.ones(6)]
    split = np.r_[[0, 0, 0, .5, .8, .3, .5, .8, 1, 1, 1, 1, 1]]
    data = np.stack((30 + 80 * broad, 150 + 30 * broad, 30 + 80 * narrow, 30 + 80 * split))
    widths, contrasts, supported = study.measure_profiles(data)
    np.testing.assert_array_equal(supported, (True, True, True, False))
    np.testing.assert_allclose(widths[:2], [3.6, 3.6])
    np.testing.assert_allclose(contrasts[:2], [80, 30])
    assert widths[2] < widths[0]


def test_matched_gap_ignores_changes_in_edge_type_mix_and_requires_overlap():
    inside = np.zeros((2, study.STRATA, 2))
    outside = np.zeros_like(inside)
    inside[0, :2] = [[45, 0], [15, 15 * np.log(3)]]
    outside[0, :2] = [[15, 0], [45, 45 * np.log(3)]]
    inside[1, 0], outside[1, 1] = [30, 0], [30, 30 * np.log(3)]
    gap, supported, common, strata = study.matched_gap(inside, outside)
    np.testing.assert_allclose(gap, [0, 0])
    np.testing.assert_array_equal(supported, [True, False])
    np.testing.assert_array_equal(common, [30, 0])
    np.testing.assert_array_equal(strata, [2, 0])


def test_authored_local_blur_has_native_location_and_global_blur_does_not_flag():
    y, x = np.indices((448, 448))
    luma = (35 + 65 * (x // 17 % 2) + 95 * (y // 23 % 2)).astype(np.uint8)
    original = np.repeat(luma[:, :, None], 3, axis=2)
    blurred = cv2.GaussianBlur(original, (0, 0), 1.55)
    local = original.copy()
    local[128:288, 128:288] = blurred[128:288, 128:288]
    truth = np.any(local != original, axis=2)
    result = study.response(local)["opposed_flanks"]
    assert result["ratio_interior_over_ring"] > 2.0
    mask = study.selection(result, study.POINT_FLOOR, truth.shape)
    assert study.useful(study.scoring.measure(mask, truth))
    assert not study.selection(result, result["score"], truth.shape).any()
    assert study.response(original)["opposed_flanks"]["score"] < study.POINT_FLOOR
    assert study.response(blurred)["opposed_flanks"]["score"] < study.POINT_FLOOR


def test_untextured_images_have_no_measured_region():
    assert study.response(np.full((384, 384, 3), 126, dtype=np.uint8)) == dict.fromkeys(study.ARMS)


def test_confirmation_roles_match_existing_native_control_names():
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    base = dict.fromkeys(study.paired.panel.CASE_ROLES, pixels)
    variants = study.native_variants(base, (8, 8, 24, 24))
    assert set(variants) == set(study.POSITIVE + study.CONTROLS)
