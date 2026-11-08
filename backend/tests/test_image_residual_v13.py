"""Noise-feature model: geometry, interventions and source-only fit invariants."""

import numpy as np
import pytest

from scripts import image_residual_localizer_v13 as method


def test_constant_and_noisy_images_have_finite_features_at_source_geometry():
    for rgb in (
        np.zeros((79, 101, 3), np.uint8),
        np.random.default_rng(13).integers(0, 256, (91, 137, 3), dtype=np.uint8),
    ):
        values = method.features(rgb)
        assert values.shape == (64, 64, 32) and values.dtype == np.float32 and np.isfinite(values).all()
        assert np.array_equal(values, method.features(rgb))
    with pytest.raises(ValueError, match="source RGB"):
        method.features(np.zeros((63, 100, 3), np.uint8))


def test_ordinary_patch_changes_only_declared_source_rectangle():
    rgb = np.random.default_rng(4).integers(0, 256, (87, 109, 3), dtype=np.uint8)
    changed = method.intervene(rgb, "ordinary_patch")
    h, w = rgb.shape[:2]
    allowed = np.zeros((h, w), bool)
    allowed[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    assert np.array_equal(rgb[~allowed], changed[~allowed])
    assert (rgb[allowed] != changed[allowed]).any()
    for kind in ("lighting", "jpeg55"):
        assert method.intervene(rgb, kind).shape == rgb.shape


def test_projection_removes_measured_nuisance_and_preserves_orthogonal_signal():
    deltas = np.zeros((40, 32))
    deltas[:20, 0] = 2
    deltas[20:, 1] = 3
    projection = method.projector([deltas], np.ones(32), 2)
    np.testing.assert_allclose(deltas @ projection, 0, atol=1e-12)
    probe = np.zeros(32)
    probe[2] = 1
    np.testing.assert_allclose(probe @ projection, probe, atol=1e-12)


def test_soft_target_fit_converges_and_orders_known_local_signal():
    rng = np.random.default_rng(12)
    x = rng.normal(0, 0.1, (256, 32))
    x[128:, 2] += 2
    y = np.r_[np.zeros(128), np.ones(128)]
    deltas = np.zeros((64, 32))
    deltas[:32, 0] = 1
    deltas[32:, 1] = 1
    for rank in (0, 2):
        head = method.fit(x, y, [deltas], rank)
        grid = np.zeros((64, 64, 32), np.float32)
        grid[32:, :, 2] = 2
        scores = method.response(grid, head)
        assert scores[48].mean() > scores[16].mean() + 0.5
        assert head["calibrated_confidence"] is False


def test_source_projection_and_metrics_use_exact_edge_aligned_coordinates():
    scores = np.zeros((64, 64))
    scores[5:10, 7:12] = 0.9
    prediction = method.mask(scores, 0.5, 137, 91)
    expected = np.zeros((91, 137), bool)
    for y in range(91):
        for x in range(137):
            expected[y, x] = 5 <= y * 64 // 91 < 10 and 7 <= x * 64 // 137 < 12
    np.testing.assert_array_equal(prediction, expected)
    result = method.metrics(prediction, expected)
    assert result["iou"] == result["pixel_f1"] == 1 and result["flag"]
    scores[:] = 0
    scores[1:4, 1:4] = 0.9
    assert not method.mask(scores, 0.5, 137, 91).any()


def test_calibration_keeps_ties_below_the_point_and_caps_source_groups():
    values = [0.1] * 56 + [0.5, 0.6, 0.7, 0.8]
    point = method.point(values)
    assert point["value"] == 0.5 and sum(v > point["value"] for v in values) == 3
    assert method.point([0.5] * 60)["value"] == 0.5
    assert not method.mask(np.full((64, 64), 0.5), 0.5, 101, 79).any()


def test_nonfinite_or_geometry_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        method.mask(np.full((64, 64), np.nan), 0.5, 91, 91)
    with pytest.raises(ValueError):
        method.metrics(np.zeros((79, 91), bool), np.zeros((91, 79), bool))
