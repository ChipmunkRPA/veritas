from __future__ import annotations

import cv2
import numpy as np

from scripts import image_shifted_jpeg_likelihood as study


def test_uniform_quantization_noise_projects_equally_at_every_phase():
    basis = study.dct_basis()
    np.testing.assert_allclose(basis @ basis.T, np.eye(8), atol=1e-14)
    table = np.full((8, 8), 12)
    for x, y in ((0, 0), (5, 3), (4, 4), (7, 7)):
        np.testing.assert_allclose(study.noise_scales(table, basis, x, y), np.sqrt(12 + study.ROUNDING_SIGMA ** 2))
    coarse = table * 2
    assert np.all(study.noise_scales(coarse, basis, 5, 3) > study.noise_scales(table, basis, 5, 3))


def test_wrapped_likelihood_rewards_nonzero_lattice_and_declines_unmeasurable_values():
    coefficients = np.asarray([[[0, 4, 20, 30, 40]]], dtype=np.float32)
    score, count = study.block_statistics(coefficients, np.asarray([[[20]]]), np.asarray([[[3]]]))
    np.testing.assert_array_equal(count, [[0, 0, 1, 1, 1]])
    assert score[0, 2] > 0 > score[0, 3]
    np.testing.assert_allclose(score[0, 2], score[0, 4])
    broad_scores, broad_counts = study.block_statistics(coefficients, np.asarray([[[20]]]), np.asarray([[[11]]]))
    assert not np.any(broad_scores) and not np.any(broad_counts)


def test_selected_mask_follows_native_shifted_coordinates_and_is_strict_at_point():
    count = np.full((64, 64), 4, dtype=np.int32)
    evidence = np.zeros_like(count, dtype=np.float32)
    evidence[20:36, 24:48] = 4
    region = study.best_window(evidence, count, 5, 3, 59)
    assert region.box == (197, 163, 389, 291)
    mask = study.selection(region, region.score - 0.01, (520, 520))
    assert mask.sum() == 192 * 128
    assert mask[163, 197] and mask[290, 388] and not mask[162, 197] and not mask[163, 196]
    assert not np.any(study.selection(region, region.score, (520, 520)))
    assert study.best_window(evidence, np.zeros_like(count), 5, 3, 59) is None


def test_authored_local_grid_recovers_the_shifted_phase_over_controls():
    rng = np.random.default_rng(1979)
    luma = np.clip(cv2.GaussianBlur(rng.normal(128, 90, (448, 448)), (5, 5), 0.7), 0, 255).astype(np.uint8)
    authored = np.repeat(luma[:, :, None], 3, axis=2)
    box = (112, 112, 272, 240)
    views = study.source.materials(authored, authored, box)
    measured = {role: study.response(study.source.delivery(views[role], 90)[0])
                for role in ("original", "global_shift", "sharp_shift", "feathered_shift")}
    control_max = max(measured[role].score for role in ("original", "global_shift"))
    for role in ("sharp_shift", "feathered_shift"):
        region = measured[role]
        assert region.phase == (5, 3) and region.prior_quality == 59
        assert region.score > control_max
        assert region.box[0] <= 192 < region.box[2] and region.box[1] <= 176 < region.box[3]
