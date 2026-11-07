from __future__ import annotations

import numpy as np
import pytest

from scripts.image_patch_drift_v8 import encoded_grid, haar_attenuate, median3, scores, token_drift, views


def test_haar_strength_retains_eighty_percent_of_block_detail():
    pixels = np.tile(np.asarray([[0, 255], [255, 0]], np.uint8), (4, 4))
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    result = haar_attenuate(rgb, alpha=.2, levels=2)
    np.testing.assert_allclose(result[pixels == 0], .1, rtol=0, atol=2e-8)
    np.testing.assert_allclose(result[pixels == 255], .9, rtol=0, atol=3e-8)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(haar_attenuate(rgb, alpha=0), rgb.astype(np.float32) / 255)


def test_haar_is_mask_blind_and_rejects_incompatible_geometry():
    with pytest.raises(ValueError, match='divisible'):
        haar_attenuate(np.zeros((7, 8, 3), np.uint8))
    with pytest.raises(ValueError, match='alpha'):
        haar_attenuate(np.zeros((8, 8, 3), np.uint8), alpha=1.01)
    constant = np.full((8, 12, 3), 128, np.uint8)
    np.testing.assert_allclose(haar_attenuate(constant), constant.astype(np.float32) / 255, rtol=0, atol=1e-7)


def test_cosine_drift_has_corresponding_patch_geometry_and_distance_units():
    left = np.asarray([[1., 0.], [1., 0.], [1., 0.]], np.float32)
    right = np.asarray([[1., 0.], [0., 1.], [-1., 0.]], np.float32)
    np.testing.assert_array_equal(token_drift(left, right), [0., 1., 2.])
    with pytest.raises(ValueError, match='shapes differ'):
        token_drift(left, right[:2])


def test_reflect_boundary_median_is_not_zero_padding():
    values = np.asarray([[8., 1., 9.], [2., 3., 4.], [7., 5., 6.]], np.float32)
    result = median3(values)
    assert result.shape == values.shape and result.dtype == np.float32
    assert result[0, 0] == 3. and result[1, 1] == 5.


@pytest.mark.parametrize('point', [.0000026, .0026, 1., 100.])
def test_strict_eligibility_survives_uint8_serialization(point):
    p = np.float32(point)
    values = np.asarray([np.nextafter(p, -np.inf, dtype=np.float32), p,
        np.nextafter(p, np.inf, dtype=np.float32), p * 10, -p], np.float32)
    stored = encoded_grid(values, float(p))
    np.testing.assert_array_equal(stored >= 128, values > float(p))
    assert stored.dtype == np.uint8


def test_within_image_normalization_is_explicit_and_constant_safe():
    values = np.arange(1024, dtype=np.float32).reshape(32, 32)
    np.testing.assert_array_equal(scores(values, 'within_image'), scores(values * 2 + 8, 'within_image'))
    np.testing.assert_array_equal(scores(np.ones((32, 32), np.float32), 'within_image'), np.zeros((32, 32)))
    with pytest.raises(ValueError, match='Unknown'):
        scores(values, 'tuned-after-results')


def test_float_views_are_not_requantized_after_the_perturbation():
    image = np.zeros((448, 448, 3), np.uint8)
    image[::2, ::2] = 255
    clean, changed, normalized = views(image)
    assert clean.shape == changed.shape == (448, 448, 3)
    assert normalized.shape == (2, 3, 448, 448) and normalized.dtype == np.float32
    assert not np.all(changed * 255 == np.rint(changed * 255))


def test_invalid_readouts_cannot_become_zero_negative_evidence():
    for value in (np.zeros((16, 16)), np.full((32, 32), np.nan)):
        with pytest.raises(ValueError):
            scores(value, 'raw')
    for point in (0., -1., float('nan')):
        with pytest.raises(ValueError):
            encoded_grid(np.zeros((32, 32)), point)
