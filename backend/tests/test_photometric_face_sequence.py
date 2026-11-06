from __future__ import annotations

import numpy as np
import pytest

from scripts.score_photometric_face_sequence import normalize


def test_unclipped_brightness_offset_does_not_create_a_new_normalized_face():
    source = np.random.default_rng(83).integers(35, 145, (112, 112, 3), dtype=np.uint8)
    brighter = (source.astype(np.int16) + 40).astype(np.uint8)
    before = source.copy()
    first, second = normalize(source), normalize(brighter)
    assert first is not None and second is not None
    assert np.max(np.abs(first.astype(int)-second.astype(int))) <= 1
    np.testing.assert_array_equal(source, before)


def test_low_contrast_crop_is_unassessed_instead_of_amplifying_noise():
    assert normalize(np.full((112, 112, 3), 80, np.uint8)) is None
    rng = np.random.default_rng(3)
    assert normalize(rng.integers(79, 83, (112, 112, 3), dtype=np.uint8)) is None


def test_valid_normalization_preserves_spatial_shape_and_channel_differences():
    gradient = np.tile(np.arange(112, dtype=np.uint8), (112, 1)) + 40
    source = np.stack([gradient+20, gradient, gradient-20], axis=-1)
    result = normalize(source)
    assert result is not None and result.dtype == np.uint8 and result.shape == source.shape
    assert np.all(result[:, :, 0] >= result[:, :, 1])
    assert np.all(result[:, :, 1] >= result[:, :, 2])
    assert not np.array_equal(result[:, :, 0], result[:, :, 2])
    assert result[30, 80, 1] > result[30, 20, 1]


@pytest.mark.parametrize("value", [np.zeros((112, 112), np.uint8), np.zeros((112, 112, 3), np.float32)])
def test_non_rgb_and_float_inputs_are_refused(value):
    with pytest.raises(ValueError, match="uint8 RGB"):
        normalize(value)
