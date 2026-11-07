"""SFace input safety and pixel-coordinate/channel contracts, without a model."""
import numpy as np
import pytest

from scripts.sface_native_v15 import TARGET, cpp_alignment, face_row, manual_blob, validate


@pytest.mark.parametrize('points', [np.zeros((5, 2)), np.ones((5, 2)), np.zeros((4, 2)),
    np.full((5, 2), np.nan), np.full((5, 2), np.inf), np.arange(10).reshape(5, 2)])
def test_invalid_or_degenerate_landmarks_never_become_evidence(points):
    with pytest.raises(ValueError):
        validate(points)


def test_image_coordinates_landmarks_preserved_in_native_row():
    points = TARGET + [30, 15]
    row = face_row(points)
    np.testing.assert_array_equal(row[0, 4:14].reshape(5, 2), points.astype(np.float32))
    assert row.dtype == np.float32 and row.shape == (1, 15)


def test_bgr_to_rgb_once_without_outer_normalization():
    bgr = np.full((112, 112, 3), [10, 20, 30], np.uint8)
    result = manual_blob(bgr)
    assert result.dtype == np.float32 and result.shape == (1, 3, 112, 112)
    np.testing.assert_array_equal(result[0, :, 0, 0], [30, 20, 10])
    assert result.flags.c_contiguous


@pytest.mark.parametrize('image', [np.zeros((112, 112, 4), np.uint8), np.zeros((112, 112, 3), np.float32),
    np.zeros((224, 224, 3), np.uint8), np.zeros((112, 112), np.uint8)])
def test_wrong_input_contract_refused(image):
    with pytest.raises(ValueError):
        manual_blob(image)


def test_scale_translation_maps_back_to_reference_landmarks():
    source = TARGET * 1.5 + [12, 7]
    matrix = cpp_alignment(source)
    mapped = source @ matrix[:, :2].T + matrix[:, 2]
    np.testing.assert_allclose(mapped, TARGET, atol=1e-3, rtol=0)
