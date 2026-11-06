from types import SimpleNamespace

import numpy as np
import pytest

from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face


def face():
    return SimpleNamespace(
        x=100, y=80, w=120, h=180, landmarks=[(130, 120), (190, 120), (160, 160), (140, 220), (180, 220)]
    )


def test_permitted_box_crop_records_approximation_and_exact_pixel_edge_mapping():
    rgb = np.zeros((320, 320, 3), np.uint8)
    crop, matrix, geometry = crop_from_face(rgb, face())
    assert crop.shape == (256, 256, 3)
    assert geometry["bbox"] == [100, 40, 220, 260]
    assert geometry["approximate_half_face_y"] == 150
    assert "approximation" in geometry["interpretation"]
    assert np.allclose(matrix @ [99.5, 39.5, 1], [-0.5, -0.5])
    assert np.allclose(matrix @ [219.5, 259.5, 1], [255.5, 255.5])


def test_mouth_support_has_finite_native_coordinates_and_zero_outside():
    alpha = alpha_from_face((320, 320, 3), face())
    assert alpha.shape == (320, 320) and np.isfinite(alpha).all()
    assert alpha[220, 160] == 1 and alpha.min() == 0 and alpha.max() == 1
    assert not alpha[:190].any() and not alpha[250:].any()
    narrow = face()
    narrow.landmarks[4] = (141, 220)
    with pytest.raises(ValueError):
        alpha_from_face((320, 320, 3), narrow)
