import cv2
import numpy as np
import pytest

from scripts import portrait_native_qualifier_v133 as qualifier


def test_native_affine_keeps_pixel_centres_when_scaling_256_to_512():
    matrix = cv2.getRotationMatrix2D((130, 110), 13, 0.8)
    points = np.array([[4, 3, 1], [70, 240, 1], [150, 30, 1]], np.float64)
    scaled = qualifier.upscale_matrix(matrix)
    assert np.allclose(points @ scaled.T, 2 * (points @ matrix.T) + 0.5, rtol=0, atol=1e-12)
    inverse = cv2.invertAffineTransform(scaled)
    mapped = np.column_stack([points @ scaled.T, np.ones(3)]) @ inverse.T
    assert np.allclose(mapped, points[:, :2], rtol=0, atol=1e-12)
    assert not np.allclose(points @ scaled.T, points @ (matrix * 2).T, rtol=0, atol=1e-12)


@pytest.mark.parametrize("matrix", [np.zeros((2, 3)), np.ones((3, 3)), [[1, 0, float("nan")], [0, 1, 0]]])
def test_invalid_or_singular_transform_cannot_qualify(matrix):
    with pytest.raises(ValueError):
        qualifier.upscale_matrix(matrix)


def test_unchanged_earlier_aperture_is_reported_without_claiming_phonetic_alignment():
    original = np.zeros((512, 512, 3), np.uint8)
    opened = original.copy()
    opened[40:44, 10:13] = 1
    assert qualifier.response_summary([original, original.copy(), original.copy(), opened]) == {
        "earlier_native_aperture": 0,
        "authored_aperture_0.5": 12,
    }
    changed_repeat = original.copy()
    changed_repeat[0, 0] = 1
    with pytest.raises(ValueError, match="repeat|Repeated"):
        qualifier.response_summary([original, changed_repeat, original, opened])
    with pytest.raises(ValueError, match="no pixel effect"):
        qualifier.response_summary([original, original, opened, original])


def test_only_fixed_neural_qualification_is_routed(monkeypatch):
    calls = []
    monkeypatch.setattr(qualifier.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    qualifier.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "out"])
    assert calls[0][3] == "scripts.portrait_native_qualifier_v133"
    with pytest.raises(ValueError):
        qualifier.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
