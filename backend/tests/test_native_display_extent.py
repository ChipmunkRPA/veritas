"""The diagnostic grid must retain border measurements and map to the full source."""
import numpy as np
import pytest

from app.detectors.image.native_forensics import _display


@pytest.mark.parametrize("shape", [(51, 4101), (4101, 51), (2501, 5001)])
def test_large_nondivisible_image_keeps_right_and_bottom_border(shape):
    values = np.zeros(shape, dtype=np.float32)
    values[-1, :] = 10
    values[:, -1] = 10
    reduced, space = _display(values)
    assert max(reduced.shape) <= 4096
    assert np.all(reduced[-1, :] > 0)
    assert np.all(reduced[:, -1] > 0)
    assert space == {"width": shape[1], "height": shape[0], "scale_to_original": 1.0}
    # Uniform area integration preserves the complete image's mean (including
    # border-only measurements that the old block truncation discarded).
    assert float(reduced.mean()) == pytest.approx(float(values.mean()), rel=1e-5)


def test_small_image_is_not_resampled():
    values = np.arange(15, dtype=np.float32).reshape(3, 5)
    reduced, space = _display(values)
    assert reduced is values
    assert space == {"width": 5, "height": 3, "scale_to_original": 1.0}
