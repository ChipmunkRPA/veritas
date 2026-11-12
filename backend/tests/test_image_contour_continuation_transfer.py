import numpy as np
import pytest

from scripts import image_contour_continuation_transfer as method


@pytest.mark.parametrize("fmt", method.panel.FORMATS)
def test_contours_stop_in_authored_sharp_and_feathered_locations(fmt):
    row, col = np.indices((512, 512))
    luma = (120 + 40 * np.sin(row * np.pi / 10) + 40 * np.sin(col * np.pi / 10)).astype(np.uint8)
    original = np.repeat(luma[..., None], 3, axis=2)
    box = (192, 160, 320, 288)
    background = np.full_like(original, 120)
    unaltered = np.asarray(method.panel.paired.encoded(original, fmt)[1])
    assert method.response(unaltered)["two_distances"]["score"] < method.POINT_FLOOR
    for feather in (False, True):
        edit = method.panel.blend.local(original, background, box, feather=feather)
        delivered = np.asarray(method.panel.paired.encoded(edit, fmt)[1])
        proposed = method.response(delivered)["two_distances"]
        assert proposed["box"] == list(box)
        assert proposed["score"] > .45
        assert min(proposed["short_counts"]) >= 10
        assert min(proposed["long_counts"]) >= 10


def test_structureless_and_one_orientation_only_leave_no_region():
    flat = np.full((320, 320, 3), 100, np.uint8)
    horizontal_bars = (120 + 50 * np.sin(np.arange(320) * np.pi / 10)).astype(np.uint8)
    bars = np.broadcast_to(horizontal_bars[:, None, None], flat.shape)
    assert method.response(flat) == dict.fromkeys(method.ARMS)
    assert method.response(bars) == dict.fromkeys(method.ARMS)


def test_crossing_loss_respects_direction_and_removes_image_edge_samples():
    gradient = np.zeros((96, 128), np.float32)
    gradient[40, :48] = 12
    planes_left = method.plane_integral(gradient, 4, 4, 1)
    planes_right = method.plane_integral(gradient, 4, 4, -1)
    np.testing.assert_array_equal(planes_left[41, 3] - planes_left[40, 3], [1, 1])
    np.testing.assert_array_equal(planes_right[41, 3] - planes_right[40, 3], [0, 0])
    np.testing.assert_array_equal(planes_left[41, 2] - planes_left[40, 2], [1, 0])
    assert not np.any(planes_left[:, 0])
