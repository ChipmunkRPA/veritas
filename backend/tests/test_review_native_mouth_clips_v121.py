import numpy as np
import pytest

from scripts import review_native_mouth_clips_v121 as review


def test_mask_is_measured_from_actual_source_and_output_pixels():
    source = np.zeros((8, 9, 3), np.uint8)
    output = source.copy()
    output[2, 3, 0] = 1
    mask = np.zeros((8, 9), np.uint8)
    mask[2, 3] = 255
    assert review.changed_pixels(source, output, mask) == 1
    assert review.changed_pixels(source, source.copy(), np.zeros_like(mask)) == 0


@pytest.mark.parametrize("problem", ["wrong_position", "nonbinary", "shape", "dtype", "extra_change"])
def test_misaligned_or_incomplete_masks_are_rejected(problem):
    source = np.zeros((8, 9, 3), np.uint8)
    output = source.copy()
    output[2, 3] = 1
    mask = np.zeros((8, 9), np.uint8)
    mask[2, 3] = 255
    if problem == "wrong_position":
        mask[2, 3] = 0
        mask[3, 2] = 255
    elif problem == "nonbinary":
        mask[2, 3] = 1
    elif problem == "shape":
        mask = mask[:-1]
    elif problem == "dtype":
        output = output.astype(np.float32)
    else:
        output[4, 5] = 1
    with pytest.raises(ValueError):
        review.changed_pixels(source, output, mask)


def test_pixel_edge_crop_transform_is_checked_separately():
    x, y, w, h = 11, 7, 128, 160
    sx, sy = 256 / w, 256 / h
    p = {
        "geometry": {"bbox": [x, y, x + w, y + h]},
        "source_to_crop_matrix_hex": [
            [float(v).hex() for v in r]
            for r in [[sx, 0, (0.5 - x) * sx - 0.5], [0, sy, (0.5 - y) * sy - 0.5]]
        ],
    }
    review.crop_transform(p)
    p["source_to_crop_matrix_hex"][0][2] = float(0).hex()
    with pytest.raises(ValueError):
        review.crop_transform(p)
