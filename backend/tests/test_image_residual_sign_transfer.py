from itertools import product

import numpy as np
import pytest

from scripts import image_residual_sign_transfer as study


@pytest.mark.parametrize(("stride", "compressed"), list(product((1, 2), (False, True))))
def test_triplets_bind_their_source_cells_and_jpeg_sample_axis(stride, compressed):
    rng = np.random.default_rng(94)
    gray = rng.integers(25, 230, (32, 48)).astype(np.float32)
    counts = np.zeros((2, 3, 4, 4))
    for direction, values in enumerate((gray, gray.T)):
        expanded = np.pad(values, ((0, 0), (stride, stride)), mode="reflect")
        residual = expanded[:, :values.shape[1]] - 2 * values + expanded[:, 2 * stride:]
        padded = np.pad(np.abs(residual), 4, mode="reflect")
        for y, x in product(range(values.shape[0]), range(stride, values.shape[1] - stride - 2)):
            if compressed and (x - stride) // 8 != (x + 2 + stride) // 8:
                continue
            triplet = residual[y, x:x + 3]
            if np.min(np.abs(triplet)) < 1:
                continue
            left, middle, right = triplet > 0
            symbol = 2 * int(left != middle) + int(middle != right)
            activity = padded[y:y + 9, x:x + 9].sum() > 4 * 81
            cell = (x // 16, y // 16) if direction else (y // 16, x // 16)
            counts[*cell, 2 * direction + int(activity), symbol] += 1
    np.testing.assert_array_equal(study.symbol_counts(gray, stride, compressed), counts)


def test_two_scale_reports_authored_texture_and_leaves_flat_source_unassessed():
    rng = np.random.default_rng(613)
    luma = rng.integers(108, 148, (320, 320), dtype=np.uint8)
    y, x = np.indices((112, 112))
    luma[96:208, 112:224] = (119 + 10 * ((x + y) % 3)).astype(np.uint8)
    result = study.response(np.repeat(luma[..., None], 3, axis=2), "png")["two_scale"]
    assert result["box"] == [96, 80, 224, 208]
    assert result["score"] > .04
    assert study.response(np.full((320, 320, 3), 128, np.uint8), "png") == dict.fromkeys(study.ARMS)
