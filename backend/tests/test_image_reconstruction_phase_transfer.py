from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts import image_reconstruction_phase_transfer as candidate
from scripts import review_image_reconstruction_phase_transfer as reviewer


@pytest.mark.parametrize("phase", candidate.PHASES)
def test_odd_image_size_keeps_exact_native_coordinates(phase):
    height, width = 257, 319
    left, right, top, bottom = candidate.phase_padding((height, width), phase)
    original = np.arange(height * width).reshape(height, width)
    padded = np.pad(original, ((top, bottom), (left, right)), mode="reflect")
    assert tuple(value % 8 for value in padded.shape) == (0, 0)
    assert (top % 8, left % 8) == phase
    assert np.array_equal(padded[top:top + height, left:left + width], original)


def test_phase_agreement_requires_three_local_error_deficits():
    rng = np.random.default_rng(618)
    # Repeat a texture tile so neighborhood texture alone does not select a region.
    rgb = np.tile(rng.integers(25, 225, (16, 16, 3), dtype=np.uint8), (32, 48, 1))
    errors = np.full((512, 768, 4), .11, dtype=np.float32)
    errors[160:288, 256:384, :2] = .015
    two = candidate.local_proposals(rgb, errors)
    assert two["fixed_phase"]["score"] > .8
    assert two["phase_agreement"]["score"] < .01

    errors[160:288, 256:384, 2] = .015
    three = candidate.local_proposals(rgb, errors)["phase_agreement"]
    x0, y0, x1, y1 = three["box"]
    assert three["score"] > .8 and x0 < 320 < x1 and y0 < 224 < y1


def test_flat_footage_has_no_assessable_reconstruction_region():
    rgb = np.full((512, 768, 3), 128, dtype=np.uint8)
    assert candidate.local_proposals(rgb, np.full((512, 768, 4), .01)) == dict.fromkeys(candidate.ARMS)


def test_independent_reviewer_counts_native_half_open_regions():
    truth = Image.new("L", (5, 6), 0)
    truth.paste(1, (2, 3, 5, 6))
    result = reviewer.measure((3, 3, 7, 7), truth)
    assert result == {"reported": 6, "target": 9, "overlap": 6, "iou": 2 / 3, "recall": 2 / 3}
    assert reviewer.useful(result)
