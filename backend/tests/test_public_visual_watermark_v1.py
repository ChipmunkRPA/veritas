"""Exact region coordinates, fixed inspection selection and unavailable messages."""

import numpy as np
import pytest

from scripts import public_visual_watermark_v1 as method
from scripts import run_public_visual_watermark_v1 as frozen
from scripts.review_public_visual_watermark_v1 import independent


def test_partial_region_preserves_source_coordinates_and_message_without_ai_meaning():
    logits = np.full((256, 256), -1, np.float32)
    logits[64:192, 64:192] = 1
    bits = np.zeros((32, 256, 256), bool)
    bits[1::2] = True
    e = frozen.packet(logits, bits)
    assert e["regions"] == [{"x": 64, "y": 64, "w": 128, "h": 128, "pixels": 16384}]
    assert e["decoded_bits_correct_of32"] == 32 and e["decoded_message"] == [i % 2 for i in range(32)]
    assert e["used_in_confidence"] is e["early_stop_eligible"] is e["provider_authenticated"] is False
    packed = np.packbits(bits.reshape(32, -1), axis=1, bitorder="little")
    mask, regions, decoded = independent(logits, packed)
    assert regions == e["regions"] and decoded == e["decoded_message"]
    assert method.overlap(mask, frozen.truth("partial"))["iou"] == 1


def test_native_cutoff_and_minimum_component_do_not_hide_raw_pixels():
    logits = np.zeros((256, 256), np.float32)
    logits[0, :256] = 1
    logits[1, :243] = 1
    e = method.spatial(logits)
    assert e["selected_pixels"] == 499 and not e["flagged_for_inspection"]
    logits[1, 243] = 1
    assert method.spatial(logits)["regions"][0]["pixels"] == 500
    empty = frozen.packet(np.zeros_like(logits), np.zeros((32, 256, 256), bool))
    assert empty["decoded_message"] is None and empty["decoded_bits_correct_of32"] is None


@pytest.mark.parametrize(
    "values",
    [
        np.full((256, 256), np.nan, np.float32),
        np.zeros((128, 128), np.float32),
        np.zeros((256, 256), np.float64),
    ],
)
def test_unusable_output_is_not_a_negative_watermark_or_originality_verdict(values):
    with pytest.raises(ValueError):
        method.spatial(values)


def test_authored_sources_and_transforms_are_exact_reproducible_rgb():
    a = method.authored()
    b = method.authored()
    assert len(a) == 6 and all(np.array_equal(a[k], b[k]) for k in a)
    for pixels in a.values():
        for name in frozen.TRANSFORMS:
            out = frozen.transform(pixels, name)
            assert out.dtype == np.uint8 and out.shape == (256, 256, 3)
