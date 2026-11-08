"""Pixel lineage requires the declared transform and every exact RGB byte."""

import io

import numpy as np
import pytest
from PIL import Image

from scripts.image_pixel_lineage_v20 import pixel_comparison, rgb, transformed_source


def image():
    return np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)


def test_full_source_match_is_not_training_admission():
    array = image()
    result = pixel_comparison(array, array.copy(), [11, 7], [11, 7], [1, 1, 8, 5])
    assert result["status"] == "exact_source_rgb" and result["geometry"] == "full_source"
    assert result["different_rgb_pixels"] == 0 and not result["training_admitted"]


def test_one_channel_at_one_pixel_is_a_visible_mismatch():
    source, published = image(), image()
    published[3, 5, 1] ^= 1
    result = pixel_comparison(source, published, [11, 7], [11, 7], [0, 0, 11, 7])
    assert result["status"] == "pixel_mismatch" and result["different_rgb_pixels"] == 1
    assert result["compared_rgb_sha256"] != result["published_rgb_sha256"]


def test_exact_metadata_crop_is_bound_to_its_offset():
    array = image()
    result = pixel_comparison(array, array[1:6, 2:10], [11, 7], [11, 7], [2, 1, 8, 5])
    assert result["status"] == "exact_source_rgb" and result["geometry"] == "declared_metadata_crop"
    wrong = pixel_comparison(array, array[0:5, 1:9], [11, 7], [11, 7], [2, 1, 8, 5])
    assert wrong["status"] == "pixel_mismatch"


def test_only_publisher_aspect_ratio_rule_can_rotate_source():
    array = image()
    rotated = np.rot90(array)
    result = pixel_comparison(array, rotated, [7, 11], [7, 11], [0, 0, 7, 11])
    assert result["status"] == "exact_source_rgb" and result["publisher_rotation_applied"]
    with pytest.raises(ValueError, match="dimensions"):
        pixel_comparison(array, rotated, [11, 7], [7, 11], [0, 0, 7, 11])


@pytest.mark.parametrize("crop", [[-1, 0, 8, 5], [0, 0, 12, 7], [0, 0, 11, 8], [1.0, 1, 8, 5], [1, 1, 8], [0, 0, 0, 0]])
def test_unbounded_or_ambiguous_crops_are_not_searched(crop):
    with pytest.raises(ValueError):
        pixel_comparison(image(), image(), [11, 7], [11, 7], crop)


def test_unexpected_image_shape_cannot_trigger_crop_search():
    with pytest.raises(ValueError, match="neither"):
        pixel_comparison(image(), image()[:3, :4], [11, 7], [11, 7], [0, 0, 8, 5])


def test_native_png_decode_does_not_apply_hidden_colour_or_resize():
    out = io.BytesIO()
    Image.fromarray(image()).save(out, format="PNG")
    assert np.array_equal(rgb(out.getvalue()), image())


def test_unhandled_exif_orientation_is_not_silently_normalized():
    value = Image.fromarray(image())
    exif = Image.Exif()
    exif[274] = 6
    out = io.BytesIO()
    value.save(out, format="JPEG", exif=exif)
    with pytest.raises(ValueError, match="EXIF"):
        rgb(out.getvalue())


def test_rotation_rule_has_explicit_native_rgb_input_contract():
    with pytest.raises(ValueError, match="Native RGB"):
        transformed_source(image().astype(np.float32), [11, 7])
