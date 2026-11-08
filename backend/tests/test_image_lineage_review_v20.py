"""Separate source/geometry checks on authored photos and metadata."""

import io

import numpy as np
import pytest
from PIL import Image

from scripts.review_image_attribution_v20 import reconstruct
from scripts.review_image_pixel_lineage_v20 import decode, geometry


def case():
    return {"coco_size": [11, 7], "publisher_full_size": [11, 7], "publisher_crop": [2, 1, 8, 5]}


def array():
    return np.arange(11 * 7 * 3, dtype=np.uint8).reshape(7, 11, 3)


def test_independent_offset_crop_and_single_pixel_recount():
    source = array()
    reference = source[1:6, 2:10].copy()
    exact = geometry(source, reference, case())
    assert exact["status"] == "exact_source_rgb" and exact["geometry"] == "declared_metadata_crop"
    reference[2, 3, 2] ^= 1
    changed = geometry(source, reference, case())
    assert changed["different_rgb_pixels"] == 1 and changed["status"] == "pixel_mismatch"


def test_separate_integer_rotation_rule_reconstructs_known_clockwise_storage():
    source = array()
    rotated = np.rot90(source)
    values = {"coco_size": [7, 11], "publisher_full_size": [7, 11], "publisher_crop": [0, 0, 7, 11]}
    result = geometry(source, rotated, values)
    assert result["status"] == "exact_source_rgb" and result["publisher_rotation_applied"]


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_two_actual_decoders_agree_on_authored_encoded_pixels(format):
    output = io.BytesIO()
    Image.fromarray(array()).save(output, format=format)
    result = decode(output.getvalue())
    assert result.shape == array().shape and result.dtype == np.uint8


def metadata():
    source = {"source_id": 11, "publisher_role": "training", "coco_image": {"flickr_url": "http://farm1.staticflickr.com/1/12345_abcdef_z.jpg"}, "recorded_license": {"id": 4, "url": "http://creativecommons.org/licenses/by/2.0/"}}
    value = {"type": "photo", "flickr_type": "photo", "provider_name": "Flickr", "version": "1.0", "url": "https://live.staticflickr.com/1/12345_abcdef_b.jpg", "author_name": "Authored creator", "author_url": "https://www.flickr.com/photos/authored/", "web_page": "https://www.flickr.com/photos/authored/12345/", "license_id": "4", "license_url": "https://creativecommons.org/licenses/by/2.0/", "width": 1024, "height": 768, "title": "Authored photo"}
    return source, value


def test_separate_attribution_reconstruction_keeps_permission_unestablished():
    result = reconstruct(*metadata())
    assert result["source_id"] == 11 and result["photo_id"] == 12345
    assert result["training_admitted"] is result["pixel_lineage_verified"] is False


@pytest.mark.parametrize("key,value", [
    ("license_id", "1"), ("license_url", "https://creativecommons.org/licenses/by-nc/2.0/"),
    ("web_page", "https://www.flickr.com/photos/other/12345/"),
    ("url", "https://live.staticflickr.com/1/12346_abcdef_b.jpg"),
    ("url", "https://live.staticflickr.com/1/12345_abcdee_b.jpg"),
    ("author_url", "https://www.flickr.com.attacker.example/photos/authored/"),
    ("type", "video"), ("width", True),
])
def test_separate_attribution_rejects_the_wrong_source_or_license(key, value):
    source, doc = metadata()
    doc[key] = value
    with pytest.raises(ValueError):
        reconstruct(source, doc)
