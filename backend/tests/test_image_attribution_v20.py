"""Authored records cannot clear the wrong photo, owner, license or provider."""

from copy import deepcopy

import pytest

from scripts.image_attribution_v20 import inspect_metadata, short_id, static_identity


def example():
    source = {"source_id": 11, "publisher_role": "training", "recorded_license": {"id": 4, "url": "http://creativecommons.org/licenses/by/2.0/"}, "coco_image": {"flickr_url": "http://farm4.staticflickr.com/3350/3506835310_1e61679920_z.jpg"}}
    metadata = {"type": "photo", "flickr_type": "photo", "provider_name": "Flickr", "version": "1.0", "url": "https://live.staticflickr.com/3350/3506835310_1e61679920_b.jpg", "author_name": "Authored photographer", "author_url": "https://www.flickr.com/photos/authored/", "web_page": "https://www.flickr.com/photos/authored/3506835310/", "license_id": "4", "license_url": "https://creativecommons.org/licenses/by/2.0/", "width": 1024, "height": 768, "title": "Authored control"}
    return source, metadata


def test_base58_known_photo_reference_round_trip():
    assert short_id(3506835310) == "6kTrXL"
    with pytest.raises(ValueError):
        short_id(0)


def test_matched_provider_attribution_does_not_prove_camera_or_training_rights():
    result = inspect_metadata(*example())
    assert result["provider_attribution_and_license_match"] is True
    assert result["training_admitted"] is result["pixel_lineage_verified"] is False
    assert result["source_id"] == 11 and result["photo_id"] == 3506835310


@pytest.mark.parametrize("field,value", [
    ("type", "video"), ("provider_name", "Copied provider"), ("version", "2.0"),
    ("author_name", ""), ("author_url", "https://www.flickr.com.attacker.example/photos/authored/"),
    ("web_page", "https://www.flickr.com/photos/other/3506835310/"),
    ("web_page", "https://www.flickr.com/photos/authored/3506835311/"),
    ("license_id", "1"), ("license_url", "https://creativecommons.org/licenses/by-nc/2.0/"),
    ("license_url", "https://creativecommons.org/licenses/by/2.0/?other=1"),
    ("url", "https://live.staticflickr.com/3350/3506835311_1e61679920_b.jpg"),
    ("url", "https://live.staticflickr.com/3350/3506835310_aaaaaaaaaa_b.jpg"),
    ("url", "https://live.staticflickr.com/3351/3506835310_1e61679920_b.jpg"),
    ("width", True), ("height", 999999),
])
def test_conflicting_metadata_stays_unresolved(field, value):
    source, metadata = example()
    metadata[field] = value
    with pytest.raises(ValueError):
        inspect_metadata(source, metadata)


@pytest.mark.parametrize("url", [
    "https://live.staticflickr.com.attacker.example/3350/3506835310_1e61679920_b.jpg",
    "https://live.staticflickr.com/3350/3506835310_1e61679920_b.jpg?other=1",
    "https://live.staticflickr.com/3350/3506835310_1e61679920_b.jpg#other",
    "https://live.staticflickr.com/../3350/3506835310_1e61679920_b.jpg",
])
def test_untrusted_static_reference_is_rejected(url):
    with pytest.raises(ValueError):
        static_identity(url)


def test_historical_noncommercial_record_cannot_be_overridden_by_current_response():
    source, metadata = example()
    restricted = deepcopy(source)
    restricted["recorded_license"]["id"] = 1
    with pytest.raises(ValueError, match="commercial"):
        inspect_metadata(restricted, metadata)
