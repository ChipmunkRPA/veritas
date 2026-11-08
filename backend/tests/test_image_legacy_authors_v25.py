"""Public owner references cannot supply a training licence or conceal a wrong photo."""

import copy

import pytest

from scripts.image_legacy_authors_v25 import inspect


def reference():
    expected = {"photo_id": 1234, "server": "99", "secret": "abcdef", "size": "z"}
    metadata = {"type": "photo", "flickr_type": "photo", "provider_name": "Flickr", "version": "1.0", "url": "https://live.staticflickr.com/99/1234_abcdef_z.jpg", "author_url": "https://www.flickr.com/photos/1@N01/", "web_page": "https://www.flickr.com/photos/1@N01/1234/", "author_name": "Authored profile", "license_id": "all-rights-reserved"}
    return expected, metadata


def test_exact_public_owner_correspondence_is_not_a_commercial_photo_grant():
    expected, metadata = reference()
    owner = inspect(expected, metadata)
    assert owner["owner"] == "1@n01" and owner["license_or_camera_authentication_assessed"] is False
    assert "training_admitted" not in owner


@pytest.mark.parametrize(("field", "value"), [("type", "video"), ("flickr_type", "video"), ("provider_name", "Other"), ("version", "2.0"), ("url", "https://live.staticflickr.com/99/1235_abcdef_z.jpg"), ("url", "https://live.staticflickr.com/98/1234_abcdef_z.jpg"), ("url", "https://live.staticflickr.com/99/1234_abcdff_z.jpg"), ("author_url", "http://www.flickr.com/photos/1@N01/"), ("web_page", "https://www.flickr.com/photos/other/1234/"), ("author_name", " ")])
def test_identity_or_profile_mismatch_stays_unresolved(field, value):
    expected, metadata = reference()
    metadata = copy.deepcopy(metadata)
    metadata[field] = value
    with pytest.raises(ValueError):
        inspect(expected, metadata)
