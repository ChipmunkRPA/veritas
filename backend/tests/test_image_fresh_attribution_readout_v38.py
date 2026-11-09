"""Known creator keys stay linked; unknown or malformed fields are not cleared."""

import pytest

from scripts.image_fresh_attribution_readout_v38 import creator_counts, known_creator_keys


def test_existing_graph_uses_keys_and_keeps_unknowns():
    keys, unknown = known_creator_keys([{"author": "authored"}, {"author": "12345@n01"}, {"author": "authored"}, {"author": None}])
    assert keys == {"authored", "12345@n01"} and unknown == 1


@pytest.mark.parametrize("value", ["https://www.flickr.com/photos/authored/", "Authored", "", "foo/bar", "foo?x=1", "foo#x", "two words", 12])
def test_wrong_schema_fails_instead_of_becoming_unlinked(value):
    with pytest.raises(ValueError):
        known_creator_keys([{"author": value}])


def test_new_profile_urls_are_normalized_against_existing_keys():
    records = [
        {"status": "matched_current_provider_metadata", "attribution": {"author_url": "https://www.flickr.com/photos/Authored/"}},
        {"status": "matched_current_provider_metadata", "attribution": {"author_url": "https://www.flickr.com/photos/new/"}},
        {"status": "matched_current_provider_metadata", "attribution": {"author_url": "https://www.flickr.com/photos/new/"}},
        {"status": "unresolved", "error": {"type": "Timeout", "message": "Authored failure"}},
    ]
    result = creator_counts(records, [{"author": "authored"}, {"author": None}])
    assert result == {"matched_linked_to_known_creator_keys": 1, "matched_without_known_creator_link": 2, "distinct_unlinked_creator_keys": 1, "prior_sources_with_unknown_creator": 1}


def test_missing_author_field_is_not_silently_assumed_independent():
    with pytest.raises(KeyError):
        known_creator_keys([{}])
