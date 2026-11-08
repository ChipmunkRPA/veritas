"""Leakage groups must be transitive, stable and conservative on weak thumbnails."""

import numpy as np
import pytest
from PIL import Image

from scripts import image_source_groups_v22 as grouping
from scripts.image_source_groups_v22 import author_key, components, near, sketch


def row(key, source_id, author=None, rgb=None, thumbnail=None, size=(100, 100)):
    return {"key": key, "source_id": source_id, "author": author, "rgb_sha256": rgb or key, "sketch": thumbnail or ("a" * 64), "size": list(size)}


def test_authored_horizontal_checker_has_declared_bit_order():
    values = np.tile((np.arange(17) % 2 * 255).astype(np.uint8), (16, 1))
    assert f"{sketch(Image.fromarray(values)):064x}" == "a" * 64


def test_transitive_creator_and_exact_pixel_links_cannot_cross_components():
    # Different informative sketches prevent an accidental thumbnail-only link.
    rows = [row("a", 1, "creator1", thumbnail="a" * 64), row("b", 2, "creator1", "pixels2", thumbnail="5" * 64), row("c", 3, "creator2", "pixels2", thumbnail="5" * 64), row("d", 4, "creator3", thumbnail="3" * 64)]
    groups, _ = components(rows)
    assert groups == [["a", "b", "c"], ["d"]]
    assert components(list(reversed(rows))) == components(rows)


def test_same_source_id_connects_prior_source_even_if_pixels_differ():
    rows = [row("new/a", 12, thumbnail="a" * 64), row("old/a", 12, thumbnail="5" * 64)]
    groups, links = components(rows)
    assert groups == [["new/a", "old/a"]] and links["same_source_id"] == 1


def test_missing_authors_and_uninformative_thumbnails_do_not_merge_unrelated_sources():
    rows = [row("a", 1, thumbnail="0" * 64), row("b", 2, thumbnail="0" * 64)]
    assert components(rows) == ([["a"], ["b"]], {})


def test_near_thumbnail_boundary_is_inclusive_at_eight_bits_only():
    base = int("a" * 64, 16)
    a = row("a", 1)
    assert near(a, row("b", 2, thumbnail=f"{base ^ 255:064x}"))
    assert not near(a, row("c", 3, thumbnail=f"{base ^ 511:064x}"))


def test_aspect_disagreement_prevents_thumbnail_link():
    a = row("a", 1)
    assert near(a, row("b", 2, size=(102, 100)))
    assert not near(a, row("b", 2, size=(103, 100)))


def test_canonical_profile_key_joins_trailing_slash_and_case_variants():
    assert author_key("https://www.flickr.com/photos/123@N01/") == author_key("https://www.flickr.com/photos/123@n01")
    assert author_key(None) is None


@pytest.mark.parametrize("url", ["http://www.flickr.com/photos/a/", "https://other.invalid/photos/a/", "https://www.flickr.com/photos/a/1", "https://www.flickr.com/photos/a/?x=1"])
def test_uncertain_profile_references_are_not_silently_canonicalized(url):
    with pytest.raises(ValueError):
        author_key(url)


def test_duplicate_graph_keys_are_refused():
    with pytest.raises(ValueError, match="Unique"):
        components([row("a", 1), row("a", 2)])


def test_legacy_string_source_ids_are_normalized_before_duplicate_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(grouping, "ROOT", tmp_path)
    path = tmp_path / "authored.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path)
    record = grouping.source_record("old/10", "10", path, grouping.sha(path), None, "prior_consumed", True)
    assert type(record["source_id"]) is int and record["source_id"] == 10


def test_boolean_source_id_is_not_an_integer_identity(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        grouping.source_record("old/1", True, tmp_path / "unused", "unused", None, "prior_consumed", True)
