"""Source qualification preserves encoded bytes, orientation and failure coverage."""

import hashlib
import json

import pytest
from PIL import Image

from scripts import image_fresh_source_cache_v39 as cache
from scripts.image_source_groups_v22 import components


def item(identifier=1, author="authored"):
    return {"source_id": identifier, "metadata_index": identifier - 1, "metadata_body_sha256": "a" * 64,
            "attribution": {"author_url": f"https://www.flickr.com/photos/{author}/", "photo_page": f"https://www.flickr.com/photos/{author}/1234/", "license_id": 4, "license_url": "https://creativecommons.org/licenses/by/2.0/", "expected_large_source_url": f"https://live.staticflickr.com/1/{1000 + identifier}_abcdef1234_b.jpg"}}


def authored_jpeg(path, orientation=None):
    image = Image.new("RGB", (48, 32), (31, 103, 211))
    exif = image.getexif()
    if orientation is not None:
        exif[274] = orientation
    image.save(path, format="JPEG", exif=exif)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_encoded_and_native_rgb_hashes_are_distinct_and_exactly_reproducible(tmp_path):
    path = tmp_path / "source.bin"
    digest = authored_jpeg(path)
    row = cache.pixel_record(item(), path, digest)
    assert row == cache.pixel_record(item(), path, digest)
    assert row["source_jpeg_sha256"] == digest and row["rgb_sha256"] != digest
    assert row["size"] == [48, 32] and row["author"] == "authored"
    assert row["roles_assigned"] is False and row["neural_forwards"] == 0


def test_non_upright_orientation_is_not_silently_rotated(tmp_path):
    path = tmp_path / "source.bin"
    digest = authored_jpeg(path, orientation=6)
    with pytest.raises(ValueError, match="orientation"):
        cache.pixel_record(item(), path, digest)


def test_changed_bytes_or_non_jpeg_cannot_be_qualified(tmp_path):
    path = tmp_path / "source.bin"
    digest = authored_jpeg(path)
    with pytest.raises(ValueError, match="encoded source bytes"):
        cache.pixel_record(item(), path, "f" * 64)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    Image.new("RGB", (48, 32)).save(path, format="PNG")
    with pytest.raises(ValueError, match="native JPEG"):
        cache.pixel_record(item(), path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_same_pixels_link_across_different_creator_keys_and_source_ids(tmp_path):
    path = tmp_path / "source.bin"
    digest = authored_jpeg(path)
    first = cache.pixel_record(item(1, "first"), path, digest)
    second = cache.pixel_record(item(2, "second"), path, digest)
    groups, links = components([first, second])
    assert groups == [["fresh-jpeg/1", "fresh-jpeg/2"]]
    assert links["exact_source_rgb"] == 1


def test_fixed_phase_denominator_has_three_last_sources_and_no_overlap():
    plan = {"items": [item(n) for n in range(1, 164)]}
    sources = [row["source_id"] for phase in range(6) for row in cache.phase_items(plan, phase)]
    assert sources == list(range(1, 164))
    assert len(cache.phase_items(plan, 5)) == 3
    with pytest.raises(ValueError):
        cache.phase_items(plan, True)
    plan["items"].pop()
    with pytest.raises(ValueError, match="Complete fixed"):
        cache.phase_items(plan, 5)


def test_unresolved_jpeg_requests_are_kept_in_complete_denominator(tmp_path):
    selected = [item(n) for n in range(1, 164)]
    plan = {"items": selected, "cache": str(tmp_path / "cache")}
    folder = tmp_path / "fetch-000"
    folder.mkdir()
    rows = [{"source_id": row["source_id"], "metadata_index": row["metadata_index"], "url": row["attribution"]["expected_large_source_url"], "status": "unresolved", "error": {"type": "Timeout", "message": "Authored timeout"}} for row in selected[:32]]
    (folder / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert cache.review_phase(tmp_path, plan, 0) == rows
    (folder / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="Complete cache denominator"):
        cache.review_phase(tmp_path, plan, 0)
