from copy import deepcopy

import pytest

from scripts import image_bg_real_roster_v218 as roster


def candidate(number, split, *, owner=None, photo=None, byte_hash=None):
    source = f"oi_{number:016x}"
    key = source[3]
    owner = owner or f"photographer-{number}"
    photo = photo or str(1000 + number)
    prefix = f"v2/images/{key}/{source}"
    common = {
        "source_image_id": source,
        "split": split,
        "width": "768",
        "height": "512",
        "robustness": "none",
        "encoding": "png",
        "jpeg_quality": "none",
        "donor_id": "",
    }
    edit = {
        **common,
        "family": "F5",
        "generator": "big-lama",
        "edit_kind": "object_removal",
        "label": "1",
        "sample_id": source + "_F5",
        "base_sample_id": source + "_F5",
        "image_path": prefix + "_F5.png",
        "mask_edit_path": f"v2/masks/edit/{key}/{source}_F5_edit.png",
        "sha256": f"{number + 1000:064x}",
        "edit_area": "0.05",
    }
    original = {
        **common,
        "family": "A0",
        "generator": "none",
        "label": "0",
        "sample_id": source + "_A0",
        "base_sample_id": source + "_A0",
        "image_path": prefix + "_A0.png",
        "sha256": byte_hash or f"{number:064x}",
    }
    recipient = {**common, "image_id": source[3:]}
    licence = {
        "role": "recipient",
        "bgreal_id": source,
        "ImageID": source[3:],
        "Author": "Example author",
        "AuthorProfileURL": f"https://www.flickr.com/people/{owner}/",
        "OriginalLandingURL": f"https://www.flickr.com/photos/{owner}/{photo}/in/photostream/",
        "OriginalURL": f"https://c8.staticflickr.com/7/123/{photo}_key_o.jpg",
        "License": roster.PHOTO_LICENCE,
        "Title": "An authored photo",
    }
    return edit, original, recipient, licence


def combine(*cases):
    membership = {split: set() for split in roster.SPLITS}
    manifest, recipients, licenses = [], [], []
    for edit, original, recipient, licence in cases:
        membership[edit["split"]].update({edit["sample_id"], original["sample_id"]})
        manifest.extend((edit, original))
        recipients.append(recipient)
        licenses.append(licence)
    return manifest, licenses, recipients, membership


def test_bounded_pool_prioritizes_test_and_remains_stable_on_input_reorder(monkeypatch):
    monkeypatch.setattr(roster, "PRIMARY", 2)
    monkeypatch.setattr(roster, "BACKUP", 1)
    cases = [
        candidate(1, "train"),
        candidate(2, "val"),
        candidate(3, "test_id"),
        candidate(4, "test_background_ood"),
        candidate(5, "train"),
    ]
    result = roster.plan(*combine(*cases), set())
    assert result == roster.plan(*combine(*reversed(cases)), set())
    counts, private = result
    assert counts["primary_planned"] == 2 and counts["backup_planned"] == 1
    assert counts["eligible_but_outside_bounded_pool"] == 2
    assert counts["backup_planned_by_split"] == {"val": 1}
    assert [row["tier"] for row in private["ordered_candidates"]] == ["primary", "primary", "backup"]
    assert all(row["semantic_mask"]["sha256"] is None for row in private["ordered_candidates"])
    assert "recipient" not in counts and "flickr_photo_id" not in counts


def test_prior_exposure_excludes_a_whole_transitively_linked_component():
    linked = [
        candidate(1, "test_id", owner="known"),
        candidate(2, "train", owner="known"),
        candidate(3, "val", owner="unseen", byte_hash=f"{2:064x}"),
        candidate(4, "test_subject_ood"),
    ]
    counts, private = roster.plan(*combine(*linked), {("owner", "known")})
    assert counts["component_exclusions"] == {"prior_exposure_component": 3}
    assert [row["flickr_photo_id"] for row in private["ordered_candidates"]] == ["1004"]


def test_metadata_disqualifications_are_counted_and_candidates_are_never_silently_swapped():
    invalid = [candidate(number, "test_id") for number in range(1, 5)]
    invalid[0][3]["AuthorProfileURL"] = "https://www.flickr.com/people/unknown/"
    invalid[1][0]["mask_edit_path"] = "../../unrelated.png"
    invalid[2][1]["width"] = "760"
    invalid[3][0]["generator"] = "sd15"
    counts, private = roster.plan(*combine(*invalid), set())
    assert counts["publisher_candidates_by_split"] == {"test_id": 3}
    assert counts["metadata_rejections"] == {
        "Publisher image or mask path is outside the expected asset location": 1,
        "Unmatched or unsupported dimensions": 1,
        "Unresolved photo or owner alias": 1,
    }
    assert counts["primary_planned"] == 0 and len(private["excluded"]) == 3


def test_membership_and_license_require_exact_published_recipient_pair():
    edit, original, recipient, licence = candidate(1, "test_id")
    memberships = {"test_id": {original["sample_id"]}}
    with pytest.raises(ValueError, match="declared split"):
        roster.pair(edit, original, licence, recipient, memberships)
    memberships["test_id"].add(edit["sample_id"])
    invalid_licence = deepcopy(licence)
    invalid_licence["License"] = "https://creativecommons.org/licenses/by-nc/2.0/"
    with pytest.raises(ValueError, match="licence attribution"):
        roster.pair(edit, original, invalid_licence, recipient, memberships)
    result = roster.pair(edit, original, licence, recipient, memberships)
    assert result["size"] == [768, 512] and result["flickr_owner"] == "photographer-1"


def test_historical_http_is_for_identity_parsing_only():
    value = "http://farm1.staticflickr.com/123/1001_key_o.jpg"
    assert roster.static_photo_id(value, historical=True) == "1001"
    with pytest.raises(ValueError, match="Unrecognized attribution URL"):
        roster.static_photo_id(value)
