"""Rights filters reject annotation-license substitution and split ambiguity."""

import json
import zipfile

import pytest

from scripts import image_source_rights_v19 as method
from scripts import review_image_rights_v19 as review


def cc(identifier, name):
    return {identifier: {"id": identifier, "name": "A misleading display name", "url": f"http://creativecommons.org/licenses/{name}/2.0/"}}


@pytest.mark.parametrize("identifier,kind,expected", [
    (1, "by-nc-sa", "excluded_noncommercial"),
    (2, "by-nc", "excluded_noncommercial"),
    (3, "by-nc-nd", "excluded_noncommercial"),
    (4, "by", "metadata_by"),
    (5, "by-sa", "metadata_by_sa"),
    (6, "by-nd", "excluded_no_derivatives"),
])
def test_per_photo_id_and_exact_license_url_control_preliminary_filter(identifier, kind, expected):
    assert method.license_status(identifier, cc(identifier, kind)) == expected


@pytest.mark.parametrize("identifier,table", [
    (4, {}), (4, cc(4, "by-nc")), (None, cc(4, "by")),
    ("4", {"4": {"url": "http://creativecommons.org/licenses/by/2.0/"}}),
    (True, cc(1, "by-nc-sa")),
    (4, {4: {"name": "CC BY 4.0 dataset", "url": "https://creativecommons.org/licenses/by/4.0/"}}),
    (4, {4: {"url": "https://creativecommons.org.attacker.example/licenses/by/2.0/"}}),
])
def test_unknown_and_conflicting_metadata_cannot_clear_photo_rights(identifier, table):
    assert method.license_status(identifier, table) == "unknown"


@pytest.mark.parametrize("identifier", [7, 8, 99])
def test_restriction_free_or_government_labels_still_require_review(identifier):
    assert method.license_status(identifier, {identifier: {"name": "No known restrictions"}}) == "additional_rights_review"


def archive(tmp_path, train, val, train_licenses=None, val_licenses=None):
    path = tmp_path / "coco.zip"
    default = list(cc(4, "by").values())
    with zipfile.ZipFile(path, "x") as zipped:
        for role, images, licenses in (("train", train, train_licenses), ("val", val, val_licenses)):
            zipped.writestr(f"annotations/captions_{role}2017.json", json.dumps({"images": images, "licenses": default if licenses is None else licenses}))
    return path


def test_source_record_lookup_is_by_id_not_annotation_license(tmp_path):
    path = archive(tmp_path, [{"id": 101, "license": 1}], [{"id": 102, "license": 4}])
    images, _, proof = method.coco_images(path, {101, 102, 999})
    assert set(images) == {101, 102}
    assert images[101]["license"] == 1 and images[102]["license"] == 4
    assert sum(p["image_count"] for p in proof.values()) == 2


@pytest.mark.parametrize("fault", ["within", "across", "license_table", "duplicate_license"])
def test_conflicting_official_records_are_not_silently_deduplicated(tmp_path, fault):
    first, second = [{"id": 101, "license": 4}], [{"id": 102, "license": 4}]
    extra = {}
    if fault == "within":
        first *= 2
    elif fault == "across":
        second = first
    elif fault == "license_table":
        extra["val_licenses"] = list(cc(5, "by-sa").values())
    else:
        extra["train_licenses"] = list(cc(4, "by").values()) * 2
    with pytest.raises(ValueError, match="Duplicate|both|differ"):
        method.coco_images(archive(tmp_path, first, second, **extra), {101, 102})


def test_every_source_remains_unadmitted_after_metadata_filter():
    rows = [{"license_status": s} for s in ("metadata_by", "metadata_by_sa", "excluded_noncommercial", "missing_coco_record")]
    result = method.aggregate(rows)
    assert result["sources"] == 4 and result["preliminary_metadata_compatible"] == 2
    assert result["training_admitted"] == 0
    assert result["attribution_unverified"] == result["source_bytes_lineage_unverified"] == 4


def test_identical_role_lists_report_overlap_without_inventing_freshness():
    roles = {"training": {11, 12}, "validation": {11, 12}, "testing": {11, 12}}
    assert method.overlap_counts(roles) == {"training/validation": 2, "training/testing": 2, "validation/testing": 2}


def test_distinct_rows_of_same_photo_are_one_source_group(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("image_id,category\n11,cat\n11,dog\n12,bird\n")
    assert method.source_ids(path) == ({11, 12}, 3)


def test_separate_reviewer_recounts_missing_sources_and_restricted_licenses():
    images = {11: {"license": 1}, 12: {"license": 4}, 13: {"license": 5}, 14: {"license": 6}}
    table = {}
    for identifier, kind in ((1, "by-nc-sa"), (4, "by"), (5, "by-sa"), (6, "by-nd")):
        table.update(cc(identifier, kind))
    actual = review.counts({11, 12, 13, 14, 99}, images, table)
    assert actual == {
        "sources": 5, "license_status": {"excluded_no_derivatives": 1, "excluded_noncommercial": 1, "metadata_by": 1, "metadata_by_sa": 1, "missing_coco_record": 1},
        "preliminary_metadata_compatible": 2, "training_admitted": 0,
        "attribution_unverified": 5, "source_bytes_lineage_unverified": 5,
    }


@pytest.mark.parametrize("url", [
    "https://creativecommons.org.attacker.example/licenses/by/2.0/",
    "https://creativecommons.org/licenses/by/2.0/?license=noncommercial",
    "https://creativecommons.org/licenses/by/2.0/#fragment",
    "https://creativecommons.org/licenses/by/2.0/;other",
    "https://creativecommons.org/licenses/by/4.0/",
])
def test_separate_reviewer_rejects_lookalike_and_extra_license_url_parts(url):
    assert review.status(4, {4: {"url": url}}) == "unknown"


@pytest.mark.parametrize("identifier,kind", [(1, "by-nc-sa"), (2, "by-nc"), (3, "by-nc-nd"), (4, "by"), (5, "by-sa"), (6, "by-nd")])
def test_independent_license_paths_agree_on_exact_primary_schemas(identifier, kind):
    assert method.license_status(identifier, cc(identifier, kind)) == review.status(identifier, cc(identifier, kind))
