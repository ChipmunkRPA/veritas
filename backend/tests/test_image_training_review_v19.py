"""A review queue must not promote restricted, consumed or leaking photos."""

from copy import deepcopy

import pytest

from scripts.prepare_image_training_review_v19 import select


def row(identifier=11, role="training"):
    return {
        "source_id": identifier, "license_status": "metadata_by", "recorded_license": {}, "coco_image": {},
        "publisher_roles": {"original": [role], "firefly": [role], "sd2": [role], "flux-fill": [role], "flux-original": ["testing", "training", "validation"]},
        "consumed_local_evaluation": False, "training_admitted": False,
        "attribution_verified": False, "source_bytes_lineage_verified": False,
    }


def test_common_source_queue_does_not_use_bad_flux_original_role_names():
    result = select([row()])
    assert len(result) == 1 and result[0]["publisher_role"] == "training"
    assert result[0]["ready_for_training"] is False
    assert len(result[0]["required_checks"]) == 4


@pytest.mark.parametrize("kind", ["excluded_noncommercial", "excluded_no_derivatives", "unknown", "additional_rights_review", "missing_coco_record"])
def test_restricted_or_unknown_photo_never_enters_review_training_pool(kind):
    item = row()
    item["license_status"] = kind
    assert select([item]) == []


def test_consumed_test_photo_cannot_reappear_as_new_confirmation():
    item = row(role="testing")
    item["consumed_local_evaluation"] = True
    assert select([item]) == []


def test_missing_generator_sources_are_not_silently_substituted():
    item = row()
    item["publisher_roles"]["flux-fill"] = []
    assert select([item]) == []


@pytest.mark.parametrize("family,roles", [("original", ["training", "testing"]), ("sd2", ["training", "testing"]), ("firefly", ["validation"])])
def test_role_overlap_or_inconsistency_refuses_queue(family, roles):
    item = row()
    item["publisher_roles"][family] = roles
    with pytest.raises(ValueError, match="role|leakage"):
        select([item])


def test_source_photo_cannot_be_counted_twice_by_category():
    item = row()
    with pytest.raises(ValueError, match="One record"):
        select([item, deepcopy(item)])


@pytest.mark.parametrize("claim", ["training_admitted", "attribution_verified", "source_bytes_lineage_verified"])
def test_no_permission_is_added_by_changing_a_claim_in_this_audit(claim):
    item = row()
    item[claim] = True
    with pytest.raises(ValueError, match="separately versioned"):
        select([item])


def test_review_order_does_not_depend_on_input_order():
    inputs = [row(11), row(12), row(13)]
    assert select(inputs) == select(list(reversed(inputs)))
