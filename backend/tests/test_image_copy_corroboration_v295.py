import hashlib

import numpy as np
import pytest

from scripts import image_copy_corroboration_v295 as trial


def test_authored_fixtures_keep_existing_regions_and_separate_outcomes():
    result = trial.authored()
    assert result["planned_views"] == result["completed_views"] == result["candidate_retained_views"] == 14
    assert result["evaluated_views"] == 12
    assert result["outcomes"] == {
        "copy_consistent": 6,
        "ambiguous_repetition": 2,
        "no_donor": 4,
        "insufficient_texture": 2,
    }
    assert result["ordinary_copy_corroborated"] == 6
    assert result["authored_noncopy_overlap_views"] == result["untouched_copy_supported"] == 0
    assert result["generator_calls"] == 0

    by_view = {(row["fixture"], row["encoding"]): row for row in result["rows"]}
    for encoding in ("native", "jpeg90"):
        translated = by_view["translated_copy", encoding]
        reflected = by_view["reflected_copy", encoding]
        mixed_copy = by_view["mixed_copy_region", encoding]
        mixed_noncopy = by_view["independent_replacement", encoding]
        repeated = by_view["repeated_texture", encoding]
        assert translated["donor"]["transform"] == "translation"
        assert reflected["donor"]["transform"] == mixed_copy["donor"]["transform"] == "horizontal_reflection"
        for row in (translated, reflected, mixed_copy):
            assert row["proposal_box_xywh"] == [31, 33, 64, 64]
            assert row["donor"]["box_xywh"] == [151, 138, 64, 64]
            assert row["copy_consistent_overlap_pixels"] == {"ordinary_only": 4096, "authored_noncopy": 0}
        assert mixed_noncopy["proposal_box_xywh"] == [145, 25, 64, 64]
        assert mixed_noncopy["authored_noncopy_pixels_under_proposal"] == 4096
        assert mixed_noncopy["status"] == "no_donor"
        assert repeated["status"] == "ambiguous_repetition"
        assert repeated["donor"] is None
        assert repeated["donor_margin"] < trial.MIN_DISTINCT_DONOR_MARGIN
        assert by_view["flat_selection", encoding]["status"] == "insufficient_texture"
        assert by_view["untouched_candidate", encoding]["status"] == "no_donor"


def test_irregular_proposal_keeps_exact_mask_and_rejects_misaligned_inputs():
    _, image, original_mask, *_ = trial.authored_inputs()[0]
    proposal = original_mask.copy()
    proposal[45:47, 45:47] = False
    before = proposal.copy()
    result = trial.corroborate(image, proposal)

    assert result["status"] == "copy_consistent"
    assert result["proposal_pixels"] == 4092
    assert result["proposal_mask_sha256"] == hashlib.sha256(before.tobytes()).hexdigest()
    assert np.array_equal(proposal, before)
    with pytest.raises(ValueError, match="source-coordinate"):
        trial.corroborate(image, proposal[:, :-1])
    with pytest.raises(ValueError, match="source-coordinate"):
        trial.corroborate(image, proposal.astype(np.uint8))


def test_large_submission_reports_unassessed_geometry():
    shape = (128, trial.SIDE_LIMIT + 1)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    proposal = trial.mask_for(shape, (25, 25, 48, 48))

    result = trial.corroborate(image, proposal)

    assert result["status"] == "unsupported_geometry"
    assert result["proposal_box_xywh"] == [25, 25, 48, 48]
    assert result["donor"] is None
