import numpy as np
import pytest

from scripts import video_faceswapai_edge_review_v194 as trial

BEFORE = np.array([1.0, 0.0])
CORE = np.array([0.0, 1.0])
CENTERS = {"before": [10, 20], "during": [30, 40, 50], "after": [60, 70]}


def vectors(first_flank=CORE, second_flank=BEFORE):
    output = {}
    for centers, vector in (
        (CENTERS["before"], BEFORE),
        (CENTERS["during"], CORE),
        ([60], first_flank),
        ([70], second_flank),
    ):
        output.update(dict.fromkeys(trial.positions(centers), vector))
    return output


def test_source_bound_end_can_extend_only_to_a_measured_core_like_crop():
    result = trial.refine(CENTERS, vectors())
    assert result["status"] == "measured_return_bracket"
    assert result["proposed_last_supported_frame"] == 61
    assert result["first_measured_return_frame"] == 69
    measured = [row["frame"] for row in result["observations"]]
    assert measured == [49, 50, 51, 59, 60, 61, 69, 70, 71]
    assert not set(measured) & set(trial.positions([10, 20, 30, 40]))


def test_early_return_retains_only_the_final_measured_core_triplet_tail():
    result = trial.refine(CENTERS, vectors(first_flank=BEFORE))
    assert result["proposed_last_supported_frame"] == 51
    assert result["first_measured_return_frame"] == 59


def test_mixed_or_missing_return_preserves_the_existing_range():
    ambiguous = vectors()
    ambiguous[60] = np.array([1.0, 1.0])
    for features in (ambiguous, vectors(second_flank=CORE)):
        result = trial.refine(CENTERS, features)
        assert result["status"] == "no_stable_return_bracket"
        assert result["proposed_last_supported_frame"] == 50
        assert result["first_measured_return_frame"] is None


def test_any_overlapping_reference_or_held_out_triplet_is_unsupported():
    for centers in (
        {**CENTERS, "during": [22, 40, 50]},
        {**CENTERS, "during": [30, 48, 50]},
        {**CENTERS, "after": [52, 70]},
    ):
        result = trial.refine(centers, {})
        assert result["status"] == "overlapping_reference_or_test_triplets"
        assert result["proposed_last_supported_frame"] == 50


def test_comparison_truth_is_scored_only_after_the_proposed_edge():
    proposal = trial.refine(CENTERS, vectors())
    pts = list(range(0, 512 * 100, 512))
    swap = trial.summarize(30, 50, proposal, list(range(28, 65)), True, pts)
    control = trial.summarize(30, 50, proposal, [], False, pts)
    assert swap["proposed"]["last_supported"] == control["proposed"]["last_supported"]
    assert swap["proposed"]["missed_native_positions"] < swap["existing"]["missed_native_positions"]
    assert control["proposed"]["outside_native_positions"] == 32
    assert swap["comparison_target"] == "publisher_swap"
    assert control["comparison_target"] == "ordinary_changes_only"
    assert swap["appended_unmeasured_positions"] == 7
    assert swap["unmeasured_positions_in_return_bracket"] == 7
    assert [row["native_frame"] for row in swap["appended_directly_measured_core_like"]] == [51, 59, 60, 61]


def test_immutable_database_requires_no_pending_wal(tmp_path):
    root = tmp_path.resolve()
    data = root / "data"
    data.mkdir()
    db, wal, shm = (data / name for name in ("veritas.db", "veritas.db-wal", "veritas.db-shm"))
    db.write_bytes(b"placeholder")
    wal.touch()
    shm.write_bytes(b"an existing read-only sidecar is permitted")
    assert trial.database_path(root) == db
    wal.write_bytes(b"uncheckpointed")
    with pytest.raises(ValueError, match="checkpointed"):
        trial.database_path(root)
