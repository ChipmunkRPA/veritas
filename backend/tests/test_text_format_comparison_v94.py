"""Source-aligned response changes, genuine missing coverage, fixed runtime and no overwrites."""

import math

import pytest

from scripts import text_format_comparison_v94 as comparison
from scripts import text_format_runtime_v94 as runtime


def view():
    return {"text": "A B", "source_positions": [0, None, 5]}


def test_only_measured_original_characters_receive_scores_or_highlights():
    mask, scores = comparison.projected(view(), [{"start": 1, "end": 3, "score": 0.7}], 0.6)
    assert mask == {5} and scores == {5: 0.7}
    assert 0 not in scores  # uncovered A is missing, not a measured original probability
    assert comparison.projected(view(), [{"start": 1, "end": 2, "score": 0.99}], 0.5) == (set(), {})


def test_overlapping_proposals_use_strongest_actual_covering_response_without_extra_positions():
    proposals = [{"start": 0, "end": 3, "score": 0.3}, {"start": 2, "end": 3, "score": 0.8}]
    mask, scores = comparison.projected(view(), proposals, 0.7)
    assert mask == {5} and scores == {0: 0.3, 5: 0.8}
    assert comparison.projected(view(), proposals, math.nextafter(1.0, math.inf))[0] == set()


@pytest.mark.parametrize(
    "proposal",
    [
        {"start": True, "end": 2, "score": 0.5},
        {"start": 0, "end": 4, "score": 0.5},
        {"start": 0, "end": 3, "score": float("nan")},
        {"start": 0, "end": 3, "score": True},
    ],
)
def test_invalid_positions_or_scores_cannot_enter_comparisons(proposal):
    with pytest.raises(ValueError):
        comparison.projected(view(), [proposal], 0.5)


def test_repeated_or_unmapped_retained_positions_are_refused():
    for mapping in ([0, None, 0], [0, None, None]):
        with pytest.raises(ValueError):
            comparison.projected(
                {"text": "A B", "source_positions": mapping}, [{"start": 0, "end": 3, "score": 0.5}], 0.5
            )


def test_changed_highlights_and_changed_scored_coverage_remain_separate():
    before = ({0}, {0: 0.9, 5: 0.4})
    after = ({5}, {5: 0.8})
    result = comparison.difference(before, after, {0: 0, 5: 1})
    assert result["highlight_changed"] and result["lost_ai_characters"] == 0
    assert result["removed_human_characters"] == 1 and result["added_ai_characters"] == 1
    assert result["shared_scored_characters"] == 1 and result["only_original_scored_characters"] == 1
    assert result["absolute_score_change_sum"] == pytest.approx(0.4)
    assert result["highlight_jaccard"] == 0


def test_unassessed_inputs_cannot_become_empty_agreement_or_original_verdict():
    assert comparison.difference(None, (set(), {}), {}) == {"status": "unassessed"}
    assert comparison.difference(None, None, {}) == {"status": "unassessed"}
    observed = comparison.difference((set(), {}), (set(), {}), {})
    assert observed["status"] == "ok" and observed["highlight_jaccard"] == 1
    assert observed["maximum_score_change"] is None and observed["shared_scored_characters"] == 0


def test_combined_masks_have_no_invented_combined_probability():
    response = {
        "status": "ok",
        "arms": {
            "current": [{"start": 0, "end": 1, "score": 0.9}],
            "interval": [{"start": 2, "end": 3, "score": 0.9}],
        },
    }
    result = comparison.methods(view(), response, {"current": {"cut": 0.5}, "interval": {"cut": 0.5}})
    assert result["union"] == ({0, 5}, {}) and result["intersection"] == (set(), {})
    assert all(v is None for v in comparison.methods(view(), {"status": "unassessed"}, {}).values())


def test_labels_for_an_unknown_source_position_are_not_inferred():
    with pytest.raises(ValueError, match="Known original"):
        comparison.difference(({5}, {5: 0.9}), ({5}, {5: 0.9}), {0: 0})


def test_neural_runtime_preserves_larger_initial_headroom_and_never_retries_partial_output(
    tmp_path, monkeypatch
):
    assert runtime.original.START_RAM == 6 * 1024**3 and runtime.original.RSS_LIMIT == 6 * 1024**3
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    path = tmp_path / "batch-00.jsonl"
    path.write_text("preserved partial response\n")
    with pytest.raises(FileExistsError):
        runtime.run("score", 0)
    assert path.read_text() == "preserved partial response\n" and not (tmp_path / "batch-00-profile").exists()
    for command, index in [("prepare", None), ("score", True), ("score", 8)]:
        with pytest.raises(ValueError):
            runtime.run(command, index)
