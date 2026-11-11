from __future__ import annotations

import pytest

from scripts.writing_shared_anchor_ranking import selections, source_windows, summary, unique_top

SENTENCES = (
    "The community kitchen kept a paper record of the ingredients donated during the weekend market.",
    "We weighed the bags in the morning so everyone could see how much flour was available for bread.",
    "The volunteers moved the remaining boxes away from the oven after an afternoon delivery filled the shelves.",
    "Everyone took their turn putting the signed receipts into the office binder before closing the building for the night.",
)


def test_shared_windows_use_identical_ordered_anchors_with_exact_focus_offsets():
    body = " ".join(SENTENCES)
    native_left, native_right, shared_left, shared_right = source_windows(body)
    def measured(spans):
        return tuple(body[start:end] for start, end in spans)

    assert measured(native_left) == SENTENCES[:3]
    assert measured(native_right) == SENTENCES[1:]
    assert measured(shared_left) == (SENTENCES[0], SENTENCES[1], SENTENCES[3])
    assert measured(shared_right) == (SENTENCES[0], SENTENCES[2], SENTENCES[3])
    assert source_windows("\n\n".join(SENTENCES)) == []


def test_disagreement_only_removes_an_existing_mark():
    agree = selections((0.7, 0.4), (0.8, 0.2))
    assert agree["matched"] == (0, 1)
    assert agree["matched_top_one"] == agree["matched_agreement"] == (0,)
    conflict = selections((0.7, 0.4), (0.2, 0.8))
    assert conflict["native_rank"] == conflict["matched_top_one"] == (0,)
    assert conflict["shared_rank"] == (1,)
    assert conflict["agreement_rank"] == conflict["matched_agreement"] == ()
    below = selections((0.07, 0.04), (0.8, 0.2))
    assert below["agreement_rank"] == (0,)
    assert below["matched_agreement"] == ()


def test_exact_ties_and_invalid_values_are_not_first_position_agreement():
    assert unique_top((0.8, 0.8)) is None
    assert selections((0.5, 0.5), (0.2, 0.2))["agreement_rank"] == ()
    for pair in ((float("nan"), 0.2), (-0.1, 0.2), (0.2, 1.1)):
        with pytest.raises(ValueError, match="finite raw"):
            unique_top(pair)


def test_rank_only_human_selection_does_not_count_as_a_matched_human_mark():
    body = " ".join(SENTENCES)
    windows = source_windows(body)
    response = {
        "focus_spans": [windows[0][1], windows[1][1]],
        "windows": [{"response": score} for score in (0.06, 0.03, 0.8, 0.2)],
    }
    row = {"group": "test", "arm": "human_sentence", "status": "constructed",
           "text": body, "ai_spans": [], "ordinary_spans": [list(windows[0][1])]}
    out = summary([(row, response)])
    assert out["counts"]["human_sentence"]["agreement_rank"]["target_selected"] == 1
    assert out["counts"]["human_sentence"]["matched_agreement"]["target_selected"] == 0
    assert all(total == 0 for total in out["human_source_groups_marked"].values())
