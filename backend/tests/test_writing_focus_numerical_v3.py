"""Hand-computed ranks, ties and character denominators for the arithmetic audit."""
import pytest

from scripts.review_writing_focus_v3 import fractions, one_document, rank_auc, rank_correlation, ranks, same


def test_midranks_and_auc_credit_ties_by_half():
    assert ranks([9, 3, 3, 6]) == [4, 1.5, 1.5, 3]
    assert rank_auc([False, True], [0.5, 0.5]) == 0.5
    assert rank_auc([False, False, True, True], [1, 3, 3, 4]) == 0.875
    assert rank_auc([True, False], [0.1, 0.9]) == 0


def test_rank_correlation_respects_order_and_ties():
    assert rank_correlation([2, 2, 8, 10], [1, 1, 9, 12]) == pytest.approx(1)
    assert rank_correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1)


def test_whole_character_union_includes_unrecovered_bot_and_flagged_human():
    source = {"total_ai_characters": 10, "total_human_characters": 9,
              "foci": [{"ai_characters": 8, "human_characters": 0},
                       {"ai_characters": 2, "human_characters": 3},
                       {"ai_characters": 0, "human_characters": 6}]}
    result = one_document(source, [0.4, 0.8, 0.9], 0.8)
    assert result["located_bot_characters"] == 2
    assert result["highlighted_human_characters"] == 9
    rates = fractions(result)
    assert rates["character_iou"] == 2 / 19
    assert rates["bot_character_precision"] == 2 / 11
    assert rates["pure_bot_recovery_rate"] == 0
    assert rates["pure_human_origin_highlight_rate"] == 1


@pytest.mark.parametrize("value", [float("nan"), 0.40001])
def test_nonfinite_or_material_numeric_change_refused(value):
    with pytest.raises(ValueError):
        same({"score": value}, {"score": 0.4})


def test_changed_denominator_and_missing_counts_refused():
    with pytest.raises(ValueError):
        same({"bot": 7, "human": 2}, {"bot": 8, "human": 2})
    with pytest.raises(ValueError):
        same({"bot": 8}, {"bot": 8, "human": 2})
