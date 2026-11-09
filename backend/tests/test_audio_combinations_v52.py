import pytest

from scripts.audio_combinations_v52 import changes, combine, summarize


@pytest.mark.parametrize("policy", ["available_no_conflict", "priority", "any_ai", "agreement_only"])
def test_unassessed_is_never_an_original_vote_or_an_implicit_empty_check(policy):
    assert combine(("ai", "unassessed"), policy) == "unassessed"
    assert combine(("unassessed", "original"), policy) == "unassessed"
    assert combine(("inconclusive", "inconclusive"), policy) == "inconclusive"


def test_conflicting_methods_and_priority_are_explicit():
    assert combine(("ai", "original"), "available_no_conflict") == "inconclusive"
    assert combine(("ai", "original"), "agreement_only") == "inconclusive"
    assert combine(("original", "ai"), "any_ai") == "ai"
    assert combine(("original", "ai"), "priority") == "original"
    assert combine(("ai", "original"), "priority") == "ai"
    assert combine(("inconclusive", "original"), "available_no_conflict") == "original"
    assert combine(("inconclusive", "ai"), "agreement_only") == "inconclusive"


@pytest.mark.parametrize("choices,policy", [((), "single"), (("ai",), "any_ai"), (("ai", "ai"), "single"),
                                            (("clear", "ai"), "priority"), (("ai", "ai"), "unknown")])
def test_missing_choices_unknown_states_and_implicit_methods_are_refused(choices, policy):
    with pytest.raises(ValueError):
        combine(choices, policy)


def test_counts_keep_missing_and_inconclusive_in_all_denominators_and_group_false_flags():
    rows = [{"label": label, "base_id": str(n), "group": str(n // 2)}
            for n, label in enumerate((1, 1, 1, 1, 0, 0, 0, 0))]
    baseline = ["ai", "original", "inconclusive", "unassessed", "ai", "ai", "original", "unassessed"]
    got = summarize(rows, baseline)
    assert got["inputs"] == 8 and got["ai_inputs"] == got["genuine_inputs"] == 4
    assert got["false_ai"] == 2 and got["genuine_groups_flagged"] == 1
    assert got["false_original"] == 1 and got["ai_without_ai_conclusion"] == 3
    assert got["unassessed"] == 2 and got["inconclusive_assessed"] == 1
    changed = changes(rows, baseline, ["ai", "ai", "ai", "unassessed", "original", "inconclusive", "original", "unassessed"])
    assert changed == {"new_correct": 3, "lost_correct": 0, "new_ai_hits": 2, "lost_ai_hits": 0,
                       "new_false_ai": 0, "removed_false_ai": 2, "new_false_original": 0, "removed_false_original": 1}


def test_reanalysis_does_not_drop_a_missing_case():
    with pytest.raises(ValueError):
        summarize([{"label": 1, "base_id": "one", "group": "a"}], [])
