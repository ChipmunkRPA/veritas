import pytest

from scripts.audio_shared_confirmation_v62 import METHODS, decisions, parity_plan, parity_selection


def sample(current, standalone, complement, old="inconclusive", status="ok"):
    return {"status": status, "original_decision": current, "candidate_decision": old,
            "candidates": {"standalone": {"decision": standalone}, "complement": {"decision": complement}}}


def test_skipped_input_never_becomes_original_even_with_stale_scores():
    value = decisions(sample("original", "original", "original", "original", "not_applicable"))
    assert set(value.values()) == {"unassessed"}
    assert set(value) == {*METHODS, "v7"}


def test_failed_input_is_rejected_instead_of_promoted_to_a_direction():
    with pytest.raises(ValueError, match="Failed"):
        decisions(sample("ai", "ai", "ai", status="failed"))


def test_complementary_head_recovers_a_case_without_voting_a_probability():
    value = decisions(sample("inconclusive", "inconclusive", "ai"))
    assert value["current"] == "inconclusive"
    assert value["pair_conflict_abstain"] == "ai"
    assert value["three_any_ai"] == "ai"
    assert all(isinstance(item, str) for item in value.values())


def test_conflicting_heads_keep_distinct_comparison_policies():
    value = decisions(sample("ai", "original", "inconclusive", old="original"))
    assert value["three_conflict_abstain"] == "inconclusive"
    assert value["three_any_ai"] == "ai"
    assert value["v7"] == "original"


def test_parity_selection_covers_each_truth_and_condition_without_results():
    rows = [{"id": f"{condition}-{label}-{suffix}", "role": "development", "condition": condition,
             "label": label} for condition in range(9) for label in (0, 1) for suffix in ("b", "a")]
    selected = parity_selection(list(reversed(rows)))
    assert len(selected) == 18
    assert all(value.endswith("-a") for value in selected)
    assert set(selected) == set(parity_selection(rows))
    with pytest.raises(ValueError, match="Both truths"):
        parity_selection([row for row in rows if row["label"] == 1])


def test_public_parity_plan_binds_order_without_identifying_records():
    identifiers = ["private-case-a", "private-case-b"]
    body = parity_plan(identifiers)
    assert body["parity_inputs"] == 2
    assert set(body) == {"parity_inputs", "parity_ids_sha256"}
    assert body != parity_plan(list(reversed(identifiers)))
    assert all(identifier not in str(body) for identifier in identifiers)
