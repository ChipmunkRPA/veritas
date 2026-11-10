import math

import pytest

from scripts.text_layout_methods_v101 import Evidence, canonical, compare, measure, project, union


def view(text):
    return {"text": text, "source_positions": [None if c.isspace() else i for i, c in enumerate(text)]}


def test_canonical_retains_unicode_punctuation_and_only_real_nonwhite_positions():
    row = view("  😀a\n\nCafé\tline—end ")
    fixed = canonical(row)
    assert fixed["text"] == "😀a Café line—end"
    assert fixed["source_positions"] == [p for p in row["source_positions"] if p is not None][:2] + [
        None
    ] + list(range(6, 10)) + [None] + list(range(11, 19))
    assert canonical(fixed) == fixed
    # No NFKC/case/punctuation normalization or invisible-character deletion.
    assert canonical(view("A\u200bB  e\u0301"))["text"] == "A\u200bB e\u0301"


def test_different_layouts_with_same_character_lineage_reach_identical_input():
    first = {"text": "a b c", "source_positions": [0, None, 8, None, 10]}
    second = {"text": "a\n\nb\tc", "source_positions": [0, None, None, 8, None, 10]}
    assert canonical(first) == canonical(second)
    assert canonical(view("a b"))["text"] != canonical(view("ab"))["text"]


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "ab", "source_positions": [0]},
        {"text": "ab", "source_positions": [0, None]},
        {"text": "a b", "source_positions": [0, 1, 2]},
        {"text": "ab", "source_positions": [2, 1]},
        {"text": "ab", "source_positions": [1, 1]},
        {"text": "ab", "source_positions": [True, 2]},
        {"text": "ab", "source_positions": [-1, 2]},
        {"text": " \n", "source_positions": [None, None]},
    ],
)
def test_invalid_or_empty_source_mapping_refused(bad):
    with pytest.raises(ValueError):
        canonical(bad)


def test_projection_does_not_fill_gaps_or_impute_missing_coverage():
    row = {"text": "A B C", "source_positions": [0, None, 20, None, 80]}
    actual = project(row, [{"start": 0, "end": 3, "score": 0.8}], 0.8)
    assert actual == Evidence(frozenset({0, 20}), frozenset({0, 20}))
    missing = measure({0: 1, 20: 0, 80: 1}, actual)
    assert missing["unscored_ai_characters"] == 1
    assert missing["missed_ai_characters"] == 1
    assert missing["highlighted_human_characters"] == 1


@pytest.mark.parametrize(
    "proposal",
    [
        {"start": -1, "end": 2, "score": 0.9},
        {"start": 0, "end": 4, "score": 0.9},
        {"start": 0, "end": 2, "score": math.nan},
        {"start": 0, "end": 2, "score": math.inf},
        {"start": 0, "end": 2, "score": True},
        {"start": 0, "end": 2, "score": 1.1},
    ],
)
def test_invalid_proposal_refused(proposal):
    with pytest.raises(ValueError):
        project(view("a b"), [proposal], 0.5)


def test_union_retains_complementary_support_without_combined_score():
    a = Evidence(frozenset({1}), frozenset({1, 4}))
    b = Evidence(frozenset({8}), frozenset({8}))
    combined = union(a, b)
    assert combined.mask == {1, 8} and combined.covered == {1, 4, 8}
    assert not hasattr(combined, "score")
    result = compare(a, b, a)
    assert result["baseline_plus_canonical_union"] == combined
    assert result["canonical_current"] == b


def test_unassessed_cannot_become_observed_empty_or_genuine_verdict():
    empty = Evidence(frozenset(), frozenset({1, 2}))
    assert union(empty, None) is None
    result = compare(empty, empty, None)
    assert result["canonical_union"] is None
    assert measure({1: 0, 2: 1}, None) == {"status": "unassessed", "ai_characters": 1, "human_characters": 1}
    assert measure({1: 0, 2: 1}, empty)["ai_document_with_no_ai_located"]


def test_character_metrics_and_known_interval_recovery_use_actual_labelled_positions():
    labels = {0: 1, 2: 1, 4: 0, 8: 1, 10: 1, 15: 0}
    measured = measure(labels, Evidence(frozenset({0, 4, 8}), frozenset(labels)))
    assert measured["known_ai_intervals"] == measured["recovered_ai_intervals"] == 2
    assert measured["located_ai_characters"] == measured["missed_ai_characters"] == 2
    assert measured["highlighted_human_characters"] == 1
    assert measured["character_iou"] == 0.4
    assert measured["useful_ai_location"]
    with pytest.raises(ValueError):
        measure(labels, Evidence(frozenset({99}), frozenset({99})))


def test_genuine_flags_require_observed_highlights_and_known_truth():
    labels = {0: 0, 1: 0}
    assert measure(labels, Evidence(frozenset({0}), frozenset(labels)))["genuine_document_flagged"]
    assert not measure(labels, Evidence(frozenset(), frozenset(labels)))["genuine_document_flagged"]
    with pytest.raises(ValueError):
        measure({0: -1}, None)
