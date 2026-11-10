from copy import deepcopy

import pytest

from scripts.review_text_layout_v102 import geometry, joined, metrics, observed


def fixtures():
    first = {
        "text": "A\n\n😀 B",
        "source_positions": [0, None, None, 3, None, 5],
        "labels": [1, -1, -1, 0, -1, 1],
    }
    other = {"text": "A 😀 B", "source_positions": [0, None, 3, None, 5], "labels": [1, -1, 0, -1, 1]}
    return first, other


def test_separate_mapping_reconstructs_source_lineage():
    first, other = fixtures()
    assert geometry(first, other) == {0: 1, 3: 0, 5: 1}


@pytest.mark.parametrize("problem", ["content", "labels", "position", "whitespace", "boundary"])
def test_geometry_refuses_changed_content_or_lineage(problem):
    first, other = fixtures()
    other = deepcopy(other)
    if problem == "content":
        other["text"] = "A X B"
    elif problem == "labels":
        other["labels"][2] = 1
    elif problem == "position":
        other["source_positions"][2] = 4
    elif problem == "whitespace":
        other["labels"][1] = 0
    else:
        other = {"text": "A😀 B", "source_positions": [0, 3, None, 5], "labels": [1, 0, -1, 1]}
    with pytest.raises(ValueError):
        geometry(first, other)


def test_observation_separates_positive_support_and_covered_support():
    _, view = fixtures()
    response = {"status": "ok", "arms": {"current": [{"start": 0, "end": 3, "score": 0.5}]}}
    assert observed(view, response, "current", 0.5) == ({0, 3}, {0, 3})
    assert observed(view, response, "current", 0.6) == (set(), {0, 3})
    assert observed(view, {"status": "unassessed"}, "current", 0.5) is None
    response["arms"]["current"][0]["end"] = 99
    with pytest.raises(ValueError):
        observed(view, response, "current", 0.5)


def test_union_has_no_negative_imputation_or_score():
    assert joined(({0}, {0, 1}), ({3}, {3})) == ({0, 3}, {0, 1, 3})
    assert joined((set(), {0}), None) is None


def test_independent_metrics_recover_two_disjoint_ai_intervals():
    actual = metrics({0: 1, 3: 0, 5: 1}, ({0, 3}, {0, 3}))
    assert actual["located_ai_characters"] == actual["highlighted_human_characters"] == 1
    assert actual["missed_ai_characters"] == actual["unscored_ai_characters"] == 1
    assert actual["known_ai_intervals"] == 2 and actual["recovered_ai_intervals"] == 1
    assert actual["character_iou"] == 1 / 3 and actual["useful_ai_location"]


def test_unassessed_empty_and_unknown_truth_are_distinct():
    assert metrics({0: 0}, None)["status"] == "unassessed"
    assert metrics({0: 0}, (set(), {0}))["genuine_document_flagged"] is False
    assert metrics({0: 0}, ({0}, {0}))["genuine_document_flagged"] is True
    with pytest.raises(ValueError):
        metrics({0: -1}, None)
