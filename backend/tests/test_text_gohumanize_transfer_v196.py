import pytest

from scripts import text_gohumanize_transfer_v196 as trial


def fixture(arm="short_ai", spans=None):
    text = "Genuine café.\n\nGenerated résumé.\n\nGenuine."
    return {
        "text": text,
        "text_sha256": trial.text_sha(text),
        "arm": arm,
        "ai_spans": [[15, 32]] if spans is None else spans,
        "status": "constructed",
    }


def test_unicode_labels_and_layout_mapping_preserve_exact_source_positions():
    row = fixture()
    primary, canonical = trial.views(row["text"])
    labels = trial.labels_for(row)
    for view in (primary, canonical):
        assert {p for p in view["source_positions"] if p is not None} == labels.keys()
        assert "".join(row["text"][p] for p in view["source_positions"] if p is not None) == "".join(
            c for c in row["text"] if not c.isspace()
        )
    assert primary["text"] != canonical["text"]
    assert labels[row["text"].index("é")] == 0
    assert labels[row["text"].rindex("é")] == 1


@pytest.mark.parametrize("spans", [[[4, 9], [8, 12]], [[0, 999]], [[4, 4]]])
def test_invalid_source_spans_fail_closed(spans):
    with pytest.raises(ValueError, match="Unicode source spans"):
        trial.labels_for(fixture(spans=spans))


def test_missing_interval_leaves_only_the_affected_combinations_unassessed():
    row = fixture()
    ordinary, canonical = trial.views(row["text"])
    response = {
        "status": "measured",
        "primary_current": [{"start": 0, "end": len(ordinary["text"]), "score": 0.6}],
        "canonical_current": [{"start": 0, "end": len(canonical["text"]), "score": 0.2}],
        "canonical_interval": None,
    }
    measured = trial.measurements(row, response, {"current": {"cut": 0.5}, "interval": {"cut": 0.5}})
    assert measured["baseline"]["located_ai_characters"] > 0
    assert measured["canonical_current"]["located_ai_characters"] == 0
    assert measured["canonical_union"]["status"] == "unassessed"
    assert measured["baseline_plus_canonical_union"]["status"] == "unassessed"


def test_unavailable_preparation_requires_an_explicit_matching_disposition():
    row = {"status": "unavailable"}
    assert all(
        item["status"] == "preparation_unavailable"
        for item in trial.measurements(row, {"status": "preparation_unavailable"}, {}).values()
    )
    with pytest.raises(ValueError, match="Unexpected prediction"):
        trial.measurements(row, {"status": "measured"}, {})
