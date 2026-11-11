"""The intake preserves Unicode boundaries, source matches and group identity."""

import pytest

from scripts import text_opai_bench_intake_v257 as intake


def test_unicode_offsets_preserve_combining_characters_and_line_breaks():
    tagged = "é\r\n<AI_Start>📃e\u0301</AI_End> original <AI_Start>β</AI_End>"
    text, spans = intake.decode_tags(tagged)
    assert text == "é\r\n📃e\u0301 original β"
    assert spans == [[3, 6], [16, 17]]
    assert [text[start:end] for start, end in spans] == ["📃e\u0301", "β"]


@pytest.mark.parametrize(
    "tagged", ["x</AI_End>", "x<AI_Start>y", "<AI_Start><AI_Start>z", "<AI_Start></AI_End>x"]
)
def test_rejects_unbalanced_nested_or_empty_tags(tagged):
    with pytest.raises(ValueError, match="span marker"):
        intake.decode_tags(tagged)


def test_requires_exact_publisher_offsets_and_untagged_lineage():
    row = {
        "record_id": "test",
        "generator": "fixture",
        "version": "v1",
        "version_index": "1",
        "edit_operation": "polish",
        "text": "début 🙂 fin",
        "text_tagged": "début <AI_Start>🙂</AI_End> fin",
        "ai_spans_char": "[[6, 7]]",
    }
    result = intake.record(row)
    assert result["ai_spans"] == [[6, 7]]
    assert intake.retained_prose_matches("début\nprevailing text fin", result)
    assert not intake.retained_prose_matches("fin missing earlier début", result)
    with pytest.raises(ValueError, match="published character intervals"):
        intake.record({**row, "ai_spans_char": "[[7, 11]]"})


def test_independent_upstream_seed_verification_uses_exact_highlight_scope():
    groups = [{"source_id": "GAO_fixture", "cases": [{"text": "first\n second"}]}]
    witness = {
        "id": "fixture",
        "url": "https://example.invalid",
        "highlight": [
            {"section_title": "Findings", "paragraphs": ["first", "second"]},
            {"section_title": "What GAO Recommends", "paragraphs": ["excluded"]},
        ],
    }
    assert intake.bind_source_records(groups, [witness]) == {"GAO_fixture": "https://example.invalid"}
    with pytest.raises(ValueError, match="exactly one GovReport"):
        intake.bind_source_records(groups, [])
    with pytest.raises(ValueError, match="independent GovReport"):
        intake.bind_source_records(groups, [{**witness, "highlight": witness["highlight"][1:]}])


def test_selection_counts_prior_overlap_in_any_variant_before_sampling(monkeypatch):
    prior_text = "zero one two three four five six seven eight nine ten eleven"
    groups = [
        {"source_id": label, "cases": [{"text": "unrelated seed"}, {"text": value}]}
        for label, value in (("a", prior_text), ("b", "other rewrite"), ("c", "another rewrite"))
    ]
    monkeypatch.setattr(intake, "MAX_GROUPS", 1)
    selected, counts = intake.select(groups, intake.exposure.text.fingerprints(prior_text))
    assert selected == intake.select(groups[::-1], intake.exposure.text.fingerprints(prior_text))[0]
    assert len(selected) == 1 and selected[0]["source_id"] != "a"
    assert counts["known_seed_fragment_matches"] == 0
    assert counts["known_family_fragment_matches"] == 1
    assert counts["novel_groups"] == 2
