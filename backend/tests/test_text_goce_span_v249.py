import pytest

from scripts import text_goce_span_v249 as study

POINTS = {"current": {"cut": 0.5}, "interval": {"cut": 0.5}}


def witness(raw, term, origin=100):
    a = raw.index(term)
    left, right = raw.index("article"), raw.index("sentence")
    return {
        "excerpt": raw,
        "article_codepoints": [origin, origin + len(raw)],
        "target_codepoints": [origin + left, origin + right],
        "retouch_codepoints": [origin + a, origin + a + len(term)],
        "target_words": len(raw[left:right].split()),
    }


def pair():
    a = "The known café article used old words in one clear and human sentence."
    b = "The known café article used new ordinary terms in one clear and human sentence."
    return [
        study.window_row(witness(a, "old words"), 1, "clean_human"),
        study.window_row(witness(b, "new ordinary terms", 113), 1, "human_retouch"),
    ]


def proposal(a, b, score):
    return {"start": a, "end": b, "score": score}


def observed(row, *, flag=False, partial=False, missing=False):
    a, b = row["locus"]
    all_text = proposal(0, len(row["text"]), 0.1)
    proposals = [proposal(a, b, 0.8 if flag else 0.1)] if partial else [all_text]
    if flag and not partial:
        proposals.append(proposal(a, b, 0.8))
    response = (
        {"status": "unassessed"}
        if missing
        else {
            "status": "measured",
            "primary_current": proposals,
            "canonical_current": proposals,
            "canonical_interval": [all_text],
        }
    )
    return {
        "case": row["case"],
        **response,
        "positions": study.private_positions(row, study.position_masks(row, response, POINTS)),
    }


def test_exact_relative_offsets_and_unchanged_context_survive_a_retouch():
    a, b = pair()
    shared = study.shared_context(a, b)
    assert len(shared) > 40
    assert all(a["text"][i] == b["text"][j] for i, j in shared)
    assert any(i != j for i, j in shared)
    assert all(not a["locus"][0] <= i < a["locus"][1] for i, _ in shared)
    assert all(not b["locus"][0] <= j < b["locus"][1] for _, j in shared)
    bad = witness(a["text"], "old words")
    bad["retouch_codepoints"][0] -= 100
    with pytest.raises(ValueError, match="lexical replacement coordinates"):
        study.window_row(bad, 1, "clean_human")


def test_false_highlight_is_counted_in_exact_locus_without_flagging_unchanged_context():
    a, b = pair()
    rows = [a, b]
    observed_rows = [observed(a), observed(b, flag=True)]
    result = study.summarise(rows, observed_rows, POINTS)
    baseline = result["by_arm_and_method"]["baseline"]
    assert baseline["clean_human"]["false_flagged_windows"] == 0
    assert baseline["human_retouch"]["false_flagged_windows"] == 1
    assert baseline["human_retouch"]["replacement_locus_false_flagged_characters"] == len("newordinaryterms")
    assert baseline["human_retouch"]["target_padding_false_flagged_characters"] == 0
    assert baseline["human_retouch"]["outside_target_false_flagged_characters"] == 0
    contrast = result["paired_false_flags_and_shared_context"]["baseline"]
    assert contrast["fully_assessed_pairs"] == contrast["retouch_only_false_flagged"] == 1
    assert contrast["unchanged_characters_with_new_false_flag"] == 0
    assert "known café" not in repr(result)


@pytest.mark.parametrize(
    ("kind", "bucket"), [("missing", "unassessed_pairs"), ("partial", "partially_assessed_pairs")]
)
def test_uncovered_pairs_do_not_count_as_neither_flagged(kind, bucket):
    a, b = pair()
    result = study.summarise([a, b], [observed(a), observed(b, **{kind: True})], POINTS)
    comparison = result["paired_false_flags_and_shared_context"]["baseline"]
    assert comparison[bucket] == 1
    assert comparison["neither_false_flagged"] == 0
    assert comparison["known_unchanged_characters"] > 0


def test_summary_rejects_changed_position_receipt():
    a, b = pair()
    before, after = observed(a), observed(b, flag=True)
    after["positions"]["methods"]["baseline"]["flagged"] = []
    with pytest.raises(ValueError, match="source-position replay differs"):
        study.summarise([a, b], [before, after], POINTS)
