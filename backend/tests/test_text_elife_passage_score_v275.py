import pytest

from scripts import text_elife_passage_score_v275 as scoring

POINTS = {"current": {"cut": 0.6}, "interval": {"cut": 0.6}}


def cohort():
    rows, originals = [], []
    for i in range(5):
        group = f"group-{i}"
        source, a, b = "Prévious words.\n\nKnown original.\n\nNext words.", 17, 32
        digest = scoring.prior.text_sha(source)
        originals.append(
            {"group": group, "text": source, "text_sha256": digest, "edit_target_codepoints": [a, b]}
        )
        rows.append(
            {
                "case": f"{group}-human",
                "article_group": group,
                "generator": None,
                "arm": "human",
                "status": "constructed",
                "text": source,
                "text_sha256": digest,
                "ai_spans": [],
            }
        )
        for family in ("qwen", "smol2"):
            available = family == "qwen" or i == 0
            replacement = "Synthetic café."
            mixed = source[:a] + replacement + source[b:]
            rows.append(
                {
                    "case": f"{group}-{family}",
                    "article_group": group,
                    "generator": family,
                    "arm": "paragraph_ai",
                    "status": "constructed" if available else "unavailable",
                    "reason": None if available else "outside_word_bound",
                    "source_sha256": digest,
                    "source_target": [a, b],
                    "text": mixed if available else None,
                    "text_sha256": scoring.prior.text_sha(mixed) if available else None,
                    "ai_spans": [[a, a + len(replacement)]] if available else None,
                }
            )
    return rows, originals


def test_source_binding_rejects_nonreplacement_drift():
    rows, originals = cohort()
    assert scoring.validate(rows, originals) == rows
    changed = "Z" + rows[1]["text"][1:]
    rows[1].update(text=changed, text_sha256=scoring.prior.text_sha(changed))
    with pytest.raises(ValueError, match="exact Unicode prefix and suffix"):
        scoring.validate(rows, originals)


def test_unavailable_attempts_cannot_receive_prediction_or_original_credit():
    rows, _ = cohort()
    row = rows[5]
    with pytest.raises(ValueError, match="must have no prediction"):
        scoring.assessments(row, {"case": row["case"], "status": "measured"}, POINTS)
    result = scoring.summarize([row], [{"case": row["case"], "status": "preparation_unavailable"}], POINTS)
    for metrics in result["smol2"].values():
        assert metrics["planned"] == metrics["preparation_unavailable"] == 1
        assert metrics["fully_assessed"] == metrics["complete_direction_matches"] == 0


def test_current_and_interval_report_separate_coverage_and_shared_position_rank():
    text = "aaa café zzz"
    row = {
        "case": "g-qwen",
        "generator": "qwen",
        "arm": "paragraph_ai",
        "status": "constructed",
        "text": text,
        "text_sha256": scoring.prior.text_sha(text),
        "ai_spans": [[4, 8]],
    }
    response = {
        "case": row["case"],
        "status": "measured",
        "native_proofs": [{}, None, {}],
        "primary_current": [
            {"start": 0, "end": 4, "score": 0.1, "raw": 0.1},
            {"start": 4, "end": 8, "score": 0.8, "raw": 0.8},
            {"start": 8, "end": 12, "score": 0.1, "raw": 0.1},
        ],
        "canonical_interval": [
            {"start": 0, "end": 4, "score": 0.7, "raw": 0.7},
            {"start": 4, "end": 8, "score": 0.4, "raw": 0.4},
        ],
    }
    result = scoring.summarize([row], [response], POINTS)["qwen"]
    current, interval = result["current"], result["interval"]
    assert (
        current["fully_assessed"] == current["complete_direction_matches"] == current["useful_locations"] == 1
    )
    assert interval["partially_assessed"] == 1 and interval["highlighted_human_characters"] == 3
    assert interval["complete_direction_matches"] == interval["useful_locations"] == 0
    assert current["mean_raw_common_position_rank"] == 1.0
    assert interval["mean_raw_common_position_rank"] == 0.0
