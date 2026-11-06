import copy
import math

import pytest

from scripts import review_mouth_artifact as review


def fixture():
    sources, cases = [], []
    for n, condition in enumerate(sorted(review.CONTROLS) + ["ai-redraw", "ai-retime"]):
        label = int(condition.startswith("ai-"))
        source = {
            "id": str(n),
            "group": "actor:" + str(n),
            "condition": condition,
            "label": label,
            "sha256": str(n) * 64,
            "time_base": "1/1000",
            "pts": [i * 40 + 500 for i in range(35)],
            "edit_time_range_s": [0.82, 1.58] if label else None,
        }
        indices = list(range(1, 34, 4))
        inputs = [
            {
                "frame_index": i,
                "pts": source["pts"][i],
                "time_base": source["time_base"],
                "time_s": source["pts"][i] / 1000,
                "fake_minus_real_logit": 2.0 if label and 8 <= i <= 26 else 0.0,
            }
            for i in range(35)
        ]
        window = {
            "track_id": "track:0",
            "centers": indices,
            "start_s": 0.86,
            "end_s": 1.5,
            "before_logit": 0.0,
            "middle_logit": 2.0 * label,
            "after_logit": 0.0,
            "response": 2.0 * label,
        }
        cases.append(
            {
                **{k: source[k] for k in ("id", "group", "condition", "label", "sha256")},
                "status": "ok",
                "model_inputs": inputs,
                "artifact_windows": [window],
                "base_windows": [],
            }
        )
        sources.append(source)
    return {
        "schema": "chipmunk.public-mouth-edit-development/4",
        "cases": sources,
        "comparison_lock": {"baseline_review_point": 0.1},
    }, {
        "schema": "chipmunk.mouth-artifact-public-review/5",
        "cases": cases,
        "candidate_review_point": math.nextafter(0.0, math.inf),
    }


def test_standard_library_review_recomputes_temporal_results():
    catalogue, receipt = fixture()
    result = review.summarize(catalogue, receipt)
    assert result["candidate"]["localized_ai"] == 2
    assert result["candidate"]["non_ai_flags"] == 0
    assert result["baseline"]["localized_ai"] == 0
    assert "Completeness" in result["does_not_verify"][1]


@pytest.mark.parametrize("corruption", ["source", "clock", "logit", "response", "point", "missing"])
def test_changed_source_arithmetic_or_denominator_is_refused(corruption):
    catalogue, original = fixture()
    receipt = copy.deepcopy(original)
    row = receipt["cases"][-1]
    if corruption == "source":
        row["sha256"] = "wrong"
    elif corruption == "clock":
        row["model_inputs"][10]["pts"] += 1
    elif corruption == "logit":
        row["model_inputs"][10]["fake_minus_real_logit"] += 10
    elif corruption == "response":
        row["artifact_windows"][0]["response"] += 0.1
    elif corruption == "point":
        receipt["candidate_review_point"] = 2
    else:
        receipt["cases"].pop()
    with pytest.raises(ValueError):
        review.summarize(catalogue, receipt)


def test_failed_positive_remains_in_total_and_missed_denominators():
    catalogue, receipt = fixture()
    row = receipt["cases"][-1]
    row.update(status="failed", reason="Decode unavailable", model_inputs=[], artifact_windows=[])
    result = review.summarize(catalogue, receipt)["candidate"]
    assert result["planned"] == 6 and result["ai_edits"] == 2
    assert result["assessed"] == 5 and result["localized_ai"] == result["ai_without_flag"] == 1
