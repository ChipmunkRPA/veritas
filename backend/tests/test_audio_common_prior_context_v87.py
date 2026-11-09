import copy

import pytest

from scripts import audio_common_prior_context_v87 as common
from scripts import audio_score_fusion_v83 as base


def examples():
    rows = []
    for name, counts in (("low", (9, 3)), ("high", (3, 9))):
        for label, count in enumerate(counts):
            for number in range(count):
                key = f"cal-{name}-{label}-{number}"
                rows.append({"role": "confidence_cal", "status": "ok", "raw": 0.5,
                             "candidates": {"complement": {"p": 0.5}}, "quality_flatness": 0.1 if name == "low" else 0.9,
                             "label": label, "kind": "genuine" if label == 0 else "stock", "base_id": key, "group": key})
    return rows


def context():
    return {"method": copy.deepcopy(base.METHOD), "arm": "flatness_context", "routing_cutoff": 0.5,
            "fit_group_ids": ["fit-only"], "fit_source_ids": ["fit-only"],
            "branches": {name: {"status": "fitted", "coefficients": [0.0, 0.0, 0.0]} for name in ("low", "high")}}


def test_common_reference_keeps_conditional_class_prior_with_uninformative_scores():
    original = context()
    snapshot = copy.deepcopy(original)
    artifact = common.calibrate(original, examples())
    assert original == snapshot
    assert artifact["branches"]["low"]["conditional_ai_reference_prior"] == pytest.approx(0.25)
    assert artifact["branches"]["high"]["conditional_ai_reference_prior"] == pytest.approx(0.75)
    for row in examples():
        result = common.predict(row, artifact)
        assert result["probability_ai"] == pytest.approx(0.25 if row["quality_flatness"] < 0.5 else 0.75, abs=1e-6)
        minimal = {key: row[key] for key in ("status", "raw", "candidates", "quality_flatness")}
        assert common.predict(minimal, artifact) == result


def test_original_balanced_branch_map_is_distinct_and_role_group_leakage_is_rejected():
    balanced = base.calibrate(context(), examples())
    assert base.predict(examples()[0], balanced)["probability_ai"] == pytest.approx(0.5)
    wrong = examples()
    wrong[0]["role"] = "selection"
    with pytest.raises(ValueError, match="Only assigned calibration"):
        common.calibrate(context(), wrong)
    reused = examples()
    reused[0]["group"] = "fit-only"
    with pytest.raises(ValueError, match="cannot reuse"):
        common.calibrate(context(), reused)


def test_missing_check_and_unsupported_context_remain_unassessed():
    artifact = common.calibrate(context(), [row for row in examples() if row["quality_flatness"] < 0.5])
    assert common.predict({"status": "not_applicable"}, artifact)["probability_ai"] is None
    high = next(row for row in examples() if row["quality_flatness"] > 0.5)
    assert common.predict(high, artifact)["decision"] == "unassessed"
