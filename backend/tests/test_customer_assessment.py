import copy
import json
from pathlib import Path

import pytest

from app.engine.customer_assessment import customer_assessment

CASES = json.loads((Path(__file__).parent / "fixtures/customer_assessment_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_customer_assessment(case):
    before = copy.deepcopy(case["input"])
    got = customer_assessment(case["input"])
    assert {key: got[key] for key in case["expected"]} == case["expected"]
    assert case["input"] == before
    assert got["rule"] == "80-percent-leaning/v1"
    if got["confidence_pct"] is not None:
        assert "not an independently established accuracy" in got["limitation"]
        assert got["confidence_label"] == "Automated estimate"
    assert "90% accurate" in got["interval_note"]
    if got["scope"] == "visual_content":
        assert "checked separately" in got["scope_note"]
        assert "audio has not been cleared as original" in got["scope_note"]


@pytest.mark.parametrize("p", [float("inf"), float("nan"), True, "0.9"])
def test_rejects_non_probability(p):
    result = {"decision": {"code": "ai_supported", "fusion_mode": "artefact", "probability_ai_calibrated": p},
              "condition_profile": {"within_validated_envelope": True}}
    assert customer_assessment(result)["confidence_pct"] is None
