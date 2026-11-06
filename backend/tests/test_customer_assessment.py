import copy
import json
from pathlib import Path

import pytest

from app.engine.customer_assessment import RULE, customer_assessment

CASES = json.loads((Path(__file__).parent / "fixtures/customer_assessment_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_customer_assessment(case):
    before = copy.deepcopy(case["input"])
    got = customer_assessment(case["input"])
    assert {key: got[key] for key in case["expected"]} == case["expected"]
    assert case["input"] == before
    assert got["rule"] == RULE
    if got["confidence_pct"] is not None:
        assert "not an independently established accuracy" in got["limitation"]
        assert got["confidence_label"] == "Reference-calibrated overall estimate"
    assert "90% accurate" in got["interval_note"]
    if got["scope"] == "visual_content":
        assert "uses sampled visual content" in got["scope_note"]
        assert got["excluded_channels"] == ["voice", "audio_picture_timing"]


@pytest.mark.parametrize("p", [float("inf"), float("nan"), True, "0.9"])
def test_rejects_non_probability(p):
    result = {"decision": {"code": "ai_supported", "fusion_mode": "artefact", "probability_ai_calibrated": p},
              "condition_profile": {"within_validated_envelope": True}}
    assert customer_assessment(result)["confidence_pct"] is None


@pytest.mark.parametrize("code,p,direction,confidence", [
    ("ai_supported", .6, "ai", 60), ("ai_supported", .599999, "inconclusive", None),
    ("no_ai_indicators", .4, "original", 60), ("no_ai_indicators", .400001, "inconclusive", None),
    ("inconclusive", .99, "inconclusive", None), ("inconclusive", .01, "inconclusive", None),
])
def test_sixty_percent_boundary_preserves_recorded_uncertainty(code, p, direction, confidence):
    result = {"decision": {"code": code, "fusion_mode": "artefact", "probability_ai_calibrated": p},
              "condition_profile": {"within_validated_envelope": True}}
    got = customer_assessment(result)
    assert got["direction"] == direction and got["confidence_pct"] == confidence
    assert "threshold_pct" not in got
    assert "threshold_pct" not in got["service_policy"]
    assert got["service_policy"]["court_use"] == "not_prepared_for_court_use"
    assert got["service_policy"]["separate_expert_engagement_required"] is True
    assert got["service_policy"]["expert_contact"] == "sales@chipmunkrpa.com"
