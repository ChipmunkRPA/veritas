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
    # Retain all historical numeric/headline expectations; only the versioned
    # presentation marker advances with the current protocol.
    expected = {**case["expected"], "rule": RULE}
    assert {key: got[key] for key in expected} == expected
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


def test_one_video_timing_window_does_not_assess_a_change():
    case = next(c for c in CASES if c["name"] == "real video excludes checked voice and timing from visual percentage")
    result = copy.deepcopy(case["input"])
    before = customer_assessment(result)
    timing = next(s for s in result["signals"] if s.get("family") == "av-sync")
    timing["metrics"] = {"windows": 1, "longest_measured_run": 1, "steps": 0, "collapses": 0}
    got = customer_assessment(result)
    assert "too few measured speech windows" in got["scope_note"]
    assert "Audio/picture timing was checked" not in got["scope_note"]
    assert all(got[key] == before[key] for key in ("direction", "confidence_pct", "excluded_channels"))
    timing["metrics"]["steps"] = 1
    assert "too few measured speech windows" not in customer_assessment(result)["scope_note"]


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
