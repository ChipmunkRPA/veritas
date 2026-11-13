"""Scope is a presentation rule; no score fusion, new calibration or truth-driven routing."""
import copy
import json
from pathlib import Path

import pytest

from app.engine.customer_assessment import customer_assessment

FIXTURE = json.loads((Path(__file__).parent / "fixtures/image_scope_cases_v50.json").read_text())


def make_input(case):
    value = copy.deepcopy(FIXTURE["base"])
    local = value["signals"][1]
    local["metrics"]["regions"] = case["regions"]
    local["status"] = case["local_status"]
    local["details"]["localisation"]["candidate_eligibility"]["eligible"] = case["eligible"]
    if case.get("ai"):
        value["decision"].update(code="ai_supported", probability_ai_calibrated=.97, ci90=[.92, .99])
    if case.get("unknown_scope"):
        value["signals"][0].update(family="unknown", hypothesis=None)
    if case.get("whole_status"):
        value["signals"][0]["status"] = case["whole_status"]
    if case.get("code"):
        value["decision"]["code"] = case["code"]
    return value


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_scoped_image_presentation(case):
    value = make_input(case)
    before = copy.deepcopy(value)
    result = customer_assessment(value)
    assert {k: result[k] for k in case["expected"]} == case["expected"]
    assert value == before
    assert result["rule"] == "evidence-led-assessment/v7"
    if case.get("component_confidence"):
        assert result["component_estimate"]["confidence_pct"] == case["component_confidence"]
        assert result["component_estimate"]["interval_pct"] == [92., 99.]
        assert "No whole-file confidence" in result["scope_note"]
        assert "97%" in result["scope_note"]
    if result.get("scope") == "image_components":
        assert "pixel" in result["scope_note"] or "highlighted" in result["scope_note"]


def test_local_response_is_not_used_as_probability_or_fused_into_existing_estimate():
    value = make_input(FIXTURE["cases"][0])
    value["signals"][1].update(raw_score=.9999, calibrated_probability=None)
    first = customer_assessment(value)
    value["signals"][1]["raw_score"] = .01
    assert customer_assessment(value) == first
    assert first["confidence_pct"] is None and first["component_estimate"]["confidence_pct"] == 97
