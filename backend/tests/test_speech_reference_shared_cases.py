"""Authored scope/tendency cases also exercised by the browser implementation."""

import json
from pathlib import Path

import pytest

from app.detectors.audio.reference import COMBINED_SHA256, combined_tendency
from app.engine.customer_assessment import customer_assessment
from app.engine.finding_assessment import finding_assessment

CASES = json.loads((Path(__file__).parent / "fixtures/combined_speech_reference_cases.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["name"])
def test_reference_contract(case):
    p, direction = combined_tendency(case["screen"])
    assert p == case["expected"]["probability_ai"] and direction == case["expected"]["direction"]


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["name"])
def test_same_reference_drives_customer_finding_and_separate_video_speech(case):
    screen = case["screen"]
    finding = {
        "id": "F1",
        "family": "A1_voice",
        "scope": None,
        "strength": "weak",
        "direction": "against" if screen["decision"] == "original" else "for",
        "evidence_ids": ["speech:global"],
    }
    result = {
        "kind": "audio",
        "status": "done",
        "findings": [finding],
        "signals": [
            {
                "id": "speech:global",
                "status": "ok",
                "scope": "global",
                "modality": "audio",
                "family": "voice",
                "category": "model",
                "calibration_ref": COMBINED_SHA256,
                "effective_weight": 1.0,
                "calibrated_probability": screen.get("probability_ai"),
                "details": {"audio_screening": screen},
            }
        ],
        "decision": {
            "fuser_sha256": COMBINED_SHA256,
            "probability_ai_calibrated": screen.get("probability_ai"),
            "code": {"ai": "ai_supported", "original": "no_ai_indicators"}.get(
                screen["decision"], "inconclusive"
            ),
            "fusion_mode": "artefact",
        },
    }
    for shown in (customer_assessment(result), finding_assessment(finding, result)):
        assert shown["direction"] == case["expected"]["direction"]
        assert shown["confidence_pct"] == case["expected"]["confidence_pct"]
    video = {**result, "kind": "video", "decision": None}
    shown = customer_assessment(video)
    assert shown["confidence_pct"] is None
    assert shown["soundtrack"]["direction"] == case["expected"]["direction"]
    assert shown["soundtrack"]["confidence_pct"] == case["expected"]["confidence_pct"]
