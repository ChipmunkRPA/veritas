"""Confidence belongs to a calibrated check and scope, never to an arbitrary observation."""
import copy
import json
from pathlib import Path

import pytest

from app.engine.finding_assessment import finding_assessment


def calibrated_example():
    finding = {"id": "F1", "family": "F3_whole_image_generation", "direction": "for", "strength": "moderate",
               "scope": None, "evidence_ids": ["check:global"]}
    result = {"kind": "image", "status": "done", "condition_profile": {"within_validated_envelope": True},
              "signals": [{"id": "check:global", "display_id": "S01", "detector_id": "check.example", "detector_version": "1.0",
                           "category": "model", "status": "ok", "scope": "global", "calibrated_probability": .86,
                           "raw_score": .99, "calibration_ref": "check.example@1.0/eval-1"}],
              "calibration": [{"subject": "check.example@1.0", "method": "platt", "bench_version": "eval-1",
                               "artefact_sha256": "a" * 64, "bench_manifest_sha256": "b" * 64}],
              "decision": {"contributions": [{"kind": "detector", "signal_id": "check:global", "counted": True, "logit": 1.0}]}}
    return finding, result


def test_matching_calibration_is_scoped_and_does_not_mutate_evidence():
    finding, result = calibrated_example()
    before = copy.deepcopy((finding, result))
    got = finding_assessment(finding, result)
    assert got["direction"] == "ai" and got["confidence_pct"] == 86
    assert got["included_in_overall"] is True
    assert got["scope_label"] == "Whole-image generation indicators"
    assert (finding, result) == before


@pytest.mark.parametrize("problem", ["region", "window", "no_reference", "wrong_reference", "unknown_manifest",
                                      "legacy_method", "no_calibration", "unsupported", "zero_reliability"])
def test_no_confidence_borrowed_from_other_scopes_or_unverified_records(problem):
    finding, result = calibrated_example()
    if problem == "region":
        finding["scope"] = {"kind": "bbox", "bbox": {"x": 0, "y": 0, "w": 2, "h": 2}}
    if problem == "window":
        result["signals"][0]["scope"] = "window"
    if problem == "no_reference":
        result["signals"][0]["calibration_ref"] = None
    if problem == "wrong_reference":
        result["signals"][0]["calibration_ref"] = "other/eval-1"
    if problem == "unknown_manifest":
        result["calibration"][0]["bench_manifest_sha256"] = None
    if problem == "legacy_method":
        result["calibration"][0]["method"] = "legacy-heuristic"
    if problem == "no_calibration":
        result["calibration"] = []
    if problem == "unsupported":
        result["condition_profile"]["within_validated_envelope"] = False
    if problem == "zero_reliability":
        result["signals"][0]["reliability"] = {"factor": 0}
    finding["probability"] = .999
    assert finding_assessment(finding, result)["confidence_pct"] is None


@pytest.mark.parametrize("status", ["failed", "skipped", "timeout", "not_applicable"])
def test_unavailable_check_never_becomes_an_original_finding(status):
    finding, result = calibrated_example()
    finding.update(direction="against", probability=.99)
    result["signals"][0].update(status=status, calibrated_probability=.01)
    got = finding_assessment(finding, result)
    assert got["direction"] == "inconclusive" and got["confidence_pct"] is None


def test_several_responses_are_not_averaged_and_metadata_is_not_an_ai_probability():
    finding, result = calibrated_example()
    second = {**result["signals"][0], "id": "second", "calibrated_probability": .98}
    result["signals"].append(second)
    finding["evidence_ids"].append("second")
    assert finding_assessment(finding, result)["confidence_pct"] is None
    finding["family"] = "F8_metadata_consistency"
    got = finding_assessment(finding, result)
    assert got["direction"] == "context" and got["confidence_pct"] is None
    assert got["included_in_overall"] is False


def test_conflicting_directions_do_not_receive_a_confidence():
    finding, result = calibrated_example()
    result["signals"][0]["legacy"] = {"direction": "supports_ai"}
    result["signals"].append({**result["signals"][0], "id": "second", "legacy": {"direction": "supports_authentic"}})
    finding["evidence_ids"].append("second")
    got = finding_assessment(finding, result)
    assert got["direction"] == "inconclusive" and got["confidence_pct"] is None


def test_a_check_can_have_its_own_estimate_while_excluded_from_the_overall():
    finding, result = calibrated_example()
    result["decision"]["contributions"][0].update(counted=False, logit=0)
    got = finding_assessment(finding, result)
    assert got["confidence_pct"] == 86 and got["included_in_overall"] is False


def test_learned_face_checks_are_not_mistaken_for_metadata_context():
    finding, result = calibrated_example()
    finding["family"] = "V1_face_track"
    result["signals"][0].update(category="face", family="learned-face")
    got = finding_assessment(finding, result)
    assert got["direction"] == "ai" and got["confidence_pct"] == 86
    assert got["scope_label"] == "Sampled face content"


CASES = json.loads((Path(__file__).parent/"fixtures/finding_assessment_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_shared_browser_report_and_json_interpretations(case):
    assert finding_assessment(case["finding"], case["result"]) == case["expected"]
