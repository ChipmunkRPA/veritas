"""A local editing clue has a specific inference, never a borrowed AI confidence."""
import copy

import pytest

from app.engine.finding_assessment import finding_assessment


def example(regions=1, eligible=True, status="ok"):
    finding = {"id": "F6", "family": "F4_localised_manipulation", "evidence_ids": ["check:global"],
               "direction": "against", "strength": "strong"}
    result = {"kind": "image", "status": "done", "signals": [
        {"id": "check:global", "family": "localiser", "category": "forensic", "status": status,
         "metrics": {"regions": regions}, "calibrated_probability": .999,
         "details": {"localisation": {"schema": "recorded-mask/1", "candidate_eligibility": {"eligible": eligible}}}}]}
    return finding, result


def test_positive_regions_are_editing_cues_with_linked_source_context_not_provenance():
    finding, result = example()
    before = copy.deepcopy((finding, result))
    got = finding_assessment(finding, result)
    assert got["version"] == "finding-interpretation/v2"
    assert got["label"] == "Possible local editing" and got["direction"] == "context"
    assert got["included_in_overall"] is False and got["confidence_pct"] is None
    assert "linked original crops, measured masks and excluded areas" in got["interpretation"]
    assert "whether AI caused them is inconclusive" in got["interpretation"]
    assert "provenance, timing" not in got["interpretation"]
    assert (finding, result) == before


@pytest.mark.parametrize("status", ["failed", "skipped", "timeout", "not_applicable"])
def test_failure_is_unavailable_not_a_negative_edit_check(status):
    finding, result = example(status=status)
    got = finding_assessment(finding, result)
    assert got["direction"] == "inconclusive" and got["confidence_pct"] is None
    assert "incomplete or unavailable" in got["interpretation"]


@pytest.mark.parametrize("regions,eligible,phrase", [
    (0, True, "No region met"), (2, False, "withheld"), (None, True, "does not establish usable region coverage"),
    (-1, True, "does not establish usable region coverage"), (.5, True, "does not establish usable region coverage")])
def test_empty_withheld_and_missing_counts_remain_distinct(regions, eligible, phrase):
    finding, result = example(regions, eligible)
    got = finding_assessment(finding, result)
    assert got["direction"] != "original" and got["confidence_pct"] is None
    assert phrase in got["interpretation"]


def test_ela_and_noise_observations_explain_physical_causes():
    finding, result = example()
    result["signals"][0].update(family="forensic", metrics={"ela_mean": 8.},
        details={"display_mapping_version": "full-extent-area/1"})
    got = finding_assessment(finding, result)
    assert got["label"] == "Compression and noise context"
    assert "Texture, lighting, re-saving" in got["interpretation"] and got["confidence_pct"] is None
