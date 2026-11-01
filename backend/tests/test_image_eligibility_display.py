"""Display regressions; source/gate binding validation is tested by packet tests."""
from __future__ import annotations

import copy
import math

import pytest
from test_image_evidence_reports import packet_snapshot

from app.reports import engine, image_evidence
from app.reports.image_eligibility import eligibility_view, inspection_check_view
from app.reports.snapshot import snapshot_hash


def gate(score: float) -> dict:
    return {"schema": "image-candidate-eligibility/1", "status": "evaluated", "scope": "whole_image",
            "comparison": "greater_or_equal", "score": round(score, 6), "score_hex": score.hex(),
            "score_exact_decimal": repr(score), "threshold": .5, "threshold_hex": (.5).hex(),
            "threshold_exact_decimal": "0.5", "eligible": score >= .5,
            "reason": "Private implementation wording is not a display label."}


def check(score=.2) -> dict:
    return {"signal_refs": [{"id": "loc.private:global", "display_id": "S06"}], "exhibit_ref": "map1",
            "candidate_eligibility": gate(score), "raw_components": 12, "eligible_components": 12 if score >= .5 else 0,
            "selected_components": 2 if score >= .5 else 0, "omitted_components": 10 if score >= .5 else 0,
            "map_grid": [512, 320], "excluded_area_fraction": .125}


def test_exact_gate_response_remains_below_threshold_after_canonical_rounding():
    value = gate(math.nextafter(.5, 0))
    assert value["score"] == .5
    display = eligibility_view(value)
    assert display["score"] == "0.49999999999999994" and display["status"] == "withheld"
    assert eligibility_view(gate(.5))["status"] == "eligible"
    assert eligibility_view(None)["status"] == "not_recorded"
    value["score_hex"] = "nan"
    assert eligibility_view(value)["status"] == "unavailable"


def test_coverage_keeps_gate_suppression_separate_from_display_caps():
    withheld, passed = inspection_check_view(check()), inspection_check_view(check(.5))
    assert withheld["counts"] == {"raw": 12, "eligible": 0, "selected": 0, "omitted": 0, "suppressed": 12}
    assert passed["counts"] == {"raw": 12, "eligible": 12, "selected": 2, "omitted": 10, "suppressed": 0}
    assert withheld["label"] == "Check S06" and withheld["excluded_percent"] == 12.5
    assert inspection_check_view({**check(), "selected_components": 1})["counts"] is None
    assert inspection_check_view({**check(), "candidate_eligibility": None})["eligible_count_label"] == "Available for region selection"


@pytest.mark.parametrize("profile", ["standard", "forensic", "summary", "newsroom"])
def test_no_packet_report_still_displays_gate_and_incomplete_coverage(tmp_path, monkeypatch, profile):
    snapshot, store = packet_snapshot(tmp_path, profile)
    frozen = copy.deepcopy(snapshot.image_evidence)
    frozen.update(version="image-evidence-3", packets=[], coverage={"inspection_checks": [check(math.nextafter(.5, 0))]})
    snapshot = snapshot.model_copy(update={"image_evidence": frozen})
    # Isolate presentation from the separately owned v3 validator and source bindings.
    monkeypatch.setattr(image_evidence, "verified_set", lambda value: value.image_evidence)
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    before = snapshot_hash(snapshot)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert "How inspection regions were selected" in html
    assert "Regions withheld by the image gate" in html
    assert "0.49999999999999994" in html
    assert "12 measured region(s) were withheld" in html
    assert "12.5% of the measured grid was excluded" in html
    assert "not a region or pixel probability" in html
    assert "absence here is not a negative finding" in html
    assert "No candidate regions are shown in this report" in html
    assert "Private implementation" not in html and "loc.private" not in html
    if profile in ("summary", "newsroom"):
        assert "image-evidence evidence-appendix" in html
    assert snapshot_hash(snapshot) == before
