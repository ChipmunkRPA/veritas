"""Verify outcome accounting and entry points without loading voice models."""

import subprocess
import sys

import pytest

from scripts import audio_voxforge_tr_transfer_v201 as study


def test_fresh_scoring_import_does_not_load_app_settings_or_model():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_voxforge_tr_transfer_v201 as s; s.pilot.require_model_free()",
        ],
        check=True,
    )


def test_crop_validation_retains_both_original_and_zero_origin_clocks():
    lock = {"crop_frame_samples": 320, "crop_frame_counts_in_preference_order": [175, 170]}

    def extent(a, b, pts=0):
        return {
            "start_sample": a,
            "end_sample": b,
            "source_start_pts": a + pts,
            "source_end_pts": b + pts,
            "time_base": "1/16000",
        }

    row = {
        "samples": 56000,
        "full_samples": 72000,
        "full_source_range": extent(0, 72000, -10),
        "source_range": extent(3200, 59200, -10),
        "wav_range": extent(0, 56000),
        "selection": "pause_bounded_crop",
    }
    study.verify_clock(row, lock)
    with pytest.raises(ValueError, match="source and prepared"):
        study.verify_clock(row | {"source_range": extent(3200, 59200)}, lock)
    with pytest.raises(ValueError, match="Complete natural"):
        study.verify_clock(row | {"selection": "natural_short_full"}, lock)


def test_summary_separates_preparation_unavailable_from_gate_rejections():
    classes = ("pause_bounded_crop", "pause_bounded_crop", "natural_short_full", study.UNAVAILABLE_CLASS)
    outcomes = (
        ("ok", "ai", "not_required_for_decision", 1),
        ("not_applicable", "unavailable", "not_run", 0),
        ("ok", "inconclusive", "computed", 2),
        ("preparation_unavailable", "unavailable", "not_run", 0),
    )
    prepared, rows = [], []
    for i, (selection, outcome) in enumerate(zip(classes, outcomes, strict=True)):
        status, decision, probe, forwards = outcome
        slot = f"genuine-{i:02d}"
        source_range = None if selection == study.UNAVAILABLE_CLASS else {"start_sample": 3200 * i}
        case = {"slot": slot, "status": "unavailable" if source_range is None else "qualified"}
        if source_range is not None:
            case.update(selection=selection, samples=56000, source_range=source_range)
        prepared.append(case)
        rows.append(
            {
                "slot": slot,
                "selection_class": selection,
                "source_range": source_range,
                "route_status": status,
                "windows": [[0, 56000]] if status == "ok" else [],
                "decision": decision,
                "stability_probe": probe,
                "pella_forwards": forwards,
            }
        )
    prior = {"planned_slots": 12, "qualified_slots": 3}
    alania = {
        "all_positive_cases": {
            "planned_positives": 24,
            "scored": 24,
            "directions": {"ai": 21, "original": 2, "inconclusive": 1, "unavailable": 0},
            "missed_ai": 3,
        },
        "by_render_mode": {},
    }
    result = study.summary(rows, prepared, prior, alania)
    assert result["genuine_voxforge"]["planned_slots"] == 4
    assert result["genuine_voxforge"]["qualified_slots"] == 3
    assert result["genuine_voxforge"]["scored_slots"] == 2
    assert result["genuine_voxforge"]["directions_over_planned"] == {
        "ai": 1,
        "original": 0,
        "inconclusive": 1,
        "unavailable": 2,
    }
    classes = result["genuine_voxforge_by_selection"]
    assert classes["pause_bounded_crop"]["speech_gate_rejections"] == 1
    assert classes["pause_bounded_crop"]["wrong_ai"] == 1
    assert classes["natural_short_full"]["directions_over_qualified"]["inconclusive"] == 1
    assert classes[study.UNAVAILABLE_CLASS]["unqualified_source_slots"] == 1
    assert classes[study.UNAVAILABLE_CLASS]["speech_gate_rejections"] == 0
    assert result["genuine_voxforge_v191_prior_result"] == prior
    assert result["synthetic_alania_v179_prior_result"]["wrong_original"] == 2
    with pytest.raises(ValueError, match="selection or original"):
        study.summary(
            [rows[0] | {"selection_class": "natural_short_full"}, *rows[1:]], prepared, prior, alania
        )


def test_worker_launch_delegates_to_the_guard_with_the_new_paths(monkeypatch):
    observed = {}

    def capture(function, overrides):
        observed.update(overrides)
        assert function is study.pilot.run
        return lambda: None

    monkeypatch.setattr(study, "scoped", capture)
    study.run()
    assert observed == {
        "TRIAL": study.TRIAL,
        "OUTPUT": study.OUTPUT,
        "MODULE": study.MODULE,
        "verify": study.verify,
        "__file__": study.__file__,
    }
