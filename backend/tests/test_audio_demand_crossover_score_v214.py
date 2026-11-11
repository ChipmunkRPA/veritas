from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audio_demand_crossover_score_v214 as study


def example_rows():
    rows = []
    for index in range(24):
        for origin in study.ORIGINS:
            for condition in study.CONDITIONS:
                missing = index < 5
                decision = ("unavailable" if missing else "original" if origin == "genuine" else
                            "ai" if condition == "matched_clean" else "inconclusive" if condition == "river_early_0db" else "original")
                rows.append({"pair": f"pair-{index:02d}", "origin": origin, "condition": condition,
                             "decision": decision, "route_status": "preparation_unavailable" if missing else "ok"})
    return rows


def test_import_is_fresh_model_and_settings_free():
    env = {**os.environ, "PYTHONPATH": str(Path(study.ROOT) / "backend"), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-B", "-c", "from scripts import audio_demand_crossover_score_v214 as s; "
                           "s.serving.prior.pilot.require_model_free()"], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_results_preserve_planned_denominators_and_wrong_original():
    rows = example_rows()
    got = study.summarize(rows)
    clean = got["outcomes"]["synthetic"]["matched_clean"]
    early = got["outcomes"]["synthetic"]["river_early_0db"]
    late = got["outcomes"]["synthetic"]["river_late_0db"]
    assert clean["planned"] == 24 and clean["directions"]["ai"] == 19 and clean["preparation_unavailable"] == 5
    assert early["missed_ai_over_planned"] == 24 and early["wrong_original"] == 0
    assert late["missed_ai_over_planned"] == 24 and late["wrong_original"] == 19
    assert got["outcomes"]["genuine"]["river_early_0db"]["false_ai"] == 0
    changes = got["within_foreground_transitions"]["synthetic"]
    assert changes["matched_clean_to_river_early_0db"]["directions"] == {"ai->inconclusive": 19, "unavailable->unavailable": 5}
    assert changes["river_early_0db_to_river_late_0db"]["directions"] == {"inconclusive->original": 19, "unavailable->unavailable": 5}
    joint = got["cross_class_shared_excerpt"]["river_late_0db"]
    assert joint["planned_pairs"] == 24 and joint["both_scored"] == 19
    assert joint["genuine_to_synthetic_directions"] == {"original->original": 19, "unavailable->unavailable": 5}
    public = json.dumps(got)
    assert "pair-" not in public and "pella" not in public and "offset" not in public


def test_speech_gate_and_preparation_unavailable_are_distinct():
    rows = example_rows()
    target = next(r for r in rows if r["pair"] == "pair-05" and r["origin"] == "genuine" and r["condition"] == "river_early_0db")
    target.update(route_status="not_applicable", decision="unavailable")
    item = study.summarize(rows)["outcomes"]["genuine"]["river_early_0db"]
    assert item["speech_gate_rejected"] == 1 and item["preparation_unavailable"] == 5
    assert item["directions"]["unavailable"] == 6


def test_summary_rejects_missing_duplicate_or_inconsistent_cases():
    rows = example_rows()
    with pytest.raises(ValueError, match="Every planned"):
        study.summarize(rows[:-1])
    with pytest.raises(ValueError, match="unique planned"):
        study.summarize([*rows[:-1], rows[0]])
    rows[30]["route_status"] = "not_applicable"
    with pytest.raises(ValueError, match="availability must agree"):
        study.summarize(rows)


def test_scoring_audit_requires_exact_window_and_forward_counts():
    case = {"case": "pair-00-genuine-matched_clean", "pair": "pair-00", "origin": "genuine",
            "condition": "matched_clean", "status": "prepared", "samples": 40000}
    row = {key: case[key] for key in ("case", "pair", "origin", "condition")}
    row.update(route_status="ok", decision="original", windows=[[0, 40000]], stability_probe="computed", pella_forwards=2)
    study.audit_case(row, case)
    with pytest.raises(ValueError, match="sample extent and call count"):
        study.audit_case({**row, "windows": [[0, 39999]]}, case)
    with pytest.raises(ValueError, match="sample extent and call count"):
        study.audit_case({**row, "pella_forwards": 1}, case)


def test_batches_are_admitted_only_after_previous_completed(monkeypatch):
    observed = []
    monkeypatch.setattr(study, "verify", lambda: "verified cases")
    monkeypatch.setattr(study, "completed_batch", lambda batch, cases: observed.append((batch, cases)))
    monkeypatch.setattr(study, "scoped", lambda *args: lambda mode, batch: observed.append((mode, batch)))
    study.run(2)
    assert observed == [(0, "verified cases"), (1, "verified cases"), ("extract", 2)]
    with pytest.raises(ValueError, match="fixed batch"):
        study.run(3)
