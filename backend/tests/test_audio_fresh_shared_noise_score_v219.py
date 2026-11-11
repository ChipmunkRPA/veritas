from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts import audio_fresh_shared_noise_score_v219 as study


def example_rows():
    rows = []
    for i in range(16):
        for arm in study.ARMS:
            for condition in study.CONDITIONS:
                missing = arm == "genuine" and i < 5
                decision = (
                    "unavailable"
                    if missing
                    else "original"
                    if arm == "genuine"
                    else "ai"
                    if condition == "matched_clean"
                    else "inconclusive"
                    if arm == "alania"
                    else "original"
                )
                rows.append(
                    {
                        "triplet": f"triplet-{i:02d}",
                        "arm": arm,
                        "condition": condition,
                        "decision": decision,
                        "route_status": "preparation_unavailable" if missing else "ok",
                    }
                )
    return rows


def test_import_preserves_strict_model_free_serving_identity():
    env = {**os.environ, "PYTHONPATH": str(study.ROOT / "backend"), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_fresh_shared_noise_score_v219 as s; "
            "s.serving.prior.pilot.require_model_free()",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_aggregate_distinguishes_missing_gates_missed_flags_and_wrong_original():
    rows = example_rows()
    target = next(
        r
        for r in rows
        if r["triplet"] == "triplet-05" and r["arm"] == "genuine" and r["condition"] == "kitchen_0db"
    )
    target.update(route_status="not_applicable", decision="unavailable")
    result = study.summarize(rows)
    genuine = result["outcomes"]["genuine"]["kitchen_0db"]
    assert (
        genuine["scored"],
        genuine["speech_gate_rejected"],
        genuine["preparation_unavailable"],
        genuine["false_ai"],
    ) == (10, 1, 5, 0)
    alania = result["outcomes"]["alania"]["kitchen_0db"]
    drifting = result["outcomes"]["drifting"]["kitchen_0db"]
    assert (alania["missed_ai_over_planned"], alania["wrong_original"]) == (16, 0)
    assert (drifting["missed_ai_over_planned"], drifting["wrong_original"]) == (16, 16)
    assert result["clean_to_noise_transitions"]["alania"]["matched_clean_to_kitchen_0db"]["directions"] == {
        "ai->inconclusive": 16
    }
    assert result["scored_pair_coverage"]["kitchen_0db"] == {
        "genuine_alania": 10,
        "genuine_drifting": 10,
        "alania_drifting": 16,
        "complete_triplets": 10,
    }
    assert "triplet-" not in json.dumps(result)


def test_summary_rejects_missing_duplicate_and_inconsistent_rows():
    rows = example_rows()
    with pytest.raises(ValueError, match="Every planned"):
        study.summarize(rows[:-1])
    with pytest.raises(ValueError, match="unique planned"):
        study.summarize([*rows[:-1], rows[0]])
    rows[-1]["route_status"] = "not_applicable"
    with pytest.raises(ValueError, match="availability must agree"):
        study.summarize(rows)


def test_audit_requires_exact_windows_and_forward_count():
    case = {
        "case": "triplet-00-genuine-matched_clean",
        "triplet": "triplet-00",
        "arm": "genuine",
        "condition": "matched_clean",
        "status": "prepared",
        "samples": 48000,
    }
    row = {key: case[key] for key in ("case", "triplet", "arm", "condition")}
    row.update(
        route_status="ok",
        decision="original",
        windows=[[0, 48000]],
        stability_probe="computed",
        pella_forwards=2,
    )
    study.audit_case(row, case)
    with pytest.raises(ValueError, match="sample extent and call count"):
        study.audit_case({**row, "windows": [[0, 47999]]}, case)
    with pytest.raises(ValueError, match="sample extent and call count"):
        study.audit_case({**row, "pella_forwards": 1}, case)


def test_batches_require_all_previous_completions(monkeypatch):
    observed = []
    monkeypatch.setattr(study, "verify", lambda: "verified cases")
    monkeypatch.setattr(study, "completed_batch", lambda batch, cases: observed.append((batch, cases)))
    monkeypatch.setattr(study, "scoped", lambda *args: lambda mode, batch: observed.append((mode, batch)))
    study.run(4)
    assert observed == [(i, "verified cases") for i in range(4)] + [("extract", 4)]
    with pytest.raises(ValueError, match="fixed batch"):
        study.run(5)


def test_v217_receipt_and_residual_commitments_are_checked(monkeypatch):
    source = []
    for i in range(16):
        for arm in study.ARMS:
            for condition in study.CONDITIONS:
                source.append(
                    {
                        "case": f"triplet-{i:02d}-{arm}-{condition}",
                        "triplet": f"triplet-{i:02d}",
                        "arm": arm,
                        "condition": condition,
                        "status": "prepared",
                        "samples": 48000,
                        "source_prefix_sha256": arm,
                        "noise_sample_offset": None if condition == "matched_clean" else (4 + 6 * i) * 16000,
                        "noise_residual_sha256": condition,
                        "noise_excerpt_sha256": condition,
                    }
                )
    frozen = {
        "schema": "chipmunk.audio-fresh-shared-noise-freeze/217",
        "planned_triplets": 16,
        "planned_cases": 240,
        "arms": list(study.ARMS),
        "conditions": list(study.CONDITIONS),
        "source_range": [0, 48000],
        "sample_rate": 16000,
        "model_calls_permitted": False,
        "detector_calls_permitted": False,
        "holdout_access_permitted": False,
        "code_sha256": {"backend/scripts/audio_fresh_shared_noise_v217.py": "digest"},
    }
    result = {
        "schema": "chipmunk.audio-fresh-shared-noise-result/217",
        "status": "complete",
        "planned_cases": 240,
        "new_neural_calls": 0,
        "detector_calls": 0,
        "holdout_scored": False,
        "freeze_sha256": "digest",
        "private_cases_sha256": "digest",
        "by_arm_and_condition": {
            arm: {condition: {"prepared": 16} for condition in study.CONDITIONS} for arm in study.ARMS
        },
    }
    objects = {
        study.PREP_FREEZE: frozen,
        study.INPUT_TRIAL / "freeze.json": frozen,
        study.INPUT / "output/result.json": result,
        study.PREP: {**result, "profile_sha256": "digest"},
        study.INPUT / "profile.json": {},
    }
    monkeypatch.setattr(study, "read", objects.__getitem__)
    monkeypatch.setattr(study, "sha", lambda _: "digest")
    monkeypatch.setattr(study.serving, "records", lambda _: source)
    monkeypatch.setattr(study.serving.prior, "verify_data_guard", lambda *args: None)
    monkeypatch.setattr(study.serving.prior.pilot, "require_model_free", lambda: None)
    monkeypatch.setattr(study, "waveform", lambda _: b"verified")
    assert study.cases() == source
    source[6]["noise_residual_sha256"] = "different"
    with pytest.raises(ValueError, match="share exact"):
        study.cases()
