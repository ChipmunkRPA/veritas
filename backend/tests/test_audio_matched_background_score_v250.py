from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from itertools import product

import numpy as np
import pytest

from scripts import audio_matched_background_score_v250 as study


def test_import_preserves_fresh_model_and_settings_boundary():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_matched_background_score_v250 as p; p.NATIVE.PILOT.require_model_free()",
        ],
        env={**os.environ, "PYTHONPATH": str(study.ROOT / "backend"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def fixture_rows():
    cases, baseline, candidate = [], [], []
    for name, triplet, arm, condition in study.expected_cases():
        index = int(triplet[-2:])
        case = {
            "case": name,
            "triplet": triplet,
            "arm": arm,
            "condition": condition,
            "status": "prepared" if index <= 2 else "unavailable",
            "samples": 48_000,
            "group": "stock" if arm == "okur" else f"{arm}-{index}",
            "cohort": "v252"
            if arm == "genuine"
            else "v236"
            if arm == "alania"
            else "v240"
            if index < 4
            else "v245",
        }
        status = "ok" if index < 2 else "not_applicable" if index == 2 else "preparation_unavailable"
        if index == 0:
            recorded = "original" if arm == "okur" else "inconclusive"
            diagnostic = "ai"
        elif index == 1:
            recorded = {"genuine": "original", "alania": "ai", "okur": "inconclusive"}[arm]
            diagnostic = "original" if arm == "alania" else "ai"
        else:
            recorded = diagnostic = "unavailable"
        base = {key: case[key] for key in study.candidate.KEYS}
        baseline.append(
            {
                **base,
                "decision": recorded,
                "route_status": status,
                "windows": [[0, 48_000]] if status == "ok" else [],
                "stability_probe": "not_required_for_decision" if status == "ok" else "not_run",
                "pella_forwards": int(status == "ok"),
            }
        )
        candidate.append(
            {
                **base,
                "recorded_direction": recorded,
                "route_status": status,
                "candidate_english_point_direction": diagnostic,
                "current_raw": 0.2 if arm == "genuine" else 0.6 if status == "ok" else None,
                "candidate_score": 0.3 if arm == "genuine" else 0.8 if status == "ok" else None,
                "encoder_forwards": int(status == "ok"),
            }
        )
        if status != "ok":
            candidate[-1]["current_raw"] = candidate[-1]["candidate_score"] = None
        cases.append(case)
    return cases, baseline, candidate


def test_only_inconclusive_to_ai_transition_is_allowed():
    for recorded, diagnostic in product(study.DIRECTIONS, repeat=2):
        expected = "ai" if (recorded, diagnostic) == ("inconclusive", "ai") else recorded
        assert study.second_opinion(recorded, diagnostic) == expected


def test_candidate_requires_all_baselines_and_inherits_their_gate():
    cases, baseline, _ = fixture_rows()
    batches = [(baseline[i :: len(study.CONDITIONS)], "profile", 0) for i in range(len(study.CONDITIONS))]
    pairs = study.candidate_pairs(cases, batches)
    assert all(len(batch) == 72 for batch in pairs)
    assert all(sum(study.candidate.admitted(*pair) for pair in batch) == 6 for batch in pairs)
    with pytest.raises(ValueError, match="Complete baseline phase"):
        study.candidate_pairs(cases, batches[:-1])
    with pytest.raises(ValueError, match="72-row"):
        study.candidate_pairs(cases, [(batches[0][0][:-1], "profile", 0), *batches[1:]])


def test_coverage_keeps_unavailable_and_counts_recoveries_separately():
    cases, _, rows = fixture_rows()
    result = study.summarize(cases, list(reversed(rows)))
    clean = result["by_condition"]["matched_clean"]
    assert clean["planned"] == 72 and clean["assessed_by_both"] == 6
    assert clean["preparation_unavailable"] == 63 and clean["speech_gate_rejected"] == 3
    assert clean["recovered_synthetic_variants"] == 2 and clean["added_false_ai_variants"] == 1
    assert (
        clean["baseline_synthetic_wrong_original_variants"]
        == clean["second_opinion_synthetic_wrong_original_variants"]
        == 1
    )
    okur = result["by_arm_and_condition"]["okur"]["matched_clean"]
    assert okur["recovered_synthetic_voice_groups"] == 1
    assert result["by_source_cohort_and_condition"]["v245"]["matched_clean"]["planned"] == 20
    rank = result["rank_by_generator_and_condition"]["matched_clean"]["okur"]["candidate_score"]
    assert rank["all"]["comparisons"] == 4 and rank["within_triplet"]["comparisons"] == 2
    with pytest.raises(ValueError, match="exactly one paired outcome"):
        study.summarize(cases, rows[:-1])
    with pytest.raises(ValueError, match="unchanged baseline speech gate"):
        study.summarize(cases, [{**rows[0], "encoder_forwards": 0}, *rows[1:]])


def test_both_routes_bind_identical_native_float32_bytes(tmp_path, monkeypatch):
    reader = study.scoped(study.baseline.serving.waveform, {"INPUT": tmp_path})
    monkeypatch.setattr(study, "verified_wav", reader)
    prepared = tmp_path / "output/prepared"
    prepared.mkdir(parents=True)
    case = {
        "case": "triplet-00-genuine-matched_clean",
        "triplet": "triplet-00",
        "arm": "genuine",
        "condition": "matched_clean",
        "status": "prepared",
        "samples": 48_000,
    }
    pcm = np.tile([-32768, -1, 0, 32767], 12_000).astype("<i2").tobytes()
    wav = prepared / f"{case['case']}.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 16_000, 48_000, "NONE", "not compressed"))
        stream.writeframes(pcm)
    case.update(wav_sha256=study.sha(wav), pcm_sha256=hashlib.sha256(pcm).hexdigest())
    case[study.NATIVE_HASH] = study.NATIVE.float_sha(study.float_input(pcm))
    assert study.baseline_waveform(case) == wav.read_bytes()
    expected = np.tile(np.array([-1, -1 / 32768, 0, 32767 / 32768], dtype=np.float32), 12_000)
    np.testing.assert_array_equal(study.candidate_waveform(case), expected)
    with pytest.raises(ValueError, match="serving samples"):
        study.candidate_waveform({**case, study.NATIVE_HASH: "wrong"})
    monkeypatch.setattr(study.candidate, "audit_observation", lambda *_: None)
    with pytest.raises(ValueError, match="packet differs"):
        study.audit_candidate({"native_features": {"input_pcm_sha256": "wrong"}}, case, {}, {}, {})


def test_baseline_receipt_accepts_exact_new_batch_size(tmp_path, monkeypatch):
    root, public = tmp_path / "baseline", tmp_path / "freeze.json"
    public.write_text("{}\n")
    monkeypatch.setattr(study, "BASELINE", root)
    monkeypatch.setattr(study, "PUBLIC", public)
    cases, rows, _ = fixture_rows()
    selected = rows[0 :: len(study.CONDITIONS)]
    stage = root / "extract-00"
    output = stage / "output"
    output.mkdir(parents=True)
    profile, journal = stage / "profile.json", output / "private-observations.jsonl"
    profile.write_text("{}\n")

    def saved_receipt(batch):
        journal.write_text("".join(json.dumps(row) + "\n" for row in batch))
        return {
            "status": "complete",
            "batch": 0,
            "condition": "matched_clean",
            "freeze_sha256": study.sha(public),
            "private_observations_sha256": study.sha(journal),
            "new_neural_calls": sum(row["pella_forwards"] for row in batch),
        }

    receipt = saved_receipt(selected)
    monkeypatch.setattr(study.NATIVE.PILOT.speech, "qualification_profile", lambda *_: receipt)
    verified, _, count = study.completed_baseline(0, cases)
    assert len(verified) == 72 and count == 6
    receipt = saved_receipt(selected[:48])
    with pytest.raises(ValueError, match="72-row"):
        study.completed_baseline(0, cases)
