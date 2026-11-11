import hashlib
import subprocess
import sys
import wave

import numpy as np
import pytest

from scripts import audio_voxforge_tr_transfer_v191 as study


def test_cold_import_does_not_initialize_settings_or_model_runtime():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_voxforge_tr_transfer_v191 as s; s.require_model_free()",
        ],
        cwd=study.ROOT / "backend",
        check=True,
        capture_output=True,
        timeout=15,
    )


def test_model_free_prepared_wave_binds_exact_production_clock(tmp_path, monkeypatch):
    stage = tmp_path / "bind-00"
    folder = stage / "output/prepared"
    folder.mkdir(parents=True)
    monkeypatch.setattr(study, "PREPARATION_STAGE", stage)
    samples = 43_408
    signed = np.arange(samples, dtype=np.int32) - samples // 2
    path = folder / "genuine-00.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, study.RATE, samples, "NONE", "not compressed"))
        wav.writeframes(signed.astype("<i2").tobytes())
    pcm = signed.astype("<f4") / np.float32(32768)
    row = {
        "slot": "genuine-00",
        "status": "qualified",
        "samples": samples,
        "sample_rate": study.RATE,
        "source_range": {
            "start_sample": 0,
            "end_sample": samples,
            "source_start_pts": 0,
            "source_end_pts": samples,
            "time_base": "1/16000",
        },
        "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
    }
    study.read_prepared_wave(row)
    with pytest.raises(ValueError, match="production PCM differs"):
        study.read_prepared_wave({**row, "pcm_sha256": "wrong"})
    with pytest.raises(ValueError, match="production sample clock"):
        study.read_prepared_wave({**row, "source_range": {**row["source_range"], "source_start_pts": 1}})
    outside = tmp_path / "not-admitted.wav"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="direct qualified v192 WAV"):
        study.read_prepared_wave(row)


def test_pilot_separates_unqualified_speech_gate_and_directional_errors():
    prepared = [
        {"slot": f"genuine-{i:02d}", "status": "qualified" if i < 3 else "unavailable", "samples": 48002}
        for i in range(12)
    ]
    rows = []
    for i, case in enumerate(prepared):
        if i < 2:
            row = {
                "route_status": "ok",
                "windows": [[0, 48002]],
                "decision": "ai" if i == 0 else "original",
                "stability_probe": "not_required_for_decision" if i == 0 else "computed",
                "pella_forwards": i + 1,
            }
        else:
            row = {**study.unavailable(), "pella_forwards": 0}
            if i == 2:
                row["route_status"] = "not_applicable"
        rows.append({"slot": case["slot"], **row})
    alania = {
        "all_positive_cases": {
            "planned_positives": 24,
            "scored": 24,
            "directions": {
                "ai": 21,
                "original": 2,
                "inconclusive": 1,
                "unavailable": 0,
            },
            "missed_ai": 3,
        },
        "by_render_mode": {},
    }
    report = study.summary(rows, prepared, alania)
    genuine = report["genuine_voxforge"]
    assert genuine["unqualified_source_slots"] == 9
    assert genuine["speech_gate_rejections"] == 1
    assert genuine["scored_slots"] == 2
    assert genuine["directions_over_planned"] == {
        "ai": 1,
        "original": 1,
        "inconclusive": 0,
        "unavailable": 10,
    }
    assert genuine["directions_over_qualified"]["unavailable"] == 1
    assert genuine["wrong_ai_share_of_qualified"] == 1 / 3
    assert report["synthetic_alania_prior_result"]["wrong_original"] == 2
    rows[3].update(decision="original")
    with pytest.raises(ValueError, match="Exact serving direction"):
        study.summary(rows, prepared, alania)
