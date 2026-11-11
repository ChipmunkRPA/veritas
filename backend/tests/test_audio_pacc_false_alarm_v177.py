import hashlib

import numpy as np
import pytest

from app.detectors.audio.decode import wav_bytes
from scripts import audio_pacc_false_alarm_v177 as study


def test_exact_pcm_reader_rejects_incorrect_clock_hash_and_symlink(tmp_path, monkeypatch):
    media = tmp_path / "prepared"
    media.mkdir()
    monkeypatch.setattr(study.source, "MEDIA", media)
    path = media / "source-00-clean.wav"
    pcm = np.linspace(-0.75, 0.75, study.source.SAMPLES, dtype=np.float32)
    pcm = np.rint(pcm * 32768) / np.float32(32768)
    data = wav_bytes(pcm)
    path.write_bytes(data)
    row = {
        "slot": "source-00",
        "condition": "clean",
        "status": "qualified",
        "source_range": study.SOURCE_RANGE,
        "wav_sha256": hashlib.sha256(data).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
    }
    assert np.array_equal(study.read_pcm(row), pcm)
    with pytest.raises(ValueError, match="production sample clock"):
        study.read_pcm({**row, "source_range": {**row["source_range"], "source_start_pts": 1}})
    with pytest.raises(ValueError, match="production PCM differs"):
        study.read_pcm({**row, "pcm_sha256": "wrong"})
    outside = tmp_path / "out-of-scope.wav"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="Only the direct saved PACC-P WAV"):
        study.read_pcm(row)


def outcome(slot, decision, **extra):
    return {
        "slot": slot,
        "decision": decision,
        "route_status": "not_applicable" if decision == "unavailable" else "ok",
        **extra,
    }


def test_matched_false_ai_and_speech_gate_flips_use_planned_denominator():
    clean = [outcome("a", "ai"), outcome("b", "original"), outcome("c", "inconclusive")]
    noisy = [outcome("a", "unavailable"), outcome("b", "ai"), outcome("c", "ai")]
    metrics = study.paired_metrics(clean, noisy)
    assert metrics["planned_pairs"] == 3
    assert metrics["both_scored"] == 2
    assert metrics["both_scored_coverage"] == 2 / 3
    assert metrics["new_false_ai"] == 2
    assert metrics["resolved_false_ai"] == 1
    assert metrics["net_false_ai_rate_change"] == 1 / 3
    assert {"clean": "ok", "noisy": "not_applicable", "count": 1} in metrics["route_status_transitions"]
    assert {"clean": "inconclusive", "noisy": "ai", "count": 1} in metrics["direction_transitions"]
    assert study.condition_metrics(noisy)["false_ai_rate"] == 2 / 3
    with pytest.raises(ValueError, match="Exact clean-to-noisy pairs"):
        study.paired_metrics(clean, list(reversed(noisy)))


def test_ami_and_vctk_contexts_are_reported_separately_with_both_mics():
    plan = [
        {
            "slot": "a",
            "pool": "AMI",
            "family": "AMI",
            "source_context": "meeting",
            "known_historical_voice": False,
        },
        *[
            {
                "slot": f"v{i}",
                "pool": f"VCTK_mic{i}",
                "family": "VCTK",
                "source_context": "same utterance",
                "known_historical_voice": True,
            }
            for i in (1, 2)
        ],
    ]
    rows = [
        outcome(
            item["slot"],
            "ai" if condition == "babble_snr00" and item["family"] == "AMI" else "unavailable",
            condition=condition,
            family=item["family"],
        )
        for item in plan
        for condition in study.source.CONDITIONS
    ]
    result = study.summary(rows, plan, rows)
    assert result["AMI"]["known_historical_voice_inputs"] == 0
    assert result["AMI"]["paired_vs_clean"]["babble_snr00"]["new_false_ai_rate"] == 1
    assert result["VCTK"]["known_historical_voice_inputs"] == 2
    assert result["VCTK"]["declared_source_contexts"] == 1
    assert result["VCTK"]["by_condition"]["clean"]["unavailable_rate"] == 1
    assert result["VCTK_mic1"]["paired_vs_clean"]["babble_snr00"]["planned_pairs"] == 1
    assert result["VCTK_mic2"]["paired_vs_clean"]["babble_snr00"]["planned_pairs"] == 1
    with pytest.raises(ValueError, match="Exact ordered serving outcomes"):
        study.summary(rows[:-1], plan, rows)
