import hashlib

import numpy as np
import pytest

from app.detectors.audio.decode import wav_bytes
from scripts import audio_babble_overlap_comparison_v183 as study


def cases():
    return [
        {
            "case": f"pair-{i:02d}-{arm}-{condition}",
            "pair": f"pair-{i:02d}",
            "component": f"opaque-{i // 4}",
            "pool": f"pool-{i // 4:02d}",
            "arm": arm,
            "condition": condition,
            "label": int(arm == "synthetic"),
            "ai_support_samples": [0, study.SAMPLES] if arm == "synthetic" else None,
            "status": "constructed",
            "samples": study.SAMPLES,
            "sample_rate": study.RATE,
            "gain": 0.95,
            "shared_noise_pcm_sha256": condition if condition != "clean" else None,
            "complete_paired_treatment": True,
        }
        for i in range(study.PAIR_COUNT)
        for arm in study.ARMS
        for condition in study.CONDITIONS
    ]


def overlap_record(marker):
    count = 15 if marker else 0
    start = (study.overlap.RECEPTIVE - study.overlap.STEP) // 2
    length = count * study.overlap.STEP
    return {
        "eligible_frames": (study.SAMPLES - study.overlap.RECEPTIVE) // study.overlap.STEP + 1,
        "overlap_frames": count,
        "longest_overlap_samples": length,
        "diagnostic_context": marker,
        "private_intervals_samples": [[start, start + length]] if marker else [],
        "mean_overlap_mass": 0.2,
    }


def test_matched_case_validation_rejects_different_noise_and_pool():
    planned = cases()
    study.validate_cases(planned)
    planned[4]["shared_noise_pcm_sha256"] = "different"
    with pytest.raises(ValueError, match="share the exact babble residual"):
        study.validate_cases(planned)
    planned = cases()
    planned[1]["pool"] = "pool-03"
    with pytest.raises(ValueError, match="changed babble pool"):
        study.validate_cases(planned)


def test_clean_is_subject_to_the_same_rule_and_unavailable_keeps_denominator():
    planned = cases()
    planned[-1]["status"] = "unavailable"
    rows = []
    for case in planned:
        ready = case["status"] == "constructed"
        marker = ready and case["pair"] == "pair-00"
        if not ready:
            direction = "unavailable"
        elif case["pair"] == "pair-01" and case["arm"] == "synthetic":
            direction = "original"
        else:
            direction = "ai"
        rows.append(
            {
                **case,
                "route_status": "ok" if ready else "preparation_unavailable",
                "windows": [[0, study.SAMPLES]] if ready else [],
                "raw": 0.9 if case["arm"] == "synthetic" else 0.8,
                "decision": direction,
                "stability_probe": "computed"
                if direction == "original"
                else ("not_required_for_decision" if ready else "not_run"),
                "pella_forwards": int(ready) + int(direction == "original"),
                "overlap": overlap_record(marker) if ready else None,
                "overlap_forwards": int(ready),
                "overlap_abstention": study.comparator(direction, marker),
            }
        )
    result = study.summary(rows, planned)
    panel = result["combined_panel"]
    clean = panel["by_condition"]["clean"]
    assert clean["arms"]["genuine"]["baseline"]["false_ai"] == 16
    assert clean["arms"]["genuine"]["overlap_abstention"]["false_ai"] == 15
    assert clean["arms"]["synthetic"]["baseline"]["wrong_original"] == 1
    assert clean["arms"]["synthetic"]["overlap_abstention"]["wrong_original"] == 1
    assert clean["paired_foreground_sensitivity"]["overlap_abstention"]["neither_ai"] == 1
    assert panel["overlap_only"]["clean"]["genuine"]["marker"] == 1
    plus5 = panel["by_condition"]["snr_plus5db"]["arms"]["synthetic"]
    assert plus5["planned"] == 16 and plus5["scored"] == 15
    assert plus5["overlap_abstention"]["missed_ai"] == 3
    assert len(result["by_connected_component"]) == 4
    assert "opaque-0" not in str(result)
    rows[0]["overlap_abstention"] = "ai"
    with pytest.raises(ValueError, match="one-way abstention"):
        study.summary(rows, planned)


def test_pcm_reader_and_mix_verifier_bind_exact_prepared_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(study.source, "MEDIA", tmp_path)
    clean, noisy = cases()[:2]
    signed = np.arange(study.SAMPLES, dtype=np.int32) - 24000
    residual = np.resize(np.array([3, -7], dtype="<i2"), study.SAMPLES)
    noisy["shared_noise_pcm_sha256"] = hashlib.sha256(residual.tobytes()).hexdigest()
    for row, samples in ((clean, signed), (noisy, signed + residual)):
        pcm = samples.astype("<f4") / np.float32(32768)
        data = wav_bytes(pcm)
        path = tmp_path / f"{row['case']}.wav"
        path.write_bytes(data)
        row.update(
            {
                "file": str(path),
                "wav_sha256": hashlib.sha256(data).hexdigest(),
                "pcm_s16le_sha256": hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest(),
                "production_f32_pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            }
        )
        assert np.array_equal(study.read_pcm(row), pcm)
    study.verify_mixtures([clean, noisy])
    with pytest.raises(ValueError, match="Observed production babble residual differs"):
        study.verify_mixtures([clean, {**noisy, "shared_noise_pcm_sha256": "wrong"}])
    with pytest.raises(ValueError, match="Only a direct constructed"):
        study.read_pcm({**clean, "file": str(tmp_path / "outside.wav")})


def test_unavailable_case_never_opens_wav_or_scores(monkeypatch):
    def unavailable(*_):
        pytest.fail("Unavailable case must not read or score")

    monkeypatch.setattr(study, "read_pcm", unavailable)
    result = study.assess({"status": "failed"}, unavailable, unavailable)
    assert result["decision"] == result["overlap_abstention"] == "unavailable"
    assert result["overlap"] is None and result["overlap_forwards"] == 0
