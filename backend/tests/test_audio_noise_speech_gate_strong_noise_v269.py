import struct

import pytest

from scripts import audio_noise_speech_gate_strong_noise_v269 as challenge


def waveform(pattern):
    return struct.pack(f"<{len(pattern)}h", *pattern) * (challenge.shared.SAMPLES // len(pattern))


def test_exact_stronger_levels_preserve_shared_residuals_and_current_mixer():
    foregrounds = {"genuine": waveform((5000, -5000)), "alania": waveform((10000, 10000, -10000, -10000))}
    tapes = {"kitchen": waveform((2000, 1000, -2000, -1000)), "traffic": waveform((6000, -6000))}
    matched = challenge.mix_shared_noise(foregrounds, tapes)
    assert set(matched.noise_s16le) == {(name, level) for name in tapes for level in (-5, -10)}
    for arm in matched.variants.values():
        for key, measured in arm.measured_snr_db.items():
            assert measured == pytest.approx(key[1], abs=0.1)
            actual = challenge.shared.samples(arm.mixed_s16le[key]).astype("i4")
            clean = challenge.shared.samples(arm.clean_s16le).astype("i4")
            assert ((actual - clean) == challenge.shared.samples(matched.noise_s16le[key])).all()
    assert challenge.shared.SNR_LEVELS_DB == challenge.controls.SNR_LEVELS_DB == (0, 5)


def build_rows(conditions, *, unavailable=False):
    for condition in conditions:
        row = {
            "triplet": "triplet-00",
            "arm": "genuine",
            "condition": condition,
            "case": f"case-{condition}",
            "status": "unavailable" if unavailable else "prepared",
        }
        if unavailable:
            yield {**row, "reason": "source_unavailable"}
        else:
            tape = condition.rsplit("_", 1)[0] if condition != "matched_clean" else None
            yield {
                **row,
                "source_wav_sha256": "wav",
                "source_prefix_sha256": "prefix",
                "noise_sample_offset": 1 if tape else None,
                "noise_excerpt_sha256": tape,
            }


def test_binding_preserves_missing_sources_and_rejects_offset_substitution():
    prior = list(build_rows(challenge.prior.CONDITIONS))
    rows = list(build_rows(challenge.CONDITIONS))
    assert len(challenge.bind_cases(prior, [(row, b"payload") for row in rows])) == len(rows)
    rows[1]["noise_sample_offset"] += 1
    with pytest.raises(ValueError, match="tape excerpt changed"):
        challenge.bind_cases(prior, [(row, b"payload") for row in rows])
    prior = list(build_rows(challenge.prior.CONDITIONS, unavailable=True))
    rows = list(build_rows(challenge.CONDITIONS, unavailable=True))
    assert all(row["payload"] is None for row in challenge.bind_cases(prior, [(row, None) for row in rows]))
    rows[0]["reason"] = "different failure"
    with pytest.raises(ValueError, match="Former unavailability"):
        challenge.bind_cases(prior, [(row, None) for row in rows])


def test_stronger_summary_does_not_count_previous_backgrounds_as_new_calls():
    row = next(build_rows(("kitchen_-5db",))) | {"payload": b"waveform"}
    observed = {"matched": [{"case": row["case"], "spans": [[0, 16000]]}], "unscreened_background": []}
    result = challenge.summarize([row], {("genuine", "triplet-00"): "a"}, [], observed)
    assert result["by_arm_and_condition"]["genuine"]["kitchen_-5db"]["additional_admissions"] == 1
    assert result["background_controls_rerun"] is False
    assert result["vad_frame_forwards"] == 94
    with pytest.raises(ValueError, match="must not enter"):
        challenge.summarize([row], {}, [{}], observed)
