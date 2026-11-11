import io
import struct
import wave

import pytest

from scripts import audio_noise_speech_rescue_downstream_v270 as downstream


def cohort_fixture():
    cases, observed, groups = [], [], {}
    for group, admission in (("triplet-00", 16000), ("triplet-01", 24000), ("triplet-02", 8000)):
        for condition, samples in ((downstream.CLEAN, 24000), (downstream.NOISE, admission)):
            case = {
                "case": f"{group}-{condition}",
                "triplet": group,
                "arm": "genuine",
                "condition": condition,
                "status": "prepared",
                "source_wav_sha256": group,
                "source_prefix_sha256": group,
            }
            cases.append(case)
            observed.append({"case": case["case"], "spans": [[0, samples]]})
        groups["genuine", group] = group
    return cases, groups, observed


def test_all_controls_and_own_clean_mates_are_required(monkeypatch):
    monkeypatch.setattr(downstream, "EXPECTED", {"genuine": {"rescued": 1, "admitted": 1}})
    cases, groups, observed = cohort_fixture()
    selected = downstream.select(cases, groups, observed)
    assert [(row["cohort"], row["condition"], row["original_gate_admitted"]) for row in selected] == [
        ("rescued", downstream.CLEAN, True),
        ("rescued", downstream.NOISE, False),
        ("admitted", downstream.CLEAN, True),
        ("admitted", downstream.NOISE, True),
    ]
    with pytest.raises(ValueError, match="same-arm control must be present"):
        downstream.select(cases[:2], groups, observed[:2])
    with pytest.raises(ValueError, match="requires a matched clean row"):
        downstream.select(cases[1:], groups, observed[1:])
    cases[0]["source_prefix_sha256"] = "unmatched"
    with pytest.raises(ValueError, match="own baseline-admitted matched clean"):
        downstream.select(cases, groups, observed)


def test_wave_wrapper_preserves_exact_signed16_payload():
    payload = struct.pack("<hh", -32768, 32767) * (downstream.gate.SAMPLES // 2)
    case = {"payload": payload, "pcm_sha256": downstream.stress.prior.digest(payload)}
    with wave.open(io.BytesIO(downstream.wav_bytes(case)), "rb") as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (
            1,
            2,
            downstream.gate.RATE,
        )
        assert reader.readframes(downstream.gate.SAMPLES) == payload
    with pytest.raises(ValueError, match="Exact frozen signed16 waveform"):
        downstream.wav_bytes({**case, "pcm_sha256": "different"})


def test_summary_keeps_original_gate_and_classifier_separate(monkeypatch):
    monkeypatch.setattr(downstream, "EXPECTED", {"genuine": {"rescued": 1, "admitted": 1}})
    cases, groups, observed = cohort_fixture()
    selected = downstream.select(cases, groups, observed)
    scores = [
        {
            "case": row["case"],
            "decision": "inconclusive",
            "raw": 0.5,
            "reference_estimate_ai": 0.5,
            "stability_probe": "not_required_for_decision",
            "stability_log_odds_change": None,
            "classifier_forwards": 1,
        }
        for row in selected
    ]
    result = downstream.summarize(selected, scores)
    rescue = result["by_arm_cohort_and_condition"]["genuine"]["rescued"][downstream.NOISE]
    assert rescue == {"assessed": 1, "directions": {"ai": 0, "original": 0, "inconclusive": 1}}
    assert result["new_neural_calls"] == 4 and result["vad_forwards"] == 0
    assert selected[1]["original_gate_admitted"] is False
