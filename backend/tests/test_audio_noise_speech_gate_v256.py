import hashlib

import pytest

from scripts import audio_noise_speech_gate_v256 as gate


def test_rescue_preserves_serving_gate_and_requires_one_second_span():
    assert gate.coverage([[0, 12_000], [24_000, 36_000]]) == {
        "speech_samples": 24_000,
        "longest_span_samples": 12_000,
        "baseline": True,
        "rescue": True,
    }
    assert gate.coverage([[0, 16_000]])["rescue"]
    assert not gate.coverage([[0, 15_999]])["rescue"]
    assert not gate.coverage([[0, 8_000], [24_000, 32_000]])["rescue"]
    assert not gate.coverage([])["rescue"]
    with pytest.raises(ValueError, match="Invalid VAD span"):
        gate.coverage([[0, 12_000], [11_999, 24_000]])


def test_unscreened_background_does_not_reuse_screened_positions_or_pcm():
    def window(value):
        return bytes([value, 0]) * gate.SAMPLES

    screened = [
        {
            "slot": 1,
            "start_sample": 0,
            "environment": "kitchen",
            "pcm_sha256": hashlib.sha256(window(1)).hexdigest(),
        }
    ]
    [row] = gate.complement(screened, {1: window(1) + window(1) + window(2) + b"\0\0"})
    assert row == {
        "slot": 1,
        "start_sample": 2 * gate.SAMPLES,
        "environment": "kitchen",
        "sha256": hashlib.sha256(window(2)).hexdigest(),
    }


def test_summary_preserves_paired_genuine_and_background_denominators():
    cases = [
        {"case": str(i), "arm": "genuine", "group": "private", "status": "prepared", "condition": condition}
        for i, condition in enumerate(gate.prior.CONDITIONS)
    ]
    recorded = {row["case"]: "ok" if i == 0 else "not_applicable" for i, row in enumerate(cases)}
    controls = [{"slot": 1, "start_sample": 0, "environment": "kitchen", "sha256": "private"}]
    observed = {
        "matched": [
            {"case": row["case"], "spans": [[0, 30_000 if i == 0 else 16_000]]} for i, row in enumerate(cases)
        ],
        "unscreened_background": [{"control": controls[0], "spans": [[0, 16_000]]}],
    }
    result = gate.summarize(cases, recorded, controls, observed)
    transitions = result["clean_admitted_genuine_to_noise"]
    assert transitions["baseline"]["groups_with_all_four_noisy_excluded"] == 1
    assert transitions["rescue"]["groups_with_any_noisy_exclusion"] == 0
    assert (
        result["unscreened_background_by_environment"]["kitchen"]["recordings_with_additional_admissions"]
        == 1
    )
    assert result["vad_frame_forwards"] == 6 * 94
    assert "private" not in repr(result)
    with pytest.raises(ValueError, match="Replayed baseline speech admission differs"):
        gate.summarize(cases, {**recorded, "1": "ok"}, controls, observed)
