import pytest

from scripts import audio_commons_slice_screen_v247 as screen


def test_slice_grid_covers_fixed_nonoverlapping_windows():
    size = screen.SAMPLES
    assert screen.starts(3 * size + 101) == [0, size, 2 * size]
    selected = screen.starts(40 * size)
    assert len(selected) == len(set(selected)) == 12
    assert selected[0] == 0 and selected[-1] == 39 * size
    assert all(a + size <= b for a, b in zip(selected, selected[1:], strict=False))


def test_screen_outcomes_at_frozen_point():
    assert screen.screen_status(0, 0.29, 0.29, False) == "machine_screened_provisional"
    assert screen.screen_status(0, 0.30, 0.01, False) == "voice_signal_detected"
    assert screen.screen_status(1, 0, 0, False) == "voice_signal_detected"
    assert screen.screen_status(0, 0.01, 0.30, False) == "music_signal_detected"
    assert screen.screen_status(0, 0, 0, True) == "constant_audio"
    with pytest.raises(ValueError, match="Invalid machine screen evidence"):
        screen.screen_status(0, 0, float("nan"), False)


def test_packet_retains_only_exact_provisional_slice_bindings():
    row = {
        "environment": "traffic",
        "slot": 1,
        "start_sample": 48_000,
        "samples": 48_000,
        "pcm_sha256": "a" * 64,
        "qualification_receipt_sha256": "b" * 64,
    }
    reading = {
        **row,
        "voice_spans": 0,
        "speech_tag": 0.1,
        "music_tag": 0.2,
        "constant": False,
        "status": "machine_screened_provisional",
    }
    packet = screen.slice_packet([row], [reading])
    assert packet["excerpts"] == [row]
    assert (
        packet["evidence_type"] == "machine_screened_provisional" and not packet["human_listening_reviewed"]
    )
    with pytest.raises(ValueError, match="input identity differs"):
        screen.slice_packet([row], [{**reading, "start_sample": 0}])
