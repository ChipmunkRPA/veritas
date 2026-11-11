from dataclasses import asdict

import pytest

from scripts import review_grid_timing_opportunity as review


def window(index, offset, peak, *, low=False):
    start = 25 * index // 2
    return {
        "window_index": index,
        "unit_ref": f"track:0@window={index}",
        "start_tick": start,
        "end_tick": start + 25,
        "hop_ticks": 12.5,
        "speech_fraction": 0.8,
        "median_face_px": 120,
        "offset_ticks": offset,
        "peak_correlation": peak,
        "low_correlation": low,
        "lag_profile": [
            {"offset_ticks": lag, "correlation": peak if lag == offset else 0.0} for lag in range(-15, 16)
        ],
    }


def track(windows, reading, baseline=None, threshold=None, segments=()):
    return {
        "track_id": "track:0",
        "minimum_trusted_peak": 0.4,
        "windows": windows,
        "reading": reading,
        "baseline_offset_ticks": baseline,
        "collapse_threshold": threshold,
        "segments": [asdict(segment) for segment in segments],
    }


def test_short_audio_cap_and_exact_threshold_for_four_search_windows():
    assert review.audio_search_indices(47_648) == {2}
    assert review.audio_search_indices(77 * 640 - 1) == {2}
    assert review.audio_search_indices(77 * 640) == {2, 3}
    assert review.audio_search_indices(102 * 640 - 1) == {2, 3, 4}
    assert review.audio_search_indices(102 * 640) == {2, 3, 4, 5}


def test_one_trusted_peak_is_measured_but_cannot_emit_change():
    saved = track([window(2, 8, 0.4)], "no_stable_offset")
    result = review.track_opportunity(saved, {2})
    assert result["trusted_windows"] == 1
    assert result["stable_offset_runs"] == 0
    assert not result["minimum_step_support"]
    assert not result["minimum_collapse_support"]


def test_step_requires_two_stable_runs_and_accepts_a_gap():
    windows = [window(i, lag, 0.75) for i, lag in [(2, 0), (3, 0), (5, 3), (6, 3)]]
    segment = review.lipsync.Segment("offset_step", 62.0, 100, 2, offset=3, baseline=0)
    saved = track(windows, "offset_step", baseline=0, threshold=0.375, segments=[segment])
    result = review.track_opportunity(saved, {2, 3, 5, 6})
    assert result["minimum_step_support"]
    assert result["stable_offset_runs"] == 2
    assert result["steps"] == 1
    assert result["collapses"] == 0


@pytest.mark.parametrize("second_index", [3, 4])
def test_collapse_requires_consecutively_indexed_low_windows(second_index):
    windows = [
        window(2, 0, 0.125, low=True),
        window(second_index, 0, 0.125, low=True),
        window(6, 0, 0.75),
        window(7, 0, 0.75),
    ]
    adjacent = second_index == 3
    segment = review.lipsync.Segment("confidence_collapse", 25, 62, 2, confidence=0.125)
    saved = track(
        windows,
        "confidence_collapse" if adjacent else "in_sync",
        baseline=0,
        threshold=0.21875,
        segments=[segment] if adjacent else [],
    )
    result = review.track_opportunity(saved, {2, second_index, 6, 7})
    assert result["minimum_collapse_support"]
    assert result["collapses"] == int(adjacent)
    assert not result["minimum_step_support"]


def test_changed_winning_lag_cannot_be_read_as_measured():
    saved = window(2, 8, 0.75)
    saved["offset_ticks"] = 7
    with pytest.raises(ValueError, match="winning lag"):
        review.replay_window("track:0", saved, {2})


def test_exact_delayed_value_routing_and_unchanged_video():
    support = {
        "status": "ready",
        "shift_samples": 5120,
        "output_samples_half_open": [17_280, 30_080],
        "source_samples_half_open": [12_160, 24_960],
        "preserve_length_origins_and_outside_samples": True,
        "padding_or_wraparound": False,
    }
    expected = {
        "audio_samples": 47_648,
        "audio_start_seconds": "0",
        "rgb_sha256": ["rgb"],
        "picture_pts_ms": [0],
    }

    def segment(out_a, out_b, in_a, in_b):
        return {
            "output_samples_half_open": [out_a, out_b],
            "input_samples_half_open": [in_a, in_b],
            "input_source_pts_half_open": [in_a, in_b],
            "time_base": "1/16000",
        }

    base = {
        "expected": expected | {"pcm_s16_sha256": "original"},
        "waveform_id": "original_pcm",
        "changed_audio_samples": 0,
        "audio_value_mapping": [segment(0, 47_648, 0, 47_648)],
    }
    delay = {
        "expected": expected | {"pcm_s16_sha256": "delayed"},
        "waveform_id": "delayed_audio_control",
        "source_time_bounds_seconds": ["27/25", "47/25"],
        "changed_audio_samples": 12_798,
        "changed_native_pixels": 0,
        "audio_value_mapping": [
            segment(0, 17_280, 0, 17_280),
            segment(17_280, 30_080, 12_160, 24_960),
            segment(30_080, 47_648, 30_080, 47_648),
        ],
    }
    result = review.delay_contract({"audio_delay": support}, base, delay)
    assert result["shift_ticks"] == 8
    assert result["source_seconds_half_open"] == ["19/25", "39/25"]
    delay["audio_value_mapping"][1]["input_samples_half_open"][0] += 1
    with pytest.raises(ValueError, match="value routing"):
        review.delay_contract({"audio_delay": support}, base, delay)
