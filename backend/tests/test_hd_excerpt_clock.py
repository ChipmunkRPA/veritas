from fractions import Fraction

import pytest

from scripts.hd_excerpt_clock import admit_excerpt

PICTURES = tuple(range(0, 8001, 40))


def test_half_millisecond_rounding_uses_decoded_origin_and_exact_sample_counts():
    audio = [(0, 648), *((14 + 20 * i, 960) for i in range(400))]
    result = admit_excerpt(PICTURES, "1/1000", audio, "1/1000", 48_000, ("1/50", "8"))
    assert result["status"] == "ready"
    assert result["decoded_audio_clock"]["decoded_origin_pts"] == 0
    assert result["decoded_audio_clock"]["pts_residual_range_seconds"] == ["0", "1/2000"]
    assert result["excerpt"] == {
        "source_seconds_half_open": ["1/25", "151/25"],
        "duration_seconds": "6",
        "video_pts_half_open": [40, 6040],
        "video_frame_indices_half_open": [1, 151],
        "audio_sample_indices_half_open": [1920, 289920],
        "audio_frame_indices_half_open": [2, 303],
        "first_audio_frame_sample_offset": 312,
        "last_audio_frame_sample_end_offset": 312,
        "conditional_audio_sample_times_half_open": ["1/25", "151/25"],
        "audio_boundary_offsets_from_video_seconds": ["0", "0"],
    }


def test_fractional_ntsc_pts_skip_to_first_exact_pcm_boundary():
    pictures = [i * 1001 for i in range(250)]
    audio = [(i * 1024, 1024) for i in range(380)]
    result = admit_excerpt(
        pictures,
        "1/30000",
        audio,
        "1/48000",
        48_000,
        ("1/100", "8"),
        audio_quantization="exact",
        sample_boundary_policy="coincident",
    )
    excerpt = result["excerpt"]
    assert excerpt["video_frame_indices_half_open"] == [5, 185]
    assert excerpt["source_seconds_half_open"] == ["1001/6000", "37037/6000"]
    assert excerpt["audio_sample_indices_half_open"] == [8008, 296296]
    assert excerpt["duration_seconds"] == "3003/500"
    assert result["decoded_audio_clock"]["pts_residual_range_seconds"] == ["0", "0"]


def test_gap_far_beyond_coarse_audio_clock_is_explicitly_unavailable():
    result = admit_excerpt(PICTURES, "1/1000", [(13, 576), (36, 1024)], "1/1000", 44_100, (0, 8))
    assert result == {
        "status": "unavailable",
        "reason": "audio_gap_exceeds_quantization",
        "excerpt": None,
        "audio_frame_index": 1,
        "adjacent_error_seconds": str(Fraction(23, 1000) - Fraction(576, 44100)),
        "audio_time_base": "1/1000",
    }


@pytest.mark.parametrize(
    ("second_pts", "reason"),
    [
        (1025, "audio_gap_exceeds_quantization"),
        (1023, "audio_overlap_exceeds_quantization"),
        (-1, "audio_pts_out_of_order"),
    ],
)
def test_exact_audio_sample_clock_rejects_one_sample_gap_or_overlap(second_pts, reason):
    result = admit_excerpt(PICTURES, "1/1000", [(0, 1024), (second_pts, 1024)], "1/48000", 48000, (0, 8))
    assert result["reason"] == reason
    assert result["excerpt"] is None


def test_subtick_errors_cannot_walk_to_inconsistent_audio_phase():
    result = admit_excerpt(PICTURES, "1/1000", [(0, 648), (14, 648), (28, 960)], "1/1000", 48000, (0, 8))
    assert result["reason"] == "audio_quantization_phase_drift"
    assert result["pts_residual_range_seconds"] == ["0", "1/1000"]


def test_one_tick_phase_spread_requires_a_declared_closed_rounding_intersection():
    audio = [(0, 648), (14, 648), *((28 + 20 * i, 960) for i in range(400))]
    result = admit_excerpt(
        PICTURES, "1/1000", audio, "1/1000", 48000, (0, 8), audio_quantization="nearest_closed"
    )
    clock = result["decoded_audio_clock"]
    assert clock["decoded_origin_seconds"] == "0"
    assert clock["conditional_lattice_origin_seconds"] == "1/2000"
    assert clock["conditional_lattice_origin_basis"] == "unique_quantization_intersection"
    assert clock["compatible_origin_bounds"] == {
        "seconds": ["1/2000", "1/2000"],
        "lower_inclusive": True,
        "upper_inclusive": True,
    }
    assert result["excerpt"]["audio_sample_indices_half_open"] == [1896, 289896]
    assert result["excerpt"]["video_pts_half_open"] == [40, 6040]
    unavailable = admit_excerpt(
        PICTURES, "1/1000", audio, "1/1000", 48000, (0, 8), audio_quantization="nearest_half_open"
    )
    assert unavailable["reason"] == "audio_quantization_phase_drift"


def test_no_convenient_origin_is_chosen_when_nearest_phase_is_unresolved():
    result = admit_excerpt(
        PICTURES,
        "1/1000",
        [(0, 643), (14, 480000)],
        "1/1000",
        48000,
        (0, 8),
        audio_quantization="nearest_closed",
    )
    assert result["reason"] == "audio_origin_phase_unresolved"
    assert result["compatible_origin_bounds"]["seconds"] == ["1/9600", "1/2000"]


def test_small_quantized_overlap_retained_and_last_audio_end_is_conservative():
    audio = [(0, 648), (13, 287352)]
    result = admit_excerpt(PICTURES, "1/1000", audio, "1/1000", 48000, (0, 8))
    assert result["reason"] == "no_common_boundaries_within_duration"
    assert result["decoded_audio_clock"]["pts_residual_range_seconds"] == ["-1/2000", "0"]
    assert result["decoded_audio_clock"]["conditional_lattice_end_seconds"] == "6"
    assert result["decoded_audio_clock"]["reported_frame_end_seconds"] == "11999/2000"


def test_shot_end_is_an_exclusion_boundary_and_no_final_duration_is_inferred():
    audio = [(0, 480_000)]
    result = admit_excerpt(PICTURES, "1/1000", audio, "1/1000", 48000, (0, 6))
    assert result["excerpt"]["video_frame_indices_half_open"] == [0, 150]
    assert result["excerpt"]["audio_frame_indices_half_open"] == [0, 1]
    unavailable = admit_excerpt(PICTURES[:150], "1/1000", audio, "1/1000", 48000, (0, 6))
    assert unavailable["reason"] == "no_common_boundaries_within_duration"


def test_first_decoded_audio_pts_can_differ_from_the_video_origin():
    result = admit_excerpt(PICTURES, "1/1000", [(-5, 480_000)], "1/1000", 48000, (0, 8))
    assert result["decoded_audio_clock"]["decoded_origin_seconds"] == "-1/200"
    assert result["excerpt"]["audio_sample_indices_half_open"] == [240, 288240]
    assert result["excerpt"]["conditional_audio_sample_times_half_open"] == ["0", "6"]


def test_inside_policy_keeps_native_ntsc_endpoints_and_reports_subsample_offsets():
    pictures = [1057056 + 1001 * i for i in range(183)]
    audio = [(i * 1024, 1024) for i in range(2030)]
    shot = (Fraction(1057056, 30000), Fraction(1239238, 30000))
    result = admit_excerpt(pictures, "1/30000", audio, "1/48000", 48000, shot, audio_quantization="exact")
    excerpt = result["excerpt"]
    assert excerpt["video_pts_half_open"] == [1057056, 1237236]
    assert excerpt["video_frame_indices_half_open"] == [0, 180]
    assert excerpt["audio_sample_indices_half_open"] == [1691290, 1979578]
    assert excerpt["audio_boundary_offsets_from_video_seconds"] == ["1/120000", "1/120000"]
    assert excerpt["conditional_audio_sample_times_half_open"] == [
        str(Fraction(1691290, 48000)),
        str(Fraction(1979578, 48000)),
    ]
    unavailable = admit_excerpt(
        pictures, "1/30000", audio, "1/48000", 48000, shot, sample_boundary_policy="coincident"
    )
    assert unavailable["reason"] == "no_exact_common_boundaries_within_duration"


def test_audio_indices_ceiling_keeps_only_sample_starts_inside_video_span():
    result = admit_excerpt([1, 6002], "1/1000", [(0, 400000)], "1/44100", 44100, (0, 8))
    lo, hi = result["excerpt"]["audio_sample_indices_half_open"]
    assert Fraction(lo - 1, 44100) < Fraction(1, 1000) <= Fraction(lo, 44100)
    assert Fraction(hi - 1, 44100) < Fraction(6002, 1000) <= Fraction(hi, 44100)


@pytest.mark.parametrize("bad_pts", [(0, 0, 6000), (0, 6000, 4000), (False, 6000), (0, 6000.0)])
def test_native_video_pts_must_be_strictly_increasing_integers(bad_pts):
    with pytest.raises(ValueError, match="strictly increasing"):
        admit_excerpt(bad_pts, "1/1000", [(0, 480000)], "1/1000", 48000, (0, 8))


def test_duration_limit_is_inclusive_and_clock_floats_are_rejected():
    audio = [(0, 480000)]
    assert admit_excerpt([0, 7200], "1/1000", audio, "1/1000", 48000, (0, 8))["status"] == "ready"
    assert admit_excerpt([0, 7201], "1/1000", audio, "1/1000", 48000, (0, 8))["status"] == "unavailable"
    with pytest.raises(ValueError, match="exact rational"):
        admit_excerpt(PICTURES, 0.001, audio, "1/1000", 48000, (0, 8))
