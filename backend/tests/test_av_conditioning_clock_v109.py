from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from scripts.av_conditioning_clock_v109 import exact, frame_plan, nominal_hull, select_chunk


def plan(index, t=None, *, audio="0", video="0", samples=32000, allowance="0"):
    return frame_plan(index, Fraction(index, 25) if t is None else t, audio, video, samples, allowance)


@pytest.mark.parametrize("index", [0, 1, 10, 25, 49])
def test_normalized_25hz_clock_matches_retained_tick_equation_and_layer_order(index):
    p = plan(index)
    assert p["encoder_anchor"] == index * 2
    hidden = np.arange(100 * 5 * 384, dtype=np.float32).reshape(100, 5, 384) + 1
    before = hidden.copy()
    actual = select_chunk(hidden, p).reshape(10, 5, 384)
    for slot, tick in enumerate(range(2 * index - 4, 2 * index + 6)):
        if 0 <= tick < 100:
            assert np.array_equal(actual[slot], hidden[tick])
        else:
            assert np.count_nonzero(actual[slot]) == 0
    assert np.array_equal(hidden, before)


def test_audio_lead_is_preserved_and_not_replaced_by_video_sample_zero():
    p = plan(0, "27/1000", video="27/1000")
    assert p["sample_position_fraction"] == "432"
    assert p["encoder_anchor"] == 1 and p["anchor_difference_from_video_zero"] == 1
    assert p["anchor_residual_seconds"] == "7/1000"


@pytest.mark.parametrize("step", ["1/24", "1/25", "1/30", "1001/30000", "1001/60000"])
def test_native_fractional_picture_times_do_not_assume_25hz(step):
    t = 20 * Fraction(step) + Fraction(27, 1000)
    p = plan(20, t, video="27/1000")
    assert p["encoder_anchor"] == (t * 50).__floor__()
    assert Fraction(p["anchor_residual_seconds"]) == t - Fraction(p["encoder_anchor"], 50)
    assert 0 <= Fraction(p["anchor_residual_seconds"]) < Fraction(1, 50)
    if Fraction(step) != Fraction(1, 25):
        assert p["anchor_difference_from_25hz_index"] != 0


def test_clock_quantization_preserves_ambiguous_anchor_instead_of_rounding_it_away():
    p = plan(0, "1/50", allowance="1/1000")
    assert p["encoder_anchor"] == 1
    assert p["possible_anchor_range"] == [0, 1] and p["clock_anchor_ambiguous"]


def test_nominal_convolution_hull_is_not_claimed_as_hidden_state_locality():
    assert nominal_hull(5) == [1080, 2120]
    assert Fraction(2120 - 1080, 16000) == Fraction(65, 1000)
    p = plan(10)
    hulls = p["nominal_convolution_hulls_samples"]
    assert hulls[-1][1] - hulls[0][0] == 3920
    assert p["nominal_convolution_windows_inside_source"]
    assert "whole padded-input context" in p["hidden_state_dependency"]
    assert p["generator_conditioning_qualified"] is False


def test_retained_prefix_does_not_make_start_or_end_convolution_support_complete():
    first, last = plan(0), plan(49)
    assert first["zero_slots_required"] == 4 and last["zero_slots_required"] == 4
    assert not first["nominal_convolution_windows_inside_source"]
    assert not last["nominal_convolution_windows_inside_source"]
    near = plan(0, "1/25")
    assert near["zero_slots_required"] == 2


def test_partial_final_tick_is_explicitly_omitted_and_never_clamped_to_padding():
    p = plan(49, samples=32123)
    hidden = np.ones((1500, 5, 384), np.float32)
    hidden[100:] = 999
    result = select_chunk(hidden, p).reshape(10, 5, 384)
    assert p["retained_tick_prefix"] == 100
    assert np.max(result) == 1 and np.count_nonzero(result[-4:]) == 0


@pytest.mark.parametrize("t", ["-1/1000", "2"])
def test_out_of_source_picture_is_unavailable_and_cannot_supply_zero_evidence(t):
    p = plan(0, t)
    assert p["source_sample_status"] == "unavailable"
    with pytest.raises(ValueError, match="Unavailable source"):
        select_chunk(np.ones((100, 5, 384), np.float32), p)


@pytest.mark.parametrize("samples", [True, 639, 480001])
def test_unqualified_waveform_extent_is_rejected(samples):
    with pytest.raises(ValueError):
        plan(0, samples=samples)


@pytest.mark.parametrize("value", [True, 0.02, float("nan")])
def test_binary_float_and_boolean_are_not_exact_source_times(value):
    with pytest.raises(ValueError):
        exact(value)


@pytest.mark.parametrize("problem", ["shape", "short", "dtype", "nan", "tick", "mask"])
def test_invalid_array_or_changed_clock_mapping_does_not_silently_supply_conditioning(problem):
    p = plan(10)
    before = deepcopy(p)
    hidden = np.ones((100, 5, 384), np.float32)
    if problem == "shape":
        hidden = hidden[:, :4]
    elif problem == "short":
        hidden = hidden[:99]
    elif problem == "dtype":
        hidden = hidden.astype(np.float64)
    elif problem == "nan":
        hidden[0, 0, 0] = np.nan
    elif problem == "tick":
        p["tick_indices"][0] += 1
    else:
        p["retained_tick_slots"][0] = False
    with pytest.raises(ValueError):
        select_chunk(hidden, p)
    assert before["generator_conditioning_qualified"] is False


def test_negative_clock_allowance_is_rejected():
    with pytest.raises(ValueError):
        plan(0, allowance="-1/1000")
