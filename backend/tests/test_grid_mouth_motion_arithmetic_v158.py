from fractions import Fraction

import numpy as np
import pytest

from scripts import grid_mouth_motion_arithmetic_v158 as motion


def time(i):
    return Fraction(i, 25)


TIMES = [time(i) for i in range(27, 47)]


def reference(period=None):
    rng = np.random.default_rng(158)
    values = rng.uniform(0.1, 0.75, size=(period or 75, 4))
    return {time(i): motion.MouthShape(tuple(values[i % len(values)]), 80.0) for i in range(75)}


def shift(ref, ticks):
    return {time(i): ref.get(time(i + ticks)) for i in range(25, 49)}


@pytest.mark.parametrize(
    ("lag", "expected"), [(0, "source_time_compatible"), (-8, "earlier_compatible"), (12, "search_boundary")]
)
def test_exact_shift_recovers_visible_lag_without_time_interpolation(lag, expected):
    ref = reference()
    result = motion.measure(shift(ref, lag), ref, TIMES)
    assert result["status"] == expected and result["near_best_lags"] == [lag]
    assert result["curve"]["supported_positions"] == 20
    assert result["curve"]["adjacent_pairs"] == 19
    assert set(result["curve"]["lags"]) == set(range(-12, 13))
    assert result["curve"]["lags"][lag] == {
        "centered_shape_rms": 0.0,
        "absolute_shape_rms": 0.0,
        "velocity_rms_per_second": 0.0,
    }


def test_periodic_near_ties_do_not_choose_source_time():
    ref = reference(period=7)
    result = motion.measure(shift(ref, 0), ref, TIMES)
    assert result["status"] == "ambiguous"
    assert result["near_best_lags"] == [-7, 0, 7]


def test_missing_outer_reference_preserves_separate_target_support():
    ref = reference()
    var = shift(ref, 0)
    for i in range(15, 20):
        ref[time(i)] = None
    result = motion.measure(var, ref, TIMES)
    assert result["status"] == "unavailable"
    assert result["curve"]["supported_positions"] == 15
    assert result["target_only_contrast"]["supported_positions"] == 19
    assert result["target_only_contrast"]["measured"]


def test_sixteen_observations_do_not_pass_with_only_eleven_native_pairs():
    ref = reference()
    var = shift(ref, 0)
    for i in (29, 33, 37, 41):
        var[time(i)] = None
    result = motion.measure(var, ref, TIMES)
    assert result["status"] == "unavailable"
    assert result["curve"] == {"supported_positions": 16, "adjacent_pairs": 11, "measured": False}


def test_static_output_and_weak_reference_are_separate():
    point = motion.MouthShape((0.05, 0.05, 0.05, 0.3), 80.0)
    ref = reference()
    static = dict.fromkeys(ref, point)
    assert motion.measure(static, ref, TIMES)["status"] == "static_output"
    assert motion.measure(ref, static, TIMES)["status"] == "weak_reference"


def test_matched_preference_recomputes_shared_support_without_gap_velocity():
    ref = reference()
    early, source = shift(ref, -8), shift(ref, 0)
    early[time(28)] = early[time(29)] = None
    source[time(30)] = source[time(31)] = None
    assert motion.measure(early, ref, TIMES)["curve"]["supported_positions"] == 18
    result = motion.paired_preference(early, source, ref, TIMES)
    assert result["supported_positions"] == 16 and result["adjacent_pairs"] == 14
    assert result["status"] == "gain"
    reverse = motion.paired_preference(source, early, ref, TIMES)
    assert reverse["status"] == "loss"
    assert result["earlier_preference_difference"] == -reverse["earlier_preference_difference"]
    assert motion.paired_preference(source, source, ref, TIMES)["status"] == "within_dead_band"


def test_observed_boundary_velocity_uses_one_native_cell():
    a, b = motion.MouthShape((0, 0, 0, 0.25), 20), motion.MouthShape((0, 0, 0, 0.5), 20)
    assert motion.boundary_velocity({time(26): a, time(27): b}, time(26), time(27)) == 6.25
    assert motion.boundary_velocity({time(26): a}, time(26), time(27)) is None
    with pytest.raises(ValueError, match="adjacent"):
        motion.boundary_velocity({}, time(26), time(28))


def test_readout_refuses_approximate_or_discontinuous_clock():
    ref = reference()
    with pytest.raises(ValueError, match="rational"):
        motion.measure(ref, ref, [float(t) for t in TIMES])
    with pytest.raises(ValueError, match="25 Hz"):
        motion.measure(ref, ref, [*TIMES[:-1], time(47)])
