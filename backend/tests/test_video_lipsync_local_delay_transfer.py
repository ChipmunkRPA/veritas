"""Bounded timing-bracket and exact intervention tests."""

import numpy as np

from scripts import video_lipsync_local_delay_transfer as transfer
from scripts import video_lipsync_short_bracket as bracket


def test_lag_requires_flank_agreement_and_new_middle_peak():
    before = np.full(13, .05)
    after = np.full(13, .10)
    middle = np.full(13, .05)
    before[6], after[6], middle[11] = .60, .68, .72
    measured = bracket._choice((before, middle, after))
    assert measured is not None and measured[1:] == (5, 0, 5)
    assert bracket._choice((before, middle, middle)) is None
    assert bracket._choice((before, before, after)) is None


def test_shifted_control_and_local_interval_are_sample_exact():
    rng = np.random.default_rng(47)
    source = rng.integers(-16000, 16000, 72 * 640, dtype=np.int16)
    rows = {label: (wave, bounds, lag) for label, wave, bounds, lag in transfer.variants(source, 72)}
    for label, sign in (("late", 1), ("early", -1)):
        local, bounds, lag = rows["local_" + label]
        delay = sign * transfer.DELTA_TICKS * 640
        assert lag == sign * transfer.DELTA_TICKS
        assert rows["global_" + label][1:] == (None, None)
        a, b = (t * 640 for t in bounds)
        fade = transfer.FADE_SAMPLES
        assert np.array_equal(local[:a], source[:a]) and np.array_equal(local[b:], source[b:])
        assert np.array_equal(local[a+fade:b-fade], source[a+fade-delay:b-fade-delay])


def test_time_overlap_and_delay_direction_are_distinct():
    truth = [33, 49]
    assert not transfer.correct({"extent_ticks": [34, 50], "relative_lag_ticks": -6}, truth, 5)
    assert transfer.correct({"extent_ticks": [34, 50], "relative_lag_ticks": 5}, truth, 5)
