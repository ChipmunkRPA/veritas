"""Renderer replay clocks retain exact PTS despite rounded display strings."""
from fractions import Fraction

import pytest

from scripts.replay_video_color_v14 import checked_time, parse_clock


def test_integer_pts_avoids_display_rounding_without_relaxing_original_tolerance():
    stderr = ('config in time_base: 1/30000, frame_rate: 30000/1001\n'
              'n: 0 pts: 37037 pts_time:1.23457 fmt:rgb24\n')
    base, pts = parse_clock(stderr, 1)
    # The display string is >3microseconds away; recorded six-decimal source
    # seconds are within one third of a microsecond of the exact source time.
    assert abs(Fraction('1.23457') - pts[0] * base) > Fraction(11, 10_000_000)
    error = checked_time(pts[0], base, {'best_effort_timestamp': 37037}, base, 1.234567)
    assert error == pytest.approx(1 / 3_000_000)


def test_equivalent_tick_bases_are_compared_as_exact_times():
    assert checked_time(123, Fraction(1, 1000), {'best_effort_timestamp': 3690},
                        Fraction(1, 30000), .123) == 0


@pytest.mark.parametrize('fault', ['wrong_tick', 'wrong_recorded_seconds'])
def test_actual_source_clock_mismatch_is_refused(fault):
    timestamp, recorded = (37038, 1.234567) if fault == 'wrong_tick' else (37037, 1.23457)
    with pytest.raises(ValueError):
        checked_time(timestamp, Fraction(1, 30000), {'best_effort_timestamp': 37037},
                     Fraction(1, 30000), recorded)


@pytest.mark.parametrize('frames', ['', 'n: 1 pts: 37037 pts_time:1.23457',
                                  'n: 0 pts: 37037 pts_time:1.23457\nn: 0 pts: 37037 pts_time:1.23457'])
def test_missing_reordered_or_duplicate_output_clock_is_refused(frames):
    with pytest.raises(ValueError):
        parse_clock('config in time_base: 1/30000\n' + frames, 1)
