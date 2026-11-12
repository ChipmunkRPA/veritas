# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction

import pytest

from scripts import video_short_return_margin_transfer as transfer


def interval(score, start="1", end="11/8"):
    return {"score": score, "start": start, "end": end}


def test_margin_transfer_preserves_current_result_object():
    incumbent = interval(.12, "1/2", "2")
    candidate = interval(.40)
    result, decision = transfer.route({"current": incumbent, "rank": candidate})
    assert result is incumbent and decision == "incumbent"


def test_margin_and_short_duration_must_both_pass():
    candidate = interval(.100001, end="149/100")
    result, decision = transfer.route({"current": interval(0.), "rank": candidate})
    assert result is candidate and decision == "additive"


@pytest.mark.parametrize("candidate", [interval(.10), interval(.05), interval(.40, end="3/2"), interval(.40, end="1")])
def test_margin_transfer_keeps_measured_nonqualifying_ranges_unflagged(candidate):
    assert transfer.route({"current": None, "rank": candidate}) == (None, "no_flag")


def test_missing_inputs_remain_unassessed():
    assert transfer.route({"current": None, "rank": None}) == (None, "unassessed")


def test_shortest_phases_map_to_exact_native_boundaries():
    times = [Fraction(i, 30) for i in range(180)]
    intervals = [transfer.trial.prior.comparison.exact_span(times, center, duration)
                 for center, duration in transfer.PHASES if duration == Fraction(3, 8)]
    assert intervals == [(85, 96), (82, 94), (86, 97)]
    _, dense, _ = transfer.trial.schedules(times)
    assert [tuple(i for i in dense if a <= i < b) for a, b in intervals] == [(86, 90, 94), (82, 86, 90), (86, 90, 94)]
