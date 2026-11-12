# SPDX-License-Identifier: Apache-2.0
import pytest

from scripts import video_short_return_secondary_transfer as secondary


def interval(score, start="1", end="11/8"):
    return {"score": score, "start": start, "end": end}


def test_distinct_candidate_can_coexist_with_incumbent():
    current = interval(.12, "2", "3")
    candidate = interval(.3)
    observed, added, duplicate = secondary.choose({"current": current, "rank": candidate})
    assert observed is current and added is candidate and not duplicate


def test_exactly_half_overlap_is_deduplicated():
    current = interval(.12, "1", "7/4")
    assert secondary.choose({"current": current, "rank": interval(.3)}) == (current, None, True)


def test_just_below_half_overlap_is_retained():
    current = interval(.12, "1", "44/25")
    candidate = interval(.3)
    assert secondary.choose({"current": current, "rank": candidate}) == (current, candidate, False)


def test_measured_current_below_threshold_does_not_deduplicate():
    candidate = interval(.3)
    assert secondary.choose({"current": interval(.1), "rank": candidate}) == (None, candidate, False)


@pytest.mark.parametrize("candidate", [None, interval(.1), interval(.3, end="3/2"), interval(.3, end="1")])
def test_invalid_rank_keeps_incumbent_evidence(candidate):
    current = interval(.12)
    assert secondary.choose({"current": current, "rank": candidate}) == (current, None, False)
