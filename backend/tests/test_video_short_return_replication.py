import pytest

from scripts import video_short_return_replication as replication


def range_record(score, start="1", end="11/8"):
    return {"score": score, "start": start, "end": end}


def test_additive_cue_preserves_incumbent_exact_range():
    incumbent = range_record(0.3, "8/5", "11/5")
    candidate = range_record(0.8)
    result, path = replication.route({"current": incumbent, "rank": candidate})
    assert result is incumbent and path == "incumbent"


def test_positive_short_range_can_add_without_a_current_flag():
    candidate = range_record(0.05)
    result, path = replication.route({"current": range_record(0.), "rank": candidate})
    assert result is candidate and path == "additive"


@pytest.mark.parametrize("candidate", [range_record(0.), range_record(-0.01), range_record(0.8, "1", "3/2"),
                                      range_record(0.8, "1", "7/4"), range_record(0.8, "1", "1")])
def test_zero_margin_long_or_invalid_ranges_do_not_add(candidate):
    assert replication.route({"current": None, "rank": candidate}) == (None, "no_flag")


def test_missing_current_and_candidate_results_remain_unassessed():
    assert replication.route({"current": None, "rank": None}) == (None, "unassessed")
