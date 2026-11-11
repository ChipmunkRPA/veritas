import pytest

from app.bench.paragraph_context import responses, selected_span

POINTS = {"isolated": 0.9, "two_sided": 3.0}


def test_paragraph_margin_uses_stronger_of_two_neighbors():
    rows = responses([(0, 150, 30, .01), (151, 301, 25, .95), (302, 452, 20, .60),
                      (453, 650, 30, .99), (651, 810, 20, .10)])
    assert rows[0]["two_sided"] == pytest.approx(2.5389738711)
    assert selected_span(rows, POINTS, "both") == (453, 650)
    assert selected_span(rows, POINTS, "isolated") == (453, 650)


def test_missing_neighbors_and_tied_qualifying_spans_have_separate_states():
    assert responses([(0, 120, 20, .01), (121, 230, 19, .99), (231, 310, 25, .01)]) == []
    rows = responses([(i * 100, i * 100 + 99, 20, score)
                      for i, score in enumerate((.01, .99, .01, .99, .01))])
    assert len(rows) == 3
    assert selected_span(rows, POINTS, "both") is None


def test_equal_to_the_point_is_below_the_strict_rule():
    rows = responses([(0, 100, 20, 0.0), (101, 200, 20, 1.0), (201, 300, 20, 0.0)])
    assert selected_span(rows, {"isolated": 1.0, "two_sided": 2.0}, "both") is None
    assert selected_span(rows, {"isolated": 0.9, "two_sided": rows[0]["two_sided"]}, "both") is None


@pytest.mark.parametrize("invalid", [(100, 150, 20, float("nan")), (100, 150, 20, 1.01), (1, 150, 20, .6)])
def test_invalid_or_overlapping_source_measurements_are_rejected(invalid):
    with pytest.raises(ValueError, match="ordered source spans"):
        responses([(0, 99, 20, .5), invalid])
