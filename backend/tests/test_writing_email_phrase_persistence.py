from __future__ import annotations

import pytest

from scripts.writing_email_phrase_persistence import readout, span_values, thresholds


def test_two_scale_read_reduces_single_spike_but_keeps_sustained_phrase():
    words = "The team sent a short note before noon"
    bounds = [(0, 3), (4, 8), (9, 13), (14, 15), (16, 21), (22, 26), (27, 33), (34, 38)]
    peak, = span_values([(*bound, 8. if index == 3 else 0.) for index, bound in enumerate(bounds)], words, 42)
    sustained, = span_values([(*bound, 3.) for bound in bounds], words, 42)
    assert peak == {"span": [42, 80], "raw_token": 8., "phrase_persistence": 1.}
    assert sustained == {"span": [42, 80], "raw_token": 3., "phrase_persistence": 3.}
    with pytest.raises(ValueError, match="incomplete native token coverage"):
        span_values([(*bound, 1.) for bound in bounds[1:]], words, 42)


def test_review_points_use_source_group_maxima_and_keep_quote_errors_separate():
    calibration = [{"group": group, "candidates": [{"raw_token": raw, "phrase_persistence": smooth}]}
                   for group, raw, smooth in (("A", 4, 3), ("A", 2, 2), ("B", 3, 1), ("C", 1, 2), ("D", 0, 0))]
    points = thresholds(calibration)
    assert points == {"raw_token": 3, "phrase_persistence": 2}
    text = "Actual email body.\n\n> Quoted words."
    case = {"case": "control", "group": "held", "kind": "human_quoted", "text": text, "ai_spans": []}
    response = {"case": "control", "roles": [{"span": [0, 18], "role": "body"}, {"span": [20, 35], "role": "quoted"}],
                "baseline": [{"spans": [[20, 35]], "raw": .6}],
                "candidates": [{"span": [0, 12], "raw_token": 3, "phrase_persistence": 2}]}
    counts = readout([case], [response], points)
    assert counts["baseline"]["human_quoted_flagged_quote"] == 1
    assert counts["baseline"]["human_quoted_flagged_body"] == 0
    assert counts["raw_token"]["human_quoted_flagged_body"] == 0
    assert counts["phrase_persistence"]["human_quoted_flagged_quote"] == 0
