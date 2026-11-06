import math

from scripts.diagnose_mouth_temporal_ranking_v11 import bound


def fixture(tied=False):
    ids = ("ordinary-a", "ordinary-b", "new-ai", "retained-ai")
    rows = [
        {
            "id": name,
            "label": int(i >= 2),
            "family": "liveportrait" if i == 2 else "musetalk",
            "edit_time_range_s": [1, 2],
        }
        for i, name in enumerate(ids)
    ]
    scores = (10 if tied else 11, 10, 12, 9)
    windows = [
        {"id": name, "windows": [{"start_s": 1, "end_s": 2, "response": score}]}
        for name, score in zip(ids, scores, strict=True)
    ]
    old = [{"id": name, "probe_windows": [{"response": 8 if i == 3 else 0}]} for i, name in enumerate(ids)]
    outcomes = [{"flagged": i == 3, "localized": i == 3} for i in range(4)]
    return {
        "panels": {"development": {"cases": rows}},
        "summary": {
            "head_sha256": "example",
            "scored_windows": windows,
            "cases": {"combined": [{"condition": "original", "flagged": False}]},
        },
        "absolute_reference": {"review_point": 7, "cases": old, "outcomes": outcomes},
    }


def test_bound_keeps_prior_locations_and_uses_inclusive_runtime_rule():
    result = bound(fixture())
    assert result["minimum_raw_point_for_at_most_one_ordinary_flag"] == math.nextafter(10, math.inf)
    assert result["ordinary_flags_at_bound"] == 1
    assert result["maximum_combined_locations"] == 2


def test_tied_controls_require_point_strictly_above_tied_margin():
    result = bound(fixture(tied=True))
    assert result["ordinary_flags_at_bound"] == 0
    assert result["maximum_combined_locations"] == 2
