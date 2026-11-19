"""Recount the source-disjoint exact-30 common-detail comparison from published scalars."""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

METHODS = ("detail_pair", "guarded_detail_pair", "matched_detail")
ROLES = ("untouched", "gaussian", "disc", "swap", "swap_plus_blur", "hard_cut")
POINT = 0.10
GATE = {
    "min_added_exact_compounds": 2,
    "min_gain_groups": 2,
    "max_new_ordinary": 1,
    "max_wrong_location": 1,
    "min_ordinary_usable": 6,
    "require_existing_preserved": True,
}


def eligible(x):
    return (
        type(x) is list
        and len(x) == 2
        and all(type(v) is int for v in x)
        and 6 <= x[0] < x[1] <= 114
        and x[1] - x[0] in (3, 4)
    )


def recount(report):
    assert report["schema"] == "chipmunk.native30-common-detail/1"
    assert report["gate"] == GATE and report["pilot_unmodified_point"] == POINT
    assert report["files"] == 24 and report["fresh_capture_count"] == 4
    expected = {(g, role) for g in range(1, 5) for role in ROLES}
    result = {m: Counter() for m in METHODS}
    scopes = {r: Counter() for r in ROLES}
    gain_sources = defaultdict(set)
    for case in report["observations"]:
        assert set(case) == {
            "group",
            "role",
            "authored",
            "truth",
            "stock",
            "stock_range",
            "arms",
            "threshold",
        }
        g, role = case["group"], case["role"]
        expected.remove((g, role))
        assert case["threshold"] == POINT and set(case["arms"]) == set(METHODS)
        truth, authored = case["truth"], case["authored"]
        positive = role in ("swap", "swap_plus_blur")
        assert truth == (authored if positive else None)
        if role == "untouched":
            assert authored is None
        elif role == "hard_cut":
            assert authored[-1] == 120 and type(authored[0]) is int and 6 <= authored[0] <= 114
        else:
            assert eligible(authored)
        status = case["stock"]
        assert status in {"localized", "no_range", "unassessed", "held_preexisting_face_return"}
        assert eligible(case["stock_range"]) if status == "localized" else case["stock_range"] is None
        stage = scopes[role]
        stage["files"] += 1
        stage["stock_exact"] += positive and case["stock_range"] == truth
        stage["stock_localized"] += status == "localized"
        stage["stock_unassessed"] += status == "unassessed"
        for arm, entry in case["arms"].items():
            assert set(entry) == {"evaluated", "new"}
            views = entry["evaluated"]
            assert len(views) <= 8 and all(eligible(v) for v in views)
            counts = result[arm]
            counts["view_coverage"] += bool(views)
            counts["usable_ordinary"] += not positive and bool(views)
            counts["geometry_true_pairs"] += positive and truth in views
            stage[arm + "_usable"] += bool(views)
            hit = entry["new"]
            if not hit:
                continue
            assert status not in {"localized", "held_preexisting_face_return"}
            assert set(hit) == {"range", "response_hex", "witness_indexes_and_pts"}
            assert hit["range"] in views
            point = float.fromhex(hit["response_hex"])
            assert math.isfinite(point) and point > POINT
            a, b = hit["range"]
            assert hit["witness_indexes_and_pts"] == [[i, i * 1000] for i in (a - 1, a + (b - a) // 2, b)]
            counts["native_witnesses"] += 3
            counts["new_ordinary"] += not positive
            counts["wrong_positive"] += positive and hit["range"] != truth
            counts["new_compound_exact"] += role == "swap_plus_blur" and hit["range"] == truth
            counts["new_exact_clean"] += role == "swap" and hit["range"] == truth
            stage[arm + "_new"] += 1
            stage[arm + "_exact"] += hit["range"] == authored
            if role == "swap_plus_blur" and hit["range"] == truth:
                gain_sources[arm].add(g)
    assert not expected and result == report["arms"] and scopes == report["by_role"]
    passes = {}
    for arm, counts in result.items():
        assert len(gain_sources[arm]) == report["exact_compound_gain_group_count"][arm]
        passes[arm] = (
            counts["new_compound_exact"] >= GATE["min_added_exact_compounds"]
            and len(gain_sources[arm]) >= GATE["min_gain_groups"]
            and counts["new_ordinary"] <= GATE["max_new_ordinary"]
            and counts["wrong_positive"] <= GATE["max_wrong_location"]
            and counts["usable_ordinary"] >= GATE["min_ordinary_usable"]
        )
    assert passes == report["passes"]
    return {"passes": passes, "arms": result, "cases": 24, "sources": 4}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_common_detail.py ANONYMIZED-RECEIPT.json")
    print(json.dumps(recount(json.loads(Path(sys.argv[1]).read_bytes())), indent=2, sort_keys=True))
