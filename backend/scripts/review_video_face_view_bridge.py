"""Recount anonymous native face view recovery evidence."""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("geometry", "detail_pair", "guarded_detail_pair")
TICKS = {"1/30000": 1001, "1/90000": 3003}
POINT = 0.10
ROLES = {"untouched", "gaussian", "disc", "swap", "swap_plus_blur"}
GATE = {
    "min_exact_compound_added": 2,
    "min_compound_source_groups": 2,
    "max_wrong_positive_ranges": 1,
    "max_new_ordinary_alerts": 1,
    "min_measured_ordinary_ranges": 8,
    "max_lost_stock_ranges": 0,
}


def check_interval(v):
    return (
        isinstance(v, list)
        and len(v) == 2
        and all(type(x) is int for x in v)
        and (6 <= v[0] < v[1] <= 114 and v[1] - v[0] in (3, 4))
    )


def audit(report):
    assert report["schema"] == "chipmunk.native-face-view-bridge-result/1"
    assert report["point"] == POINT and report["gate"] == GATE
    assert report["files"] == 26 and report["source_groups"] == 6
    expected = {
        (group, role)
        for group in range(1, 7)
        for role in (ROLES if group < 5 else ROLES - {"swap", "swap_plus_blur"})
    }
    summary = defaultdict(Counter)
    sources = defaultdict(set)
    for row in report["observations"]:
        assert set(row) == {"group", "role", "time_base", "truth", "authored", "old", "view_pairs", "results"}
        group, role = row["group"], row["role"]
        expected.remove((group, role))
        authored, truth = row["authored"], row["truth"]
        assert row["time_base"] in TICKS
        assert row["old"] in {
            "localized",
            "no_range",
            "unassessed",
            "preexisting_short",
            "held_separate_shot_boundary",
            "held_preexisting_face_return",
        }
        assert check_interval(authored) if authored is not None else role == "untouched"
        positive = role in ("swap", "swap_plus_blur")
        assert truth == (authored if positive else None)
        assert set(row["results"]) == set(row["view_pairs"]) == set(ARMS)
        assert row["view_pairs"]["detail_pair"] == row["view_pairs"]["guarded_detail_pair"]
        for arm in ARMS:
            votes = summary[arm]
            pairs = row["view_pairs"][arm]
            assert len(pairs) <= 8 if arm != "geometry" else len(pairs) <= 32
            assert all(check_interval(x) for x in pairs)
            votes["comparable_ordinary"] += not positive and bool(pairs)
            votes["matched_compound_views"] += role == "swap_plus_blur" and authored in pairs
            outcome = row["results"][arm]
            if outcome is None:
                continue
            assert set(outcome) == {
                "range",
                "exact_response_hex",
                "native_witness_frame_and_pts",
                "explained_by_softening",
            }
            assert row["old"] != "localized"
            prob = float.fromhex(outcome["exact_response_hex"])
            assert math.isfinite(prob) and prob > POINT
            assert type(outcome["explained_by_softening"]) is bool
            if arm == "guarded_detail_pair":
                assert outcome["explained_by_softening"] is False
            band = outcome["range"]
            assert band in pairs
            a, b = band
            tick = TICKS[row["time_base"]]
            assert outcome["native_witness_frame_and_pts"] == [
                [i, i * tick] for i in (a - 1, a + (b - a) // 2, b)
            ]
            votes["ordinary_added"] += not positive
            votes["wrong_positive_region"] += positive and band != truth
            votes["native_witnesses"] += 3
            if role == "swap_plus_blur" and band == truth:
                votes["new_exact_compound"] += 1
                sources[arm].add(group)
    assert not expected and summary == report["arms"]
    passed = {}
    for arm in ARMS:
        c = summary[arm]
        assert len(sources[arm]) == report["source_groups_with_exact_compound"][arm]
        passed[arm] = (
            c["new_exact_compound"] >= GATE["min_exact_compound_added"]
            and len(sources[arm]) >= GATE["min_compound_source_groups"]
            and c["wrong_positive_region"] <= GATE["max_wrong_positive_ranges"]
            and c["ordinary_added"] <= GATE["max_new_ordinary_alerts"]
            and c["comparable_ordinary"] >= GATE["min_measured_ordinary_ranges"]
        )
    assert passed == report["pass_by_arm"]
    return {"source_groups": 6, "files": 26, "arms": summary, "passes": passed}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_face_view_bridge.py RECEIPT.json")
    print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_bytes())), sort_keys=True, indent=2))
