"""Recount the expanded native-30 common-detail study from anonymous scalars."""

import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ARMS = ("detail_pair", "guarded_detail_pair", "matched_detail")
POSITIVE = {"swap", "swap_plus_blur"}
ROLES = {"gaussian", "disc", "swap", "swap_plus_blur", "untouched", "hard_cut"}
CASE_KEYS = {
    "treatment",
    "group",
    "role",
    "position",
    "blur_family",
    "blur_strength",
    "authored",
    "truth",
    "stock_status",
    "stock_ranges",
    "candidates",
}
POINT = 0.10
GATE = {
    "min_added_exact_compounds": 2,
    "min_gain_groups": 2,
    "max_new_ordinary": 1,
    "max_wrong_location": 1,
    "min_ordinary_usable": 6,
    "require_existing_preserved": True,
}


def interval(pair):
    return (
        type(pair) is list
        and len(pair) == 2
        and all(type(x) is int for x in pair)
        and 6 <= pair[0] < pair[1] <= 114
        and pair[1] - pair[0] in (3, 4)
    )


def overlap(left, right):
    span = max(left[1], right[1]) - min(left[0], right[0])
    return Fraction(max(0, min(left[1], right[1]) - max(left[0], right[0])), span)


def review(receipt):
    assert receipt["schema"] == "chipmunk.native30-common-detail-expanded/1"
    assert (receipt["groups"], receipt["files"], receipt["pinned_appearance_point"]) == (6, 48, POINT)
    assert receipt["frozen_gate"] == GATE
    result = {arm: Counter() for arm in ARMS}
    roles = {role: Counter() for role in ROLES}
    strata = defaultdict(Counter)
    seen = defaultdict(set)
    matched = {}
    recovered = defaultdict(set)
    missed, missed_groups, missed_exact, missed_exact_groups = set(), set(), set(), set()
    retained = 0
    for row in receipt["observations"]:
        assert set(row) == CASE_KEYS and set(row["candidates"]) == set(ARMS)
        t, g, role, position = (row[key] for key in ("treatment", "group", "role", "position"))
        assert type(t) is type(g) is int and g == (t + 1) // 2 and g in range(1, 7)
        assert position == ("early" if t % 2 else "late") and role in ROLES
        assert role not in seen[t]
        seen[t].add(role)
        kernel, strength = row["blur_family"], row["blur_strength"]
        assert kernel in ("gaussian", "disc") and strength in (4.5, 6.0, 8.0)
        old = row["stock_ranges"]
        assert all(len(p) == 2 and all(type(i) is int for i in p) and 0 <= p[0] < p[1] < 120 for p in old)
        status = row["stock_status"]
        assert status in {"localized", "no_range", "unassessed", "held_preexisting_face_return"}
        actual, truth = row["authored"], row["truth"]
        real = role in POSITIVE
        assert truth == (actual if real else None)
        if role == "untouched":
            assert actual is None and position == "early"
        elif role == "hard_cut":
            assert 61 <= actual[0] <= 78 and actual[1] == 120 and position == "early"
        else:
            assert role in {kernel, *POSITIVE} and interval(actual)
            span = (61, 78) if position == "early" else (91, 105)
            assert span[0] <= actual[0] <= span[1]
            assert matched.setdefault(t, (actual, kernel, strength)) == (actual, kernel, strength)
        old_exact = real and truth in old
        old_useful = real and any(overlap(truth, found) >= Fraction(1, 2) for found in old)
        if role == "swap_plus_blur":
            if not old_useful:
                missed.add(t)
                missed_groups.add(g)
            if not old_exact:
                missed_exact.add(t)
                missed_exact_groups.add(g)
        retained += bool(old)
        sc = roles[role]
        groups = (sc, strata[position], strata[kernel])
        for summary in groups:
            summary["files"] += 1
            summary["stock_exact"] += old_exact
            summary["stock_useful"] += old_useful
            summary["stock_ordinary"] += not real and bool(old)
            summary["stock_unassessed"] += status == "unassessed"
        for arm, candidate in row["candidates"].items():
            assert set(candidate) == {"usable_ranges", "selected"}
            views = candidate["usable_ranges"]
            assert len(views) <= 8 and all(interval(v) for v in views)
            score = result[arm]
            score["usable_views"] += bool(views)
            score["usable_ordinary"] += not real and bool(views)
            score["at_truth_views"] += real and truth in views
            sc[arm + "_usable"] += bool(views)
            outcome = candidate["selected"]
            if outcome is None:
                continue
            assert status not in {"localized", "held_preexisting_face_return"}
            assert set(outcome) == {"interval", "appearance_hex", "frame_anchors"}
            selected = outcome["interval"]
            assert selected in views
            point = float.fromhex(outcome["appearance_hex"])
            assert math.isfinite(point) and point > POINT
            start, end = selected
            assert outcome["frame_anchors"] == [
                [i, i * 1000] for i in (start - 1, start + (end - start) // 2, end)
            ]
            added_exact = real and selected == truth and not old_exact
            added_useful = real and overlap(selected, truth) >= Fraction(1, 2) and not old_useful
            score["new_ordinary"] += not real
            score["new_exact"] += added_exact
            score["new_useful"] += added_useful
            score["new_exact_compound"] += added_exact and role == "swap_plus_blur"
            score["wrong_positive"] += real and overlap(selected, truth) < Fraction(1, 2)
            score["native_witnesses"] += 3
            if added_exact and role == "swap_plus_blur":
                recovered[arm].add(g)
            for summary in groups:
                summary[arm + "_added_exact"] += added_exact
                summary[arm + "_new_ordinary"] += not real
    assert len(seen) == len(matched) == 12
    for treatment, (_, kernel, _) in matched.items():
        expected = {*POSITIVE, kernel}
        if treatment % 2:
            expected.update(("untouched", "hard_cut"))
        assert seen[treatment] == expected
    assert Counter(v[0][1] - v[0][0] for v in matched.values()) == {3: 6, 4: 6}
    assert Counter(v[1] for v in matched.values()) == {"gaussian": 6, "disc": 6}
    assert Counter(v[2] for v in matched.values()) == {4.5: 4, 6.0: 4, 8.0: 4}
    assert retained == receipt["incumbent_located_cases_preserved"]
    assert result == receipt["arms"] and roles == receipt["roles"] and strata == receipt["strata"]
    opportunities = {
        "no_stock_useful": len(missed),
        "no_stock_useful_in_groups": len(missed_groups),
        "no_stock_exact": len(missed_exact),
        "no_stock_exact_in_groups": len(missed_exact_groups),
    }
    assert opportunities == receipt["usable_compound_opportunities"]
    verdict = {}
    for arm, v in result.items():
        assert len(recovered[arm]) == receipt["distinct_gain_groups"][arm]
        verdict[arm] = (
            v["new_exact_compound"] >= GATE["min_added_exact_compounds"]
            and len(recovered[arm]) >= GATE["min_gain_groups"]
            and v["new_ordinary"] <= GATE["max_new_ordinary"]
            and v["wrong_positive"] <= GATE["max_wrong_location"]
            and v["usable_ordinary"] >= GATE["min_ordinary_usable"]
        )
    assert verdict == receipt["passes"]
    return {"passes": verdict, "missed_compound_opportunities": opportunities, "arms": result, "roles": roles}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_expanded_common_detail.py ANONYMOUS-RECEIPT.json")
    print(json.dumps(review(json.loads(Path(sys.argv[1]).read_bytes())), sort_keys=True, indent=2))
