"""Recount the fixed local-softening rule from anonymous scalar evidence."""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLOCKS = {"1/30000": 1001, "1/90000": 3003}
ROLES = ("untouched", "gaussian", "disc", "swap", "swap_plus_blur")
STATUS = {
    "localized",
    "no_range",
    "unassessed",
    "preexisting_short",
    "held_preexisting_face_return",
    "held_separate_shot_boundary",
}
READINGS = {
    "min_reduction",
    "min_loss",
    "min_input_detail",
    "min_fit_sigma",
    "max_blur_error",
    "max_coarse_error",
    "flank_coarse_error",
}
POINTS = {
    "min_reduction": 0.32,
    "min_loss": 0.50,
    "min_input_detail": 2.5,
    "min_fit_sigma": 1.25,
    "max_blur_error": 7.5,
    "max_coarse_error": 7.0,
    "flank_coarse_floor": 4.0,
    "flank_multiplier": 1.9,
    "flank_offset": 1.0,
}
GATE = {
    "min_raw_exact_swaps": 3,
    "min_raw_exact_compounds": 2,
    "min_ordinary_blur_alerts": 2,
    "min_suppressed_ordinary_blur_alerts": 2,
    "min_suppressed_source_groups": 2,
    "max_lost_exact_swaps": 0,
    "max_lost_exact_compounds": 0,
    "max_unrelated_suppression": 0,
}


def explains(row):
    assert set(row) == READINGS
    x = {k: float.fromhex(v) for k, v in row.items()}
    assert all(math.isfinite(v) for v in x.values())
    coarse_limit = min(
        POINTS["max_coarse_error"],
        max(
            POINTS["flank_coarse_floor"],
            POINTS["flank_multiplier"] * x["flank_coarse_error"] + POINTS["flank_offset"],
        ),
    )
    return (
        all(
            x[key] >= POINTS[key]
            for key in ("min_reduction", "min_loss", "min_input_detail", "min_fit_sigma")
        )
        and x["max_blur_error"] <= POINTS["max_blur_error"]
        and x["max_coarse_error"] <= coarse_limit
    )


def review(data):
    assert data["schema"] == "chipmunk.video-local-short-blur-hold/1"
    assert data["thresholds"] == POINTS and data["gate_requirements"] == GATE
    expected = {(g, r) for g in range(1, 7) for r in (ROLES if g <= 4 else ROLES[:3])}
    dispositions = defaultdict(Counter)
    withheld_groups = set()
    planned_treatments = {}
    anchor_count = 0
    for case in data["cases"]:
        group, role = case["group"], case["role"]
        expected.remove((group, role))
        clock = case["time_base"]
        assert clock == ("1/30000", "1/90000")[int((group + ROLES.index(role)) % 2 == 0)]
        assert case["camera_profile"] == ("rtfs_native_ntsc" if group <= 4 else "vcd2_30fps_one_to_one_ntsc")
        interval = case["authored"]
        if role == "untouched":
            assert interval is None
        else:
            a, b = interval
            assert 0 < a < b < 120 and b - a in (3, 4)
            assert planned_treatments.setdefault(group, interval) == interval
        state = case["raw_status"]
        assert state in STATUS
        selected = case["candidate_range"]
        present = state == "localized"
        assert present == (selected is not None)
        suppressed = False
        if present:
            a, b = selected
            assert 0 < a < b < 120 and b - a in (3, 4)
            expected_indices = (a - 1, a + (b - a) // 2, b)
            assert case["bound_frame_pts"] == [[i, i * CLOCKS[clock]] for i in expected_indices]
            suppressed = explains(case["scalar_hex"])
            anchor_count += 3
        else:
            assert "scalar_hex" not in case and "bound_frame_pts" not in case
        exact = present and interval == selected
        false_suppression = suppressed and role in ROLES[:3] and not exact
        if suppressed and exact and role in ("gaussian", "disc"):
            withheld_groups.add(group)
        for bucket in (role, "all"):
            c = dispositions[bucket]
            c["cases"] += 1
            c["localized"] += present
            c["unassessed"] += state == "unassessed"
            c["exact_raw"] += exact
            c["exact_remaining"] += exact and not suppressed
            c["guard_applied"] += suppressed
            c["unrelated_suppression"] += false_suppression
    assert not expected and len(planned_treatments) == 6
    assert dispositions == data["recount"]
    assert anchor_count == data["native_evidence_crops_replayed"]
    assert data["source_mapping_entries_rechecked"] == 720 + 2 * sum(
        planned_treatments[group][1] - planned_treatments[group][0] for group in range(1, 5)
    )
    blur = dispositions["gaussian"] + dispositions["disc"]
    swaps = dispositions["swap"]
    both = dispositions["swap_plus_blur"]
    passed = (
        swaps["exact_raw"] >= GATE["min_raw_exact_swaps"]
        and both["exact_raw"] >= GATE["min_raw_exact_compounds"]
        and blur["exact_raw"] >= GATE["min_ordinary_blur_alerts"]
        and blur["exact_raw"] - blur["exact_remaining"] >= GATE["min_suppressed_ordinary_blur_alerts"]
        and len(withheld_groups) >= GATE["min_suppressed_source_groups"]
        and swaps["guard_applied"] <= GATE["max_lost_exact_swaps"]
        and both["guard_applied"] <= GATE["max_lost_exact_compounds"]
        and dispositions["all"]["unrelated_suppression"] <= GATE["max_unrelated_suppression"]
    )
    assert passed == data["gate_pass"] and len(withheld_groups) == data["ordinary_blur_groups_suppressed"]
    return {
        "cases": dispositions,
        "gate_pass": passed,
        "crops": anchor_count,
        "ordinary_blur_groups_suppressed": len(withheld_groups),
        "comment": "Recounts released held scalars and frame positions. Source-video pixels and independent native decode stay private.",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_face_short_local_blur.py RECEIPT.json")
    print(json.dumps(review(json.loads(Path(sys.argv[1]).read_bytes())), indent=2, sort_keys=True))
