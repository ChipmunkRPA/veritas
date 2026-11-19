"""Independently recount the published 720p face-transition comparison."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-face-boundary-720p/1"
DEFAULT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/video-face-boundary-720p-2026-11-18.json"
ROLES = {"original", "prefix", "suffix", "focus", "jump", "cut", "exposure"}
EDIT_ROLES = {"prefix", "suffix"}
MAX_FRAME = 119


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def frame(value: int | None) -> bool:
    return value is None or type(value) is int and 0 <= value <= MAX_FRAME


def review(data: dict) -> dict:
    require(set(data) == {"schema", "commitment", "groups", "native_replay", "separate_camera"}
            and data["schema"] == SCHEMA and re.fullmatch(r"[a-f0-9]{64}", data["commitment"]) is not None,
            "Invalid 720p study envelope")
    groups = data["groups"]
    require([row["group"] for row in groups] == list(range(1, 7)), "Unexpected source groups")
    require(Counter(group["capture_height"] for group in groups) == {720: 2, 1080: 4},
            "Original source raster mix differs")
    counts = Counter()
    useful_groups, ordinary_groups = set(), set()
    directions, renderers = Counter(), Counter()
    for group in groups:
        require(set(group) == {"group", "capture_height", "cases"} and set(group["cases"]) == ROLES,
                "Missing comparison role")
        for role, case in group["cases"].items():
            require(set(case) == {"truth", "older", "boundary", "status", "renderer"} and frame(case["truth"])
                    and frame(case["boundary"]), "Invalid case boundary")
            require(isinstance(case["older"], list) and len(case["older"]) <= 2, "Invalid older ranges")
            for interval in case["older"]:
                require(isinstance(interval, list) and len(interval) == 2 and all(value is not None and frame(value) for value in interval)
                        and interval[0] < interval[1], "Invalid older interval")
            marked = case["boundary"] is not None
            truth = case["truth"]
            state = case["status"]
            require(state in {"localized", "no_match", "unavailable"} and marked == (state == "localized"),
                    "The candidate status differs from its measured frame")
            if role in EDIT_ROLES:
                require(truth is not None and case["renderer"] in {"swap_a", "swap_b"}, "Missing edited-file truth")
                counts["edits"] += 1
                counts["new_unassessed_edits"] += int(state == "unavailable")
                useful = marked and abs(case["boundary"] - truth) <= 1
                older = any(abs(interval[1 if role == "prefix" else 0] - truth) <= 1 for interval in case["older"])
                counts["current_useful"] += int(older)
                counts["exact_new"] += int(marked and case["boundary"] == truth)
                counts["wrong_new"] += int(marked and not useful)
                counts["combined_useful"] += int(useful or older)
                if useful and not older:
                    counts["new_useful"] += 1
                    useful_groups.add(group["group"])
                    directions[role] += 1
                    renderers[case["renderer"]] += 1
                    counts["exact_net_from_native_720"] += int(case["boundary"] == truth and group["capture_height"] == 720)
            else:
                require(truth is None and case["renderer"] is None, "Ordinary case carries an edited-file label")
                counts["ordinary"] += 1
                counts["new_unassessed_ordinary"] += int(state == "unavailable")
                counts["current_ordinary_alerts"] += int(bool(case["older"]))
                if marked and not case["older"]:
                    counts["new_ordinary_alerts"] += 1
                    ordinary_groups.add(group["group"])
                counts["new_hard_cut_alerts"] += int(role == "cut" and marked)
    native = data["native_replay"]
    require(native == {"native_frames": (counts["edits"] + counts["ordinary"]) * 120, "supplied_swap_frames": 528,
                       "frame_triplet_anchors": 39}, "Unexpected original-frame coverage totals")
    cameras = data["separate_camera"]
    require(cameras == {"uploads": 3, "native_clock_frames": 360, "ordinary_alerts": 0}, "Unexpected camera check")
    gate = (counts["new_useful"] >= 5 and len(useful_groups) >= 4
            and counts["exact_net_from_native_720"] >= 2 and counts["new_ordinary_alerts"] <= 3 and len(ordinary_groups) <= 2
            and counts["new_hard_cut_alerts"] == 0 and counts["wrong_new"] <= 1
            and all(directions[role] >= 2 for role in EDIT_ROLES)
            and all(renderers[name] >= 2 for name in ("swap_a", "swap_b")))
    return {**counts, "source_groups_with_new_locations": len(useful_groups),
            "source_groups_with_new_ordinary_alerts": len(ordinary_groups),
            "by_edit_direction": dict(sorted(directions.items())), "by_renderer": dict(sorted(renderers.items())),
            "frozen_gate_passed": gate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    with args.receipt.open(encoding="utf-8") as stream:
        print(json.dumps(review(json.load(stream)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
