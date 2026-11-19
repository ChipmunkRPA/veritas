"""Recount the anonymous HEVC short-face transfer without models or media."""

import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-short-face-hevc-25-holdout/1"
POINT = (0.20).hex()
PROFILE = {
    "frames": 120,
    "rate": "25/1",
    "time_base": "1/90000",
    "step": 3600,
    "size": [1280, 720],
}
GATE = {
    "gain": 2,
    "gain_groups": 2,
    "max_added_ordinary": 1,
    "max_added_cut": 0,
    "max_wrong": 0,
    "min_ordinary_with_views": 4,
}
ROLES = {
    "untouched",
    "cut",
    "ordinary_blur",
    "ordinary_light",
    "clean_swap",
    "swap_plus_blur",
}
EDITS = {"clean_swap", "swap_plus_blur"}
PAIR_ROLES = {"ordinary_blur", "swap_plus_blur"}


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def interval(x):
    return (
        isinstance(x, list)
        and len(x) == 2
        and all(type(i) is int for i in x)
        and 0 <= x[0] < x[1] < 120
    )


def iou(a, b):
    return Fraction(
        max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0])
    )


def review(doc):
    require(
        doc["schema"] == SCHEMA
        and doc["profile"] == PROFILE
        and doc["gate"] == GATE
        and doc["point_hex"] == POINT,
        "Frozen method differs",
    )
    require(
        doc["verified_originals"]
        == {
            "files": 40,
            "frames_rgb_pts": 4800,
            "exact_paired_sequences": 10,
            "source_ordinals": 600,
            "native_roi_observed": 5,
        },
        "Original-frame verification counts differ",
    )
    require(
        set(doc["commitments"])
        == {
            "freeze.json",
            "media.json",
            "independent-media.json",
            "scorer-lock.json",
            "measurements.jsonl",
            "result-final.json",
        }
        and all(
            isinstance(v, str)
            and len(v) == 64
            and all(c in "0123456789abcdef" for c in v)
            for v in doc["commitments"].values()
        ),
        "Expected aggregate replay commitments",
    )
    rows = defaultdict(Counter)
    roster = set()
    truths = {}
    gain_groups = set()
    anchors = bound_views = exact_cuts = 0
    for item in doc["observations"]:
        group, codec, role, staged, truth, stock, new, audit = (
            item[k]
            for k in (
                "group",
                "codec",
                "role",
                "staged_start",
                "truth",
                "incumbent",
                "new_hevc_range",
                "native_replay",
            )
        )
        key = group, codec, role
        require(
            type(group) is int
            and 1 <= group <= 5
            and codec in {"hevc", "avc"}
            and role in ROLES
            and key not in roster,
            "Repeated or unknown test role",
        )
        roster.add(key)
        if role in EDITS:
            require(
                interval(truth)
                and truth[1] - truth[0] in (3, 4)
                and staged == truth[0],
                "Edited source clocks differ",
            )
            if group in truths:
                require(truth == truths[group], "A paired edit clock changed")
            truths[group] = truth
        else:
            require(
                truth is None, "An ordinary control was labeled as a planted face swap"
            )
        require(
            isinstance(stock, list) and all(interval(v) for v in stock),
            "Incumbent range outside frame clock",
        )
        if new is not None:
            require(
                codec == "hevc"
                and interval(new)
                and new[1] - new[0] in (3, 4)
                and audit is not None
                and new == audit["range"],
                "New range has no independent frame witness",
            )
        counts = rows[f"{codec}/{role}"]
        counts["files"] += 1
        state, usable = item["coverage"], item["with_views"]
        require(
            state in (None, "unassessed", "no_match", "localized")
            and type(usable) is bool,
            "Recorded face coverage differs",
        )
        if state is not None:
            counts[state] += 1
            counts["with_views"] += usable
        else:
            require(not usable, "Unavailable face check shows measured views")
        exact_cuts += role == "cut" and item["exact_cut_observed"] is True
        all_ranges = stock + ([new] if new else [])
        if truth is None:
            counts["stock_ordinary_files"] += bool(stock)
            counts["new_ordinary_files"] += bool(all_ranges)
            counts["added_ordinary_ranges"] += new is not None
        else:
            counts["stock_exact"] += truth in stock
            counts["stock_useful"] += any(
                iou(i, truth) >= Fraction(1, 2) for i in stock
            )
            counts["enhanced_exact"] += truth in all_ranges
            counts["enhanced_useful"] += any(
                iou(i, truth) >= Fraction(1, 2) for i in all_ranges
            )
            counts["wrong_added"] += new is not None and new != truth
            gained = new == truth and truth not in stock
            counts["strict_added"] += gained
            if gained and role == "swap_plus_blur":
                gain_groups.add(group)
        if audit is not None:
            a, b = audit["range"]
            require(
                (
                    codec == "hevc"
                    and new is not None
                    or codec == "avc"
                    and audit["range"] in stock
                )
                and state == "localized"
                and usable
                and interval(audit["range"])
                and b - a in (3, 4)
                and (audit["display_frames"], audit["range_ticks"], audit["views"])
                == ([a - 1, a + (b - a) // 2, b], [a * 3600, b * 3600], b - a + 6)
                and float.fromhex(POINT) < float.fromhex(audit["response_hex"]) <= 2.0,
                "Invalid local face proof",
            )
            anchors += len(audit["display_frames"])
            bound_views += audit["views"]
    require(
        roster
        == {
            (g, c, r)
            for g in range(1, 6)
            for r in ROLES
            for c in (("hevc", "avc") if r in PAIR_ROLES else ("hevc",))
        },
        "Holdout matrix incomplete",
    )
    require(
        all(
            item["staged_start"]
            == (None if item["role"] == "untouched" else truths[item["group"]][0])
            for item in doc["observations"]
        ),
        "Matched treatments use different authored clocks",
    )
    ordinary = [rows[f"hevc/{r}"] for r in ROLES - EDITS]
    ordinary_flags = sum(row["added_ordinary_ranges"] for row in ordinary)
    views = sum(row["with_views"] for row in ordinary)
    wrong = sum(rows[f"hevc/{r}"]["wrong_added"] for r in EDITS)
    new_cut = rows["hevc/cut"]["added_ordinary_ranges"]
    qualified = (
        rows["hevc/swap_plus_blur"]["strict_added"] >= GATE["gain"]
        and len(gain_groups) >= GATE["gain_groups"]
        and ordinary_flags <= GATE["max_added_ordinary"]
        and new_cut <= GATE["max_added_cut"]
        and wrong <= GATE["max_wrong"]
        and views >= GATE["min_ordinary_with_views"]
    )
    return {
        "qualifies": qualified,
        "gain_groups": len(gain_groups),
        "wrong_added_ranges": wrong,
        "added_ordinary_ranges": ordinary_flags,
        "ordinary_measured_files": views,
        "native_rgb_anchors": anchors,
        "bound_native_views": bound_views,
        "verified_literal_cut_edges": exact_cuts,
        "rows": {k: dict(v) for k, v in rows.items()},
    }


if __name__ == "__main__":
    receipt = json.loads(Path(sys.argv[1]).read_bytes())
    result = review(receipt)
    require(
        result == receipt["summary"],
        "Independent count differs from the stored aggregate",
    )
    print(json.dumps(result, sort_keys=True))
