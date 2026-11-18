# SPDX-License-Identifier: Apache-2.0
"""Recount three-copy-pair outcomes from anonymous geometry and scalar results."""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SCHEMA = "chipmunk.image-three-exact-pairs-holdout-review/1"
POSITIVE = {"translation_turn_reflection", "translation_two_reflections"}
ORDINARY = {"untouched", "repeated_ordinary", "whole_symmetry"}
GUARDS = {"two_only", "shared_endpoint", "adjacent_endpoint", "four_disjoint"}
TRANSFORMS = {"same_orientation", "rotate_180", "horizontal_reflection", "vertical_reflection"}
CARRIERS = {"PNG", "WEBP"}


def fingerprint(geometry):
    transform, boxes = geometry
    if transform not in TRANSFORMS or len(boxes) != 2:
        raise ValueError("Unexpected transform or crop count")
    for x, y, w, h in boxes:
        if not 16 <= min(w, h) <= max(w, h) <= 56 or not (0 <= x <= 512 - w and 0 <= y <= 512 - h):
            raise ValueError("Crop falls outside the frozen source geometry")
    return transform, tuple(sorted(tuple(box) for box in boxes))


def separate(two_a, two_b):
    return all(
        x + w + 16 <= bx or x >= bx + bw + 16 or y + h + 16 <= by or y >= by + bh + 16
        for x, y, w, h in two_a
        for bx, by, bw, bh in two_b
    )


def summarize(data):
    counts = defaultdict(Counter)
    unique = set()
    gains = defaultdict(set)
    if data["schema"] != SCHEMA or data["profile"] != {
        "size": [512, 512],
        "separation": 16,
        "pair_capacity": 3,
    }:
        raise ValueError("Unknown held comparison contract")
    if len(data["rows"]) != 140 or set(row["carrier"] for row in data["rows"]) != CARRIERS:
        raise ValueError("Incomplete encoding roster")
    for row in data["rows"]:
        carrier, group, role = (row[key] for key in ("carrier", "group", "role"))
        tag = carrier, group, role
        if tag in unique or role not in POSITIVE | ORDINARY | GUARDS or not 1 <= group <= 14:
            raise ValueError("Duplicate or unexpected held delivery")
        unique.add(tag)
        truth = {fingerprint(x) for x in row["truth"]}
        estimates = [fingerprint(x) for x in row["proposed"]]
        if len(estimates) not in (0, 3) or len(set(estimates)) != len(estimates):
            raise ValueError("The candidate must withhold or return three distinct pairs")
        if not all(separate(a[1], b[1]) for a, b in combinations(estimates, 2)):
            raise ValueError("Pair endpoints do not follow the native separation rule")
        hit = len(set(estimates) & truth)
        s = counts[carrier]
        if role in POSITIVE:
            if not len(truth) == 3 or not 0 <= row["current"] <= row["combined"] <= 3:
                raise ValueError("Incorrect staged comparison counts")
            if row["combined"] < hit:
                raise ValueError("The combined comparison discarded exact evidence")
            improved = row["combined"] > row["current"]
            s.update(
                positive_files=1,
                old_useful_regions=row["current"],
                combined_useful_regions=row["combined"],
                old_full_three=row["current"] == 3,
                combined_full_three=row["combined"] == 3,
                exact_three_files=hit == 3,
                proposed_wrong=len(estimates) - hit,
                true_three_individually_eligible=row["individually_fit"] == 3,
                positive_files_improved=improved,
            )
            if improved:
                gains[carrier].add(group)
        elif role in ORDINARY:
            if truth:
                raise ValueError("Ordinary controls cannot have authored copy locations")
            s.update(ordinary_files=1, added_ordinary_alerts=bool(estimates))
        else:
            s.update({role + "_files": 1, role + "_alerts": bool(estimates)})
            if role == "two_only":
                s["ordinary_two_pair_still_currently_exact"] += row["current_exact"] == 2
            if role == "four_disjoint" and row["individually_fit"] == 4:
                s.update(all_four_eligible_files=1, all_four_bad_alert=bool(estimates))
    out = {}
    for carrier, s in counts.items():
        if (
            s["positive_files"] != 16
            or s["ordinary_files"] != 22
            or any(s[role + "_files"] != 8 for role in GUARDS)
        ):
            raise ValueError("A comparison condition is missing held images")
        extra_groups = len(gains[carrier])
        zero = (
            "added_ordinary_alerts",
            "proposed_wrong",
            "shared_endpoint_alerts",
            "adjacent_endpoint_alerts",
            "all_four_bad_alert",
        )
        passed = s["positive_files_improved"] >= 8 and extra_groups >= 3 and not any(s[k] for k in zero)
        out[carrier] = dict(s) | {"gain_groups": extra_groups, "passed": passed}
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_three_pairs_public.py HELD_RECEIPT.json")
    data = json.loads(Path(sys.argv[1]).read_bytes())
    result = summarize(data)
    if data["summary"] != result:
        raise ValueError("The reported outcome differs from the independently recounted pairs")
    print(
        json.dumps(
            {"held": len(data["rows"]), "freeze_sha256": data["freeze_sha256"], "summary": result},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
