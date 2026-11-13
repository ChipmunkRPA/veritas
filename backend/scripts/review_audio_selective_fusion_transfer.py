# SPDX-License-Identifier: Apache-2.0
"""Recalculate selective audio results from the public source-free receipts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ARMS = ("raw", "global", "context", "common_reference", "common_agreement")
POOLED = {"raw", "global"}
DIRECTIONS = {"ai", "original", "inconclusive", "unassessed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(rows, arm):
    outcome = [row["current"] if arm == "current" else row["directions"][arm] for row in rows]
    paired = list(zip(rows, outcome, strict=True))
    genuine = [row for row in rows if row["label"] == 0]
    ai = len(rows) - len(genuine)
    false_genuine = [row for row, direction in paired if row["label"] == 0 and direction == "ai"]
    found = sum(row["label"] == 1 and direction == "ai" for row, direction in paired)
    return {"inputs": len(rows), "source_files": len({r["base_id"] for r in rows}), "connected_groups": len({r["group"] for r in rows}),
            "ai_inputs": ai, "genuine_inputs": len(genuine), "ai_found": found, "false_ai": len(false_genuine),
            "genuine_correct": sum(row["label"] == 0 and direction == "original" for row, direction in paired),
            "false_original": sum(row["label"] == 1 and direction == "original" for row, direction in paired),
            "inconclusive_assessed": outcome.count("inconclusive"), "unassessed": outcome.count("unassessed"),
            "genuine_groups_flagged": len({r["group"] for r in false_genuine}), "ai_without_ai_conclusion": ai - found,
            "without_direction": outcome.count("inconclusive") + outcome.count("unassessed")}


def frozen_points(data):
    points = data["points"]
    require(set(points) == set(ARMS[:-1]), "Unexpected policy arms")
    for arm, buckets in points.items():
        require(set(buckets) == ({"all"} if arm in POOLED else {"low", "high"}), "Context scope differs")
        for item in buckets.values():
            maxima = item["genuine_group_maxima"]
            require(maxima == sorted(maxima, reverse=True) and all(math.isfinite(v) and 0 <= v <= 1 for v in maxima)
                    and item["point"] == (maxima[2] if len(maxima) >= 8 else None), "Fixed genuine point differs")
    return points


def verify_archived(data):
    require(data["schema"] == "chipmunk.audio-selective-fusion-transfer/1", "Wrong archived receipt schema")
    points = frozen_points(data)
    require(set(data["roles"]) == {"selection", "confirmation"}, "Both archived roles are required")
    for name, role in data["roles"].items():
        rows = role["cases"]
        require(len(rows) == 1044 and len({(row["base_id"], row["condition"]) for row in rows}) == len(rows), "Role size differs")
        require(len({row["base_id"] for row in rows}) == 116 and set(role["result"]["arms"]) == set(ARMS), "Source roster differs")
        for row in rows:
            scores, context, current = row["scores"], row["context"], row["current"]
            require(current in DIRECTIONS and context in {"low", "high"}
                    and (not scores or current == "inconclusive" and set(scores) == set(ARMS[:-1])), "Out-of-scope input was scored")
            require(all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in scores.values()), "Invalid score")
            found = set()
            for arm, value in scores.items():
                point = points[arm]["all" if arm in POOLED else context]["point"]
                if point is not None and value is not None and value > point:
                    found.add(arm)
            if {"raw", "common_reference"} <= found:
                found.add("common_agreement")
            require(row["directions"] == {arm: "ai" if arm in found else current for arm in ARMS}, "Direction differs from frozen policy")
        current = counts(rows, "current")
        require(current == role["result"]["current"], "Current aggregate differs")
        for arm, result in role["result"]["arms"].items():
            new = [row for row in rows if row["current"] != "ai" and row["directions"][arm] == "ai"]
            ai = [row for row in new if row["label"] == 1]
            genuine = [row for row in new if row["label"] == 0]
            existing = {row["group"] for row in rows if row["label"] == 0 and row["current"] == "ai"}
            added = len({row["group"] for row in genuine} - existing)
            families, sources = sorted({row["kind"] for row in ai}), len({row["base_id"] for row in ai})
            require((result["new_ai_views"], result["new_genuine_views"], result["new_genuine_groups"],
                     result["new_ai_sources"], result["new_ai_families"], result["qualifies_for_fresh_test"])
                    == (len(ai), len(genuine), added, sources, families, sources >= 4 and len(families) >= 2 and added <= 2),
                    "Paired outcome or advancement differs")
            require(result["total"] == counts(rows, arm), "Candidate aggregate differs")
            for field in ("condition", "context"):
                groups = result["by_" + field]
                require(set(groups) == {row[field] for row in rows}, "Condition coverage differs")
                for key, value in groups.items():
                    selected = [row for row in rows if row[field] == key]
                    require(value == {"current": counts(selected, "current"), "candidate": counts(selected, arm)}, "Condition count differs")
        print(name, {arm: value["new_ai_views"] for arm, value in role["result"]["arms"].items()})
    return points


def verify_radar(data, points):
    require(data["schema"] == "chipmunk.audio-selective-radar-transfer/1", "Wrong prospective receipt schema")
    require(data["points"] == {arm: points[arm]["all"]["point"] for arm in POOLED}, "External points were changed")
    rows = data["cases"]
    require(len(rows) == data["independent_native_inputs"] == 32, "Prospective roster differs")
    require(Counter((r["split"], r["label"]) for r in rows) == {
        (split, label): 8 for split in ("development", "evaluation") for label in ("bonafide", "spoof")}, "Publisher label strata differ")
    for row in rows:
        require((row["start_sample"], row["end_sample"]) == (0, 64000), "Source interval differs")
        directions, scores = row["directions"], row["scores"]
        current = directions["current"]
        require(current in DIRECTIONS and set(scores) == POOLED, "Unknown prospective direction or arm")
        for arm, value in scores.items():
            require(value is None or current == "inconclusive" and math.isfinite(value) and 0 <= value <= 1, "Out-of-scope score")
            expected = "ai" if value is not None and value > data["points"][arm] else current
            require(directions[arm] == expected, "Prospective direction differs")
    by_split = {}
    for split in ("development", "evaluation", "all"):
        selected = [row for row in rows if split == "all" or row["split"] == split]
        by_split[split] = {arm: {label: dict(Counter(row["directions"][arm] for row in selected if row["label"] == label))
                                 for label in ("bonafide", "spoof")} for arm in ("current", *sorted(POOLED))}
    advancement = {}
    for arm in POOLED:
        gained = [row for row in rows if row["directions"][arm] == "ai" and row["directions"]["current"] != "ai"]
        spoof = [row for row in gained if row["label"] == "spoof"]
        advancement[arm] = len(spoof) >= 4 and len({row["split"] for row in spoof}) == 2 and len(gained) - len(spoof) <= 1
    require(data["results"] == {**by_split, "advances": advancement}, "Prospective counts differ")
    print("RADAR", by_split["all"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archived", type=Path)
    parser.add_argument("radar", type=Path)
    args = parser.parse_args()
    verify_radar(json.loads(args.radar.read_bytes()), verify_archived(json.loads(args.archived.read_bytes())))
