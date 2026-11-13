# SPDX-License-Identifier: Apache-2.0
"""Recompute the held operating decisions for persistent perioral sectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def overlap(a, b):
    x, y = map(Fraction, a)
    u, v = map(Fraction, b)
    return max(0, min(y, v) - max(x, u)) / (max(y, v) - min(x, u))


def classify(item, point, truth, minimum):
    if item is None:
        return "unassessed"
    if item["score"] <= point:
        return "no_range"
    if truth is None:
        return "ordinary_flag"
    return "useful" if overlap(item["source_clock_s"], truth) >= minimum else "misplaced"


def outcome(rows, spec, points):
    minimum = Fraction(spec["gate"]["minimum_useful_iou"])
    by_bank, conditions, compatible = (
        defaultdict(lambda: defaultdict(Counter)),
        defaultdict(lambda: defaultdict(Counter)),
        defaultdict(lambda: defaultdict(Counter)),
    )
    group_gains, added, lost, new_controls = defaultdict(set), Counter(), Counter(), Counter()
    for row in rows:
        bank = row["condition"] if row["bank"] == "delivery" else row["bank"]
        calls = {}
        for arm, item in row["arms"].items():
            if arm == "direct" and bank == "reserve":
                continue
            state = classify(item, points[arm], row["truth"], minimum)
            calls[arm] = state
            by_bank[bank][arm][state] += 1
            by_bank[bank][arm]["edited" if row["truth"] else "ordinary"] += 1
            if item is not None:
                by_bank[bank][arm]["measured_edited" if row["truth"] else "measured_ordinary"] += 1
            conditions[f"{bank}/{row['condition']}"][arm][state] += 1
            if row["compatible"]:
                compatible[bank][arm][state] += 1
        previous, proposed = calls.get("direct"), calls["persistent_sectors"]
        if proposed == "useful" and previous != "useful":
            added[bank] += 1
            added[bank + "_compatible"] += row["compatible"]
            group_gains[bank].add(row["component"])
        if previous == "useful" and proposed != "useful":
            lost[bank] += 1
        if proposed == "ordinary_flag" and (
            previous != "ordinary_flag"
            or overlap(
                row["arms"]["persistent_sectors"]["source_clock_s"], row["arms"]["direct"]["source_clock_s"]
            )
            < minimum
        ):
            new_controls[bank] += 1
    gate = spec["gate"]
    coverage = all(
        4 * t[f"measured_{cls}"] >= 3 * t[cls]
        for bank in ("grid", "native")
        for t in (by_bank[bank]["persistent_sectors"],)
        for cls in ("edited", "ordinary")
    )
    criteria = {
        "grid": added["grid"] >= gate["min_grid_added"]
        and len(group_gains["grid"]) >= gate["min_grid_groups"],
        "compatible": added["grid_compatible"] >= gate["min_compatible_added"],
        "crema": added["native"] >= gate["min_crema_added"],
        "grid_controls": new_controls["grid"] <= gate["max_grid_ordinary"],
        "crema_controls": new_controls["native"] <= gate["max_crema_ordinary"],
        "reencode_controls": new_controls["reencode"] <= gate["max_reencode"],
        "cut_controls": new_controls["hard_cut"] <= gate["max_cut"],
        "reserve": added["reserve_compatible"] >= gate["min_reserve_compatible"]
        and new_controls["reserve"] <= gate["max_reserve_ordinary"],
        "coverage": coverage,
    }
    return {
        "banks": by_bank,
        "conditions": conditions,
        "compatible": compatible,
        "added": added,
        "lost": lost,
        "new_controls": new_controls,
        "groups_added": {k: len(v) for k, v in group_gains.items()},
        "criteria": criteria,
        "advance": all(criteria.values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    p.add_argument("--direct", type=Path)
    p.add_argument("--native-audit", type=Path)
    a = p.parse_args()
    if a.assemble:
        run = a.assemble
        freeze, point = read(run / "freeze.json"), read(run / "points.json")
        require(
            point["freeze_sha256"] == sha(run / "freeze.json")
            and point["journal_sha256"] == sha(run / "development.jsonl"),
            "Frozen ordinary development point differs",
        )
        rows = [*journal(run / "held.jsonl"), *journal(run / "reserve.jsonl")]
        require(
            Counter(r["bank"] for r in rows) == {"grid": 88, "native": 60, "delivery": 12, "reserve": 8},
            "Held roster differs",
        )
        calibration = read(a.direct)
        points = {**point["points"], "direct": calibration["points"]["direct"]}
        spec = freeze["spec"]
        document = {
            "schema": "chipmunk.video-mouth-sector-persistence-receipt/1",
            "spec": spec,
            "points": points,
            "freeze_sha256": sha(run / "freeze.json"),
            "ordinary_sha256": sha(run / "development.jsonl"),
            "rows": rows,
            "results": outcome(rows, spec, points),
            "native_audit": None if a.native_audit is None else read(a.native_audit),
        }
        a.receipt.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
    receipt = read(a.receipt)
    results = outcome(receipt["rows"], receipt["spec"], receipt["points"])
    require(
        receipt["schema"] == "chipmunk.video-mouth-sector-persistence-receipt/1"
        and results == receipt["results"],
        "Published held outcomes differ",
    )
    print(
        json.dumps(
            {
                k: results[k]
                for k in ("banks", "compatible", "added", "lost", "new_controls", "criteria", "advance")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
