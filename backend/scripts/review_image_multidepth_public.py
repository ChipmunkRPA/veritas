# SPDX-License-Identifier: Apache-2.0
"""Check the public image-localization counts using the released scalar ledger."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

FORMATS = ("png", "jpeg75")
ARMS = ("last_layer", "three_depths", "format_reference")
POSITIVE = ("edited", "feathered_edit")
CONTROLS = ("original", "local_processing", "global_processing", "blur", "feathered_processing", "local_contrast")
PANELS = ("earlier", "later")


def require(test, message):
    if not test:
        raise ValueError(message)


def useful(reported, target, overlap):
    return target > 0 and 2 * overlap >= target and 10 * overlap >= 3 * (reported + target - overlap)


def panel_counts(rows, points):
    source_ids = {row[0] for row in rows}
    expected = set(product(source_ids, FORMATS, (*POSITIVE, *CONTROLS)))
    require(len(source_ids) == 12 and len(rows) == len(expected) == 192, "Expected twelve complete source groups")
    counts = {
        fmt: {arm: Counter(sharp=0, feathered=0, ordinary=0, original=0, extra_ordinary=0, extra_feathered=0,
                           ungated_useful=None if arm == ARMS[-1] else 0) for arm in ARMS} for fmt in FORMATS
    }
    new_sources = {fmt: {arm: set() for arm in ARMS} for fmt in FORMATS}
    for source_id, fmt, role, target, values in rows:
        expected.remove((source_id, fmt, role))
        require(type(target) is int and (target > 0 if role in POSITIVE else target == 0)
                and len(values) == len(ARMS), "Invalid target or response count")
        baseline_area, baseline_overlap, _, _ = values[-1]
        baseline_hit = useful(baseline_area, target, baseline_overlap)
        for arm, (area, overlap, score, ungated) in zip(ARMS, values, strict=True):
            require(type(area) is type(overlap) is int and 0 <= overlap <= min(target, area), "Invalid pixel counts")
            result = counts[fmt][arm]
            if arm == ARMS[-1]:
                require(score is None and ungated is None, "Unexpected incumbent response")
            else:
                require(type(score) in (float, int) and math.isfinite(score) and type(ungated) is bool,
                        "Invalid scalar response")
                require((score > points[fmt][arm]["point"]) == (area > 0), "Response and selected region disagree")
                require(role in POSITIVE or not ungated, "Ordinary control has no positive target")
                result["ungated_useful"] += ungated
            if role in POSITIVE and useful(area, target, overlap):
                result["sharp" if role == POSITIVE[0] else "feathered"] += 1
                result["extra_feathered"] += role == POSITIVE[1] and not baseline_hit
            if role in CONTROLS and area > 0:
                result["ordinary"] += 1
                result["original"] += role == CONTROLS[0]
                if not baseline_area:
                    result["extra_ordinary"] += 1
                    new_sources[fmt][arm].add(source_id)
    require(not expected, "Incomplete roster")
    for fmt in FORMATS:
        for arm in ARMS:
            counts[fmt][arm]["extra_ordinary_sources"] = len(new_sources[fmt][arm])
    return counts


def review(receipt):
    ledger = receipt["public_scalar_ledger"]
    require(tuple(ledger["arm_order"]) == ARMS and set(ledger["rows"]) == set(PANELS), "Unexpected public ledger layout")
    points = receipt["operating_points"]
    for fmt, arm in product(FORMATS, ARMS[:-1]):
        record = points[fmt][arm]
        values = record["source_maxima"]
        require(len(values) == 12 and all(math.isfinite(value) for value in values)
                and record["point"] == sorted(values)[-2], "Incorrect fixed point")
    counts = {label: panel_counts(ledger["rows"][label], points) for label in PANELS}
    require(counts == receipt["result"]["results"], "Public outcome counts differ from the ledger")
    decisions = {}
    for fmt in FORMATS:
        by_panel = [result[fmt]["three_depths"] for result in counts.values()]
        decisions[fmt] = (min(row["extra_feathered"] for row in by_panel) >= 2
                          and max(row["extra_ordinary"] for row in by_panel) <= 4
                          and max(row["extra_ordinary_sources"] for row in by_panel) <= 3)
    require(decisions == receipt["result"]["three_depths_advances"], "Public advancement decision differs")
    return {"verified_scalar_views": 384, "verified_region_counts": 1152, "three_depths_advances": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
