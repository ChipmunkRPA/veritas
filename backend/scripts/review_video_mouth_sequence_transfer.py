# SPDX-License-Identifier: Apache-2.0
"""Recompute the paired-mouth benchmarks from their public case receipts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def summarize(rows, arm):
    groups = set()
    summary = Counter()
    per_role = {}
    for row in rows:
        if arm not in row["arms"]:
            raise ValueError("Receipt is missing a method")
        by_role = per_role.setdefault(row["role"], Counter())
        data = row["arms"][arm]
        truth, match = row["authored"], data["match"]
        if bool(data["assessed"]) is False and match is not None:
            raise ValueError("Unassessed case carries a match")
        by_role["cases"] += 1
        by_role["assessed"] += data["assessed"]
        if not match:
            continue
        (a, b), (c, d) = match["ranges"]
        if not (
            2 <= a < b <= c < d <= 48
            and 7 <= b - a == d - c <= 12
            and c - a == match["lag"]
            and 9 <= match["lag"] <= 32
            and 0 <= match["error"] <= data["point"]
            and len(match["frames"]) == 8
        ):
            raise ValueError("Invalid source ranges")
        indices = (a - 1, a, (a + b) // 2, b, c - 1, c, (c + d) // 2, d)
        for index, witness in zip(indices, match["frames"], strict=True):
            if (
                witness["frame"] != index
                or witness["pts"] != index * 1000
                or any(len(bytes.fromhex(witness[key])) != 32 for key in ("crop_sha256", "rgb_sha256"))
            ):
                raise ValueError("Invalid native frame commitments")
        if not truth:
            by_role["false"] += 1
        elif c - a == truth[1][0] - truth[0][0] and min(
            overlap(first, second) for first, second in zip(match["ranges"], truth, strict=True)
        ) >= Fraction(3, 5):
            by_role["correct"] += 1
            by_role["exact"] += int(match["ranges"] == truth)
            groups.add(row["group"])
        else:
            by_role["wrong"] += 1
    for values in per_role.values():
        summary.update(values)
    return {"total": dict(summary), "groups": len(groups), "roles": {k: dict(v) for k, v in per_role.items()}}


def review(document):
    if document["schema"] != "chipmunk.video-mouth-sequence-transfer/1":
        raise ValueError("Unsupported receipt")
    output = {}
    for panel in document["panels"]:
        rows = panel["rows"]
        if len({(row["role"], row["group"]) for row in rows}) != len(rows):
            raise ValueError("Duplicated source group role")
        for arm, expected in panel["summary"].items():
            actual = summarize(rows, arm)
            if expected != actual:
                raise ValueError("Published results differ from source commitments")
            output[f"{panel['bank']}/{arm}"] = actual["total"]
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    args = p.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
