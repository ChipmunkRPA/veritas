# SPDX-License-Identifier: Apache-2.0
"""Recount native exact copy locations from a public receipt without image libraries."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def identity(match):
    return match["transform"], tuple(sorted(tuple(box) for box in match["pair"]))


def recount(receipt):
    if receipt.get("schema") != "chipmunk.image-two-exact-copy-pairs-evaluation/1":
        raise ValueError("Unsupported evaluation receipt")
    totals = {}
    width, height = receipt["size"]
    seen = set()
    for row in receipt["rows"]:
        role = row["role"]
        key = row["source"], role
        if key in seen:
            raise ValueError("Repeated evaluation input")
        seen.add(key)
        counts = totals.setdefault(role, Counter())
        candidate = row["candidate"]
        pairs = candidate["pairs"]
        if len(pairs) != (2 if candidate["status"] == "supported_pair" else 0):
            raise ValueError("The number of exact pairs differs from the recorded result")
        truth = {identity(item) for item in row["truth"] if item["exact_shape_eligible"]}
        for match in pairs:
            area = match["measurement"]["matching_pixels"]
            for x, y, w, h in match["pair"]:
                if not (0 <= x < x + w <= width and 0 <= y < y + h <= height and 16 <= min(w, h) <= max(w, h) <= 56):
                    raise ValueError("Exact crop exceeds the registered image geometry")
                if area != w * h:
                    raise ValueError("Exact pixel count differs from the crop area")
        exact = len({identity(match) for match in pairs} & truth)
        counts.update(files=1, candidate_results=bool(pairs), exact_pairs=exact, both_exact=exact == 2,
                      eligible_truth_pairs=len(truth))
    return dict(sorted(totals.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    print(json.dumps(recount(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
