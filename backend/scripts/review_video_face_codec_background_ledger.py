# SPDX-License-Identifier: Apache-2.0
"""Verify the public codec comparison counts from the anonymous decision ledger."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

POINT = .60
EXPECTED = {
    "held": Counter(swap=12, cut=12, ordinary=18),
    "exposed": Counter(swap=8, cut=4, ordinary=20),
}


def valid_span(span):
    return (type(span) is list and len(span) == 2 and all(type(n) is int for n in span)
            and 0 <= span[0] < span[1] < 120)


def hits(span, truth):
    if truth is None:
        return False
    a, b = span
    x, y = truth
    return 2 * max(0, min(b, y) - max(a, x)) >= max(b, y) - min(a, x)


def verify(ledger):
    if ledger["schema"] != "chipmunk.video-face-codec-background-ledger/1":
        raise ValueError("Unknown public decision ledger schema")
    totals = {}
    for phase, expected in EXPECTED.items():
        totals[phase] = {}
        for codec in ("vp9", "hevc"):
            rows = ledger[phase][codec]
            counts = {kind: Counter() for kind in expected}
            added_groups = set()
            for row in rows:
                truth, prior, dense, quartile = (row[key] for key in ("truth", "baseline", "dense", "guard_min_quartile"))
                kind = row["kind"]
                if ((kind == "swap") != (truth is not None) or type(row["measured"]) is not bool
                        or any(not valid_span(x) for x in prior + ([truth] if truth is not None else []))
                        or (dense is not None and (not valid_span(dense) or not row["measured"]))
                        or (quartile is not None and (type(quartile) not in (float, int)
                                                      or not math.isfinite(quartile) or not -1.001 <= quartile <= 1.001))
                        or (dense is None) != (quartile is None)):
                    raise ValueError("Invalid public comparison row")
                admitted = quartile is not None and quartile >= POINT
                original = any(hits(span, truth) for span in prior)
                added = dense is not None and hits(dense, truth)
                if added and admitted and not original:
                    added_groups.add(row["group"])
                counts[kind].update({"cases": 1, "baseline_flags": bool(prior), "fallback_measured": row["measured"],
                    "dense_flags": dense is not None, "guarded_dense_flags": admitted, "baseline_located": original,
                    "unfiltered_located": original or added, "guarded_located": original or (added and admitted)})
            if Counter({kind: c["cases"] for kind, c in counts.items()}) != expected:
                raise ValueError("Missing study deliveries")
            totals[phase][codec] = {"counts": {kind: dict(c) for kind, c in counts.items()}, "added_held_groups": len(added_groups)}
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_bytes())
    totals = verify(ledger)
    if totals != ledger["expected"]:
        raise ValueError("Recomputed results differ from the expected aggregate")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
