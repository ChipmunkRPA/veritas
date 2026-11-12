"""Replay public time ranges, aggregate counts and the fixed advancement test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-partial-layer-boundary/1"
ARMS = ("incumbent_frames", "weighted_local", "early_local", "middle_local", "deep_local", "layer_boundaries")
BASELINE, CANDIDATE = ARMS[0], ARMS[-1]
CELL, SAMPLES = 3200, 64000


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def hit(bounds, truth):
    if truth is None:
        return False
    start, end = bounds
    a, b = truth
    common = len(range(max(start, a), min(end, b)))
    union = end - start + b - a - common
    return common * 10 >= union * 3 and common * 2 >= b - a


def verify(receipt, parent_bytes):
    require(receipt["schema"] == SCHEMA, "Unknown speech layer receipt")
    require(receipt["parent_public_sha256"] == hashlib.sha256(parent_bytes).hexdigest(), "Prior public receipt differs")
    parent = {row["id"]: row for row in json.loads(parent_bytes)["decisions"]}
    rows = receipt["decisions"]
    require(len(rows) == len(parent) == len({r["id"] for r in rows}) == 168, "Public case coverage differs")
    counts, located, false_pairs, backgrounds = defaultdict(Counter), defaultdict(set), defaultdict(set), Counter()
    for row in rows:
        original = parent[row["id"]]
        require(all(row[key] == original[key] for key in ("role", "condition", "pair", "family", "arm", "label", "truth")),
                "Prior input truth, role or condition differs")
        require(tuple(row["checks"]) == ARMS, "The fixed comparison arms differ")
        for arm, decision in row["checks"].items():
            start, end = decision["range"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= SAMPLES and
                    start % CELL == 0 and end - start == 3 * CELL, "Invalid time cell extent")
            label, flag = row["label"], decision["flag"]
            rank, oracle = decision["rank_both"], decision["oracle_rank_both"]
            require(all(type(value) is bool for value in (flag, rank, oracle)) and (label or not (rank or oracle)),
                    "Invalid flag or matched-pair direction")
            if arm == BASELINE:
                prior = original["checks"][BASELINE]
                require((prior["range"], prior["flag"], prior["beats_both"], prior["oracle_beats_both"]) ==
                        (decision["range"], flag, rank, oracle), "Original comparator changed")
            useful = hit((start, end), row["truth"])
            counts[row["role"], row["condition"], arm].update(positive=label, genuine=1-label,
                ai_flags=int(label and flag), useful=int(flag and useful), false_flags=int(not label and flag),
                threshold_free_locations=int(useful), rank_both=int(rank), oracle_rank_both=int(oracle))
            if row["role"] == "confirmation" and flag:
                if useful:
                    located[arm].add(row["id"])
                if row["condition"] == "speech_minus5db" and not label:
                    if row["pair"] is not None:
                        false_pairs[arm].add(row["pair"])
                    if row["arm"] == "background_only":
                        backgrounds[arm] += 1
    severe = {r["id"] for r in rows if r["role"] == "confirmation" and r["condition"] == "speech_minus5db"}
    dry = {r["id"] for r in rows if r["role"] == "confirmation" and r["condition"] == "dry"}
    added, lost = located[CANDIDATE] - located[BASELINE], located[BASELINE] - located[CANDIDATE]
    gate = {"added_useful": len(added & severe), "lost_useful": len(lost & severe),
            "added_families": len({parent[i]["family"] for i in added & severe}),
            "extra_false_pairs": len(false_pairs[CANDIDATE] - false_pairs[BASELINE]),
            "lost_dry_useful": len(lost & dry), "background_flags": backgrounds[CANDIDATE]}
    follow_up = (gate["added_useful"] - gate["lost_useful"] >= 2 and gate["added_families"] >= 2
                 and gate["extra_false_pairs"] <= 1 and gate["lost_dry_useful"] == gate["background_flags"] == 0)
    cells = [{"role": r, "condition": c, "arm": a, **counts[r, c, a]} for r, c, a in sorted(counts)]
    expected = {"counts": cells, "primary": gate, "follow_up": follow_up, "release_eligible": False}
    require(receipt["aggregate"] == expected, "Published summary cannot be reconstructed from the public decisions")
    return {"decisions": len(rows) * len(ARMS), "aggregate_cells": len(cells), **gate,
            "follow_up": follow_up, "release_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("parent_receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.receipt.read_bytes()), args.parent_receipt.read_bytes()), sort_keys=True))
