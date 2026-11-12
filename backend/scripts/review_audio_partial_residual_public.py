"""Replay released temporal-audio locations and counts using only the standard library."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SAMPLES = 64_000
CELL = 3_200
SCHEMA = "chipmunk.audio-partial-residual-transfer/1"
ARMS = ("incumbent_clip", "incumbent_frames", "clean_head", "mixture_head", "paired_head")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def located(bounds, truth):
    if not truth:
        return False
    a, b = bounds
    p, q = truth
    overlap = max(0, min(b, q) - max(a, p))
    return 10 * overlap >= 3 * (b - a + q - p - overlap) and 2 * overlap >= q - p


def verify(document):
    require(document["schema"] == SCHEMA, "Unknown receipt format")
    counts, sets = defaultdict(Counter), defaultdict(set)
    rows = document["decisions"]
    require(len(rows) == len({r["id"] for r in rows}) == 168, "Missing or repeated public views")
    for row in rows:
        require(row["role"] in {"selection", "confirmation"} and row["label"] in {0, 1}, "Unknown source role or truth")
        truth = row["truth"]
        require((truth is not None) == bool(row["label"]) and (truth is None or 0 <= truth[0] < truth[1] <= SAMPLES),
                "Exact source interval is out of bounds")
        require(tuple(row["checks"]) == ARMS, "Missing comparison")
        for arm, check in row["checks"].items():
            start, stop = check["range"]
            require(type(start) is int and type(stop) is int and 0 <= start < stop <= SAMPLES, "Scored range is out of bounds")
            require((start, stop) == (0, SAMPLES) if arm == "incumbent_clip"
                    else start % CELL == 0 and stop - start == 3 * CELL, "Scored range differs from its sampling grid")
            flag = check["flag"]
            hit = located((start, stop), truth)
            label = row["label"]
            require(type(flag) is bool and not (not label and (check["beats_both"] or check["oracle_beats_both"])), "Invalid direction")
            counts[row["role"], row["condition"], arm].update(
                positive=label, controls=1-label, true_flag=int(label and flag), false_flag=int(not label and flag),
                useful=int(flag and hit), wrong_location=int(label and flag and not hit),
                inconclusive_positive=int(label and not flag), inconclusive_control=int(not label and not flag),
                threshold_free_useful=int(hit), beats_both=int(check["beats_both"]), oracle_beats_both=int(check["oracle_beats_both"]))
            if row["role"] == "confirmation" and flag and hit:
                sets["locations", arm].add(row["id"])
            if row["role"] == "confirmation" and row["condition"] != "dry" and flag and not label:
                sets["false", arm].add(row["pair"])
    aggregate = document["aggregate"]
    cells = [{"role": role, "condition": condition, "arm": arm, **values} for (role, condition, arm), values in sorted(counts.items())]
    require(cells == aggregate["counts"], "Public decisions do not reproduce the reported outcome counts")
    baseline, candidate = "incumbent_frames", "paired_head"
    gains = sets["locations", candidate] - sets["locations", baseline]
    losses = sets["locations", baseline] - sets["locations", candidate]
    held = [row for row in rows if row["role"] == "confirmation"]
    added = [row for row in held if row["condition"] != "dry" and row["id"] in gains]
    gate = {"added_noisy_views": len(added), "lost_noisy_views": sum(row["condition"] != "dry" and row["id"] in losses for row in held),
            "added_noisy_pairs": len({row["pair"] for row in added}), "added_noisy_families": len({row["family"] for row in added}),
            "added_noisy_false_pairs": len(sets["false", candidate] - sets["false", baseline]),
            "lost_dry_useful_pairs": len({row["pair"] for row in held if row["condition"] == "dry" and row["id"] in losses}),
            "background_flags": sum(row["arm"] == "background_only" and row["checks"][candidate]["flag"] for row in held)}
    advancement = (gate["added_noisy_views"] - gate["lost_noisy_views"] >= 4 and gate["added_noisy_pairs"] >= 3
                   and gate["added_noisy_families"] >= 2 and gate["added_noisy_false_pairs"] <= 1
                   and gate["lost_dry_useful_pairs"] == gate["background_flags"] == 0)
    families = []
    for family in sorted({row["family"] for row in held if row["label"] and row["condition"] != "dry"}):
        ids = {row["id"] for row in held if row["label"] and row["condition"] != "dry" and row["family"] == family}
        families.append({"family": family, "positive": len(ids), "useful": {arm: len(ids & sets["locations", arm]) for arm in ARMS}})
    require((gate, advancement, families) == (aggregate["gate"], aggregate["qualifies"], aggregate["held_noisy_by_family"]),
            "Public decisions do not reproduce the held criteria and family counts")
    return {"decisions": len(rows) * len(ARMS), "aggregate_cells": len(cells), "advances": advancement, **gate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.receipt.read_bytes())), sort_keys=True))
