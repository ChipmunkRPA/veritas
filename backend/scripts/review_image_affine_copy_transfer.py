"""Audit affine-copy receipt counts with integer overlap arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROLES = {"sharp_affine", "feathered_affine", "original", "contrast", "blur", "whole_resize", "local_tone"}
FORMATS = {"png", "jpeg75"}
HEX = re.compile(r"[a-f0-9]{64}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(side):
    predicted, truth, overlap = (side[name] for name in ("reported", "target", "overlap"))
    if (any(type(value) is not int for value in (predicted, truth, overlap))
            or not 0 <= overlap <= min(predicted, truth) or predicted <= 0):
        raise ValueError("Invalid retained-pair pixel counts")
    iou = Fraction(overlap, predicted + truth - overlap)
    recall = Fraction(overlap, truth) if truth else None
    if (side["iou"] != (float(iou) if truth else None)
            or side["recall"] != (float(recall) if recall is not None else None)):
        raise ValueError("Reported metrics disagree with exact counts")
    return iou, recall, Fraction(overlap, predicted), truth


def score(pairs, positive):
    all_values = [[values(side) for side in pair] for pair in pairs]
    if any(len(pair) != 2 or any(bool(side[3]) != positive for side in pair) for pair in all_values):
        raise ValueError("Each undirected pair needs two matching truth roles")
    useful = [pair for pair in all_values if all(side[0] >= Fraction(3, 10) and side[1] is not None
                                                and side[1] >= Fraction(1, 2) for side in pair)]
    if not useful:
        return None
    best = max(useful, key=lambda pair: min(side[0] for side in pair))
    return {key: float(min(side[index] for side in best)) for index, key in enumerate(("iou", "recall", "precision"))}


def audit(path):
    plan = json.loads((path / "freeze.json").read_bytes())
    recorded = json.loads((path / "aggregate.json").read_bytes())
    if (recorded["freeze_sha256"] != sha(path / "freeze.json") or set(plan["formats"]) != FORMATS
            or set(plan["positive"]) | set(plan["controls"]) != ROLES):
        raise ValueError("The aggregate does not bind the expected protocol")
    tallies = {fmt: {role: Counter() for role in ROLES} for fmt in FORMATS}
    common = {fmt: {role: [] for role in plan["positive"]} for fmt in FORMATS}
    groups = {fmt: set() for fmt in FORMATS}
    count = pair_sides = 0
    for panel in ("earlier", "later"):
        journal = path / f"{panel}.jsonl"
        if sha(journal) != recorded["journal_sha256"][panel]:
            raise ValueError("The aggregate does not bind a panel journal")
        seen = set()
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            slot, fmt, role = (row[key] for key in ("slot", "delivery", "role"))
            key = slot, fmt, role
            if (type(slot) is not int or not 0 <= slot < 12 or key in seen or fmt not in FORMATS or role not in ROLES
                    or len(row["truth_sha256"]) != 2 or not all(HEX.fullmatch(value) for value in
                                                                  [row["encoded_sha256"], *row["truth_sha256"]])):
                raise ValueError("Malformed or duplicate panel input")
            seen.add(key)
            incumbent = row["baseline"]
            refined = row["dense"]
            proposals = [pair["proposal"] for pair in refined]
            if (type(row["assessed"]) is not bool or (not row["assessed"] and (incumbent or refined))
                    or proposals != sorted(set(proposals))
                    or any(type(i) is not int or not 0 <= i < min(3, len(incumbent)) for i in proposals)
                    or any(len(pair["mask_sha256"]) != 2 or not all(HEX.fullmatch(mask) for mask in pair["mask_sha256"])
                           for pair in refined)):
                raise ValueError("A dense pair lacks its corresponding sparse proposal or masks")
            positive = role in plan["positive"]
            old, new = score(incumbent, positive), score([pair["overlap"] for pair in refined], positive)
            old_useful, new_useful = old is not None, new is not None
            tallies[fmt][role].update({"views": 1, "assessed": row["assessed"], "incumbent_flagged": bool(incumbent),
                                       "dense_flagged": bool(refined), "incumbent_useful": old_useful, "dense_useful": new_useful,
                                       "added_useful": new_useful and not old_useful, "lost_useful": old_useful and not new_useful,
                                       "added_flag": bool(refined) and not incumbent})
            if old is not None and new is not None:
                common[fmt][role].append((old, new))
            if role == "feathered_affine" and new_useful and not old_useful:
                groups[fmt].add((panel, slot))
            pair_sides += 2 * (len(incumbent) + len(refined))
        if len(seen) != 168:
            raise ValueError("Panel journal is incomplete")
        count += len(seen)
    advances = {}
    jointly_useful = {}
    for fmt in sorted(FORMATS):
        advances[fmt] = (len(groups[fmt]) >= 3 and sum(tallies[fmt][role]["lost_useful"] for role in plan["positive"]) <= 2
                         and sum(tallies[fmt][role]["added_flag"] for role in plan["controls"]) == 0)
        jointly_useful[fmt] = {}
        for role, comparisons in common[fmt].items():
            medians = {key: [statistics.median(pair[arm][key] for pair in comparisons) for arm in (0, 1)]
                       for key in ("iou", "recall", "precision")} if comparisons else None
            jointly_useful[fmt][role] = {"views": len(comparisons), "median_worse_side_incumbent_dense": medians}
    if recorded["views"] != count or recorded["results"] != tallies or recorded["advances"] != advances:
        raise ValueError("Recomputed decisions differ from the aggregate")
    return {"verified_views": count, "verified_pair_sides": pair_sides, "advances": advances,
            "jointly_useful_diagnostics": jointly_useful}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    print(json.dumps(audit(parser.parse_args().receipts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
