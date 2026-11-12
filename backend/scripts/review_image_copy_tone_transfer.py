"""Independently recalculate a same-image tone-correction comparison from its receipts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.image-copy-tone-transfer/1"
FORMATS = ("png", "jpeg75")
POSITIVE = ("sharp", "feathered", "sharp_tone", "feathered_tone")
CONTROLS = ("published", "global_tone", "local_tone", "resize")
SHA = re.compile(r"[0-9a-f]{64}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pairs(pairs, truth):
    useful = False
    for pair in pairs:
        if len(pair["mask_sha256"]) != 2 or not all(SHA.fullmatch(value) for value in pair["mask_sha256"]):
            raise ValueError("Each pair must commit both masks")
        if len(pair["counts"]) != 2 or sorted(side[1] for side in pair["counts"]) != sorted(truth):
            raise ValueError("Undirected pair counts must match the two truth regions")
        good = []
        for predicted, target, overlap in pair["counts"]:
            if (any(type(value) is not int for value in (predicted, target, overlap))
                    or predicted < 300 or not 0 <= overlap <= min(predicted, target)):
                raise ValueError("Invalid mask pixel counts")
            union = predicted + target - overlap
            good.append(bool(target) and Fraction(overlap, union) >= Fraction(3, 10)
                        and Fraction(overlap, target) >= Fraction(1, 2))
        useful |= all(good)
    return useful


def review(path):
    plan, journal = path / "freeze.json", path / "journal.jsonl"
    protocol = json.loads(plan.read_bytes())
    receipt = json.loads((path / "aggregate.json").read_bytes())
    if (protocol["schema"] != SCHEMA or receipt["schema"] != SCHEMA or receipt["freeze_sha256"] != sha(plan)
            or receipt["journal_sha256"] != sha(journal) or tuple(protocol["formats"]) != FORMATS
            or tuple(protocol["positive"]) != POSITIVE or tuple(protocol["controls"]) != CONTROLS):
        raise ValueError("The receipts do not bind the expected protocol and rows")
    sources = {row["slot"]: row["cohort"] for row in protocol["sources"] if "sha256" in row}
    remaining = set(itertools.product(sources, FORMATS, (*POSITIVE, *CONTROLS)))
    totals = {fmt: {role: Counter() for role in (*POSITIVE, *CONTROLS)} for fmt in FORMATS}
    by_cohort = {cohort: {fmt: Counter() for fmt in FORMATS} for cohort in set(sources.values())}
    gains = {fmt: set() for fmt in FORMATS}
    rows = pair_sides = 0
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        slot, fmt, role = (row[field] for field in ("slot", "format", "role"))
        if ((slot, fmt, role) not in remaining or sources[slot] != row["cohort"]
                or len(row["truth_sha256"]) != 2 or len(row["truth_pixels"]) != 2
                or not all(SHA.fullmatch(value) for value in (row["input_sha256"], *row["truth_sha256"]))
                or any(type(value) is not int or value < 0 for value in row["truth_pixels"])):
            raise ValueError("Unexpected or duplicate view")
        remaining.remove((slot, fmt, role))
        assessed = row["status"] == "assessed"
        old, extra, proposals = row["incumbent"], row["added"], row["proposals"]
        indices = [pair["proposal"] for pair in old + extra]
        if (row["status"] not in {"assessed", "unassessed_match_budget"} or type(proposals) is not int
                or not 0 <= proposals <= 3 or (not assessed and (indices or proposals))
                or any(type(index) is not int or not 0 <= index < proposals for index in indices)
                or len(set(indices)) != len(indices)
                or any(pair["correction"] is not None for pair in old)
                or any(len(pair["correction"]) != 2 or not .75 <= pair["correction"][0] <= 1.35
                       or abs(pair["correction"][1]) > 38 for pair in extra)):
            raise ValueError("The additive cue must preserve all measured incumbent proposals")
        positive = role in POSITIVE
        if any(bool(value) != positive for value in row["truth_pixels"]):
            raise ValueError("Positive and control views must keep distinct truth labels")
        first = validate_pairs(old, row["truth_pixels"])
        second = validate_pairs(extra, row["truth_pixels"])
        totals[fmt][role].update({"views": 1, "assessed": int(assessed), "proposals": int(proposals > 0),
                                  "incumbent_flag": int(bool(old)), "added_pair": int(bool(extra)),
                                  "additional_flag": int(bool(extra) and not old), "incumbent_useful": int(first),
                                  "combined_useful": int(first or second)})
        group = by_cohort[row["cohort"]][fmt]
        group.update({("positive_views" if positive else "ordinary_views"): 1, "assessed": int(assessed)})
        if positive:
            group.update({"incumbent_useful": int(first), "combined_useful": int(first or second)})
            if role.endswith("_tone") and second and not first:
                gains[fmt].add(slot)
        else:
            group.update({"incumbent_ordinary_flags": int(bool(old)), "additional_ordinary_pairs": int(bool(extra))})
        rows += 1
        pair_sides += 2 * len(indices)
    gain_counts = {fmt: len(slots) for fmt, slots in gains.items()}
    decision = {fmt: (gain_counts[fmt] >= 3 and totals[fmt]["published"]["added_pair"] == 0
                      and sum(totals[fmt][role]["added_pair"] for role in CONTROLS) <= 1) for fmt in FORMATS}
    if (remaining or rows != receipt["views"] or totals != receipt["results"] or decision != receipt["advances"]
            or gain_counts != receipt["gained_groups"]):
        raise ValueError("The comparison aggregate differs from the independently computed results")
    return {"verified_views": rows, "verified_pair_sides": pair_sides, "cohort_results": by_cohort,
            "advances": decision, "gained_groups": gain_counts}


def review_stress(path):
    plan_path, screen_path = path / "freeze.json", path / "screen.json"
    plan, screen = (json.loads(item.read_bytes()) for item in (plan_path, screen_path))
    comparison = json.loads((path / "comparison.json").read_bytes())
    if (screen["freeze_sha256"] != sha(plan_path) or comparison["screen_sha256"] != sha(screen_path)
            or not all(value["schema"] == "chipmunk.image-copy-tone-ordinary-stress/1"
                       for value in (plan, screen, comparison))):
        raise ValueError("Ordinary stress receipts are unbound")
    formats = ("rgb", "jpeg75")
    rows = len(plan["roster"])
    shortlist = screen["shortlist"]
    records = comparison["records"]
    if tuple(plan["deliveries"]) != formats or len(shortlist) != len(records):
        raise ValueError("The comparison must cover both deliveries and all shortlisted rows")
    counts = {fmt: Counter() for fmt in formats}
    seen = set()
    pairs = fallback_proposals = 0
    for before, after in zip(shortlist, records, strict=True):
        fmt, slot = before["format"], before["slot"]
        old, new = after["incumbent"], after["added"]
        indices = [row["proposal"] for row in old + new]
        if (fmt not in formats or not 0 <= slot < rows or (fmt, slot) in seen
                or any(after[key] != value for key, value in before.items()) or not SHA.fullmatch(before["input_sha256"])
                or not 1 <= before["proposals"] <= 3 or after["status"] != "assessed"
                or len(set(indices)) != len(indices) or any(not 0 <= value < before["proposals"] for value in indices)
                or validate_pairs(old, [0, 0]) or validate_pairs(new, [0, 0])):
            raise ValueError("Invalid measured ordinary proposal")
        seen.add((fmt, slot))
        pairs += before["proposals"]
        fallback_proposals += before["proposals"] - len(old)
        counts[fmt].update({"shortlisted": 1, "incumbent_flag": int(bool(old)), "additional_pair": int(bool(new)),
                            "additional_flag": int(bool(new) and not old)})
    for fmt in formats:
        sparse = screen["counts"][fmt]
        if (sparse["views"] != rows or sparse.get("assessed", 0) + sparse.get("unassessed_match_budget", 0) != rows
                or counts[fmt]["shortlisted"] != sparse.get("ordinary_pair_proposals", 0)
                or counts[fmt] != Counter(comparison["counts"][fmt])):
            raise ValueError("Independently counted shortlist differs from recorded totals")
    return {"screened_views": rows * len(formats), "measured_shortlisted_views": len(records),
            "shortlisted_source_groups": len({slot for _, slot in seen}), "measured_proposals": pairs,
            "fallback_proposals": fallback_proposals, "comparison_counts": counts, "screen_counts": screen["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--ordinary-stress", type=Path)
    args = parser.parse_args()
    result = review(args.receipts)
    if args.ordinary_stress is not None:
        result["ordinary_stress"] = review_stress(args.ordinary_stress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
