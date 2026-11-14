"""Recalibrate an exact writing-pattern ratio with held source works."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import writing_missed_passage_transfer as causal

PANEL = causal.pool.PANEL_DIR / "writing-missed-passage-transfer-2026-11-14.json"
SCHEMA = "chipmunk.writing-ratio-context-review/1"
ARMS = ("absolute", "local_median", "both_flanks", "length_bins")
METHOD = {
    "scope": "Use the three interior sentences and require exact ratio coverage for each candidate as "
    "well as both immediate neighbors. Keep the primary classifier contexts as an independent proxy.",
    "arms": "Compare the absolute ratio, the difference from the median of all other available sentences, "
    "and the minimum difference from the two neighbors. The last arm replaces a fixed absolute ratio "
    "with a length-conditioned empirical lower-tail rank.",
    "length": "Build lower-tail references from genuine fit source works. In each token bin 12–20, 21–35, "
    "and 36 or more use only the single lowest eligible ratio per source work. Require ten works per bin. "
    "Use (1 + count at or below the submitted ratio) / (1 + works) for the rank. Lower is more unusual.",
    "calibration": "Use genuine calibration files with no primary proxy flag. Compare the maximum "
    "response and the second highest. Use strict exceedance and abstain on ties for all arms.",
    "baseline": "Keep the primary proxy's original locations. Also count the original absolute ratio marks "
    "on every exact sentence at its existing .9505 point. Evaluate corrected flags and misplaced ranges.",
    "decision": "An optional ratio policy needs three additional useful locations on distinct held works "
    "across both generator families, at most two new genuine flags and at most two misplaced locations "
    "on files previously free of any flag. The same frozen arm must add a useful location with at most "
    "one new genuine flag on the alternate-generator transfer. All data already had project exposure.",
}


def candidates(row):
    ratios = causal.ratio_values(row)
    for i in (1, 2, 3):
        left, target, right = ratios[i - 1:i + 2]
        if None in (left, target, right):
            continue
        yield i, target, left, right, median(q for j, q in enumerate(ratios) if j != i and q is not None)


def token_bin(count):
    if count < 21:
        return "12-20"
    return "21-35" if count < 36 else "36+"


def length_references(rows):
    references = {}
    for row in rows:
        if row["role"] != "fit" or row["truth"] is not None:
            continue
        for i, target, *_ in candidates(row):
            domain = token_bin(row["ratio"][i]["tokens"])
            works = references.setdefault(domain, {})
            works[row["group"]] = min(target, works.get(row["group"], math.inf))
    return {domain: sorted(works.values()) for domain, works in references.items() if len(works) >= 10}


def best(row, arm, lengths):
    values = []
    for i, target, left, right, other in candidates(row):
        if arm == "length_bins":
            category = token_bin(row["ratio"][i]["tokens"])
            if not (reference := lengths.get(category)):
                continue
            score = -(1 + sum(value <= target for value in reference)) / (1 + len(reference))
        else:
            score = {"absolute": causal.POINT - target, "local_median": other - target,
                     "both_flanks": min(left, right) - target}[arm]
        values.append((score, i))
    values.sort(reverse=True)
    return values[0] if values and (len(values) == 1 or values[0][0] > values[1][0]) else None


def calibrate(rows):
    references = length_references(rows)
    controls = [row for row in rows if row["role"] == "calibration" and row["truth"] is None
                and not causal.pool.baseline(row)]
    thresholds = {}
    for arm in ARMS:
        values = sorted(winner[0] for row in controls if (winner := best(row, arm, references)) is not None)
        if len(values) < 8:
            raise ValueError("Insufficient ordinary calibration works")
        thresholds[arm] = {"most_conservative": values[-1], "one_control": values[-2], "controls": len(values)}
    return {"length_references": references, "thresholds": thresholds}


def row_summary(rows, frozen):
    counts = Counter()
    gains = {}
    for row in rows:
        truth = row["truth"]
        baseline = causal.pool.baseline(row)
        current = causal.measured_spans(row)
        found = any(causal.pool.useful(row, span) for span in baseline)
        counts["edited" if truth else "ordinary"] += 1
        counts["classifier/useful"] += found
        counts["classifier/ordinary_flags"] += truth is None and bool(baseline)
        counts["current_union/useful"] += any(causal.pool.useful(row, span) for span in current)
        counts["current_union/ordinary_flags"] += truth is None and bool(current)
        for arm in ARMS:
            winner = best(row, arm, frozen["length_references"])
            if winner is None:
                counts[f"{arm}/no_unique_candidate"] += 1
                continue
            for point in ("most_conservative", "one_control"):
                if winner[0] <= frozen["thresholds"][arm][point]:
                    continue
                selected = row["spans"][winner[1]]
                key = f"{arm}/{point}"
                counts[f"{key}/selected"] += 1
                if not truth:
                    counts[f"{key}/ordinary_flags"] += 1
                    counts[f"{key}/added_ordinary_flags"] += not baseline
                elif selected == truth:
                    counts[f"{key}/exact"] += 1
                    if not found:
                        counts[f"{key}/added_useful"] += 1
                        counts[f"{key}/gain/{row['family']}"] += 1
                        gains.setdefault(key, set()).add(row["group"])
                else:
                    counts[f"{key}/wrong"] += 1
                    counts[f"{key}/added_wrong"] += not baseline
    for key, works in gains.items():
        counts[f"{key}/gain_works"] = len(works)
    return dict(counts)


def summarize(rows, frozen):
    results = {}
    for role in ("held", "transfer"):
        results[role] = {
            domain: row_summary([r for r in rows if r["role"] == role and domain in (r["domain"], "all")], frozen)
            for domain in (*causal.pool.DOMAINS, "all")
        }
    held, transfer = (results[r]["all"] for r in ("held", "transfer"))
    results["advance"] = {}
    for arm in ARMS:
        for point in ("most_conservative", "one_control"):
            key = f"{arm}/{point}"
            results["advance"][key] = (
                held.get(f"{key}/gain_works", 0) >= 3
                and all(held.get(f"{key}/gain/{family}", 0) for family in causal.pool.GENERATORS)
                and held.get(f"{key}/added_ordinary_flags", 0) <= 2
                and held.get(f"{key}/added_wrong", 0) <= 2
                and transfer.get(f"{key}/added_useful", 0) >= 1
                and transfer.get(f"{key}/added_ordinary_flags", 0) <= 1
            )
    return results


def report():
    _, rows = causal.load_numeric(PANEL)
    parameters = calibrate(rows)
    development_receipt = {
        "schema": SCHEMA, "method": METHOD, "numeric_input_sha256": causal.pool.file_hash(PANEL),
        "runner_sha256": causal.pool.file_hash(Path(__file__)), "parameters": parameters,
    }
    return development_receipt | {"results": summarize(rows, parameters)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    expected = report()
    if args.mode == "run":
        causal.pool.write(args.path, expected)
    elif json.loads(args.path.read_text()) != expected:
        raise ValueError("The ratio review no longer reproduces")
    print(json.dumps(expected["results"], indent=2))


if __name__ == "__main__":
    main()
