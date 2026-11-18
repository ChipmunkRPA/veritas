# SPDX-License-Identifier: Apache-2.0
"""Recount scalar audio partition readings without models or media."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.review_audio_two_second_opus_transfer import matches, require

SCHEMA = "chipmunk.audio-phase-partition-diagnostic/1"
LEVELS = ("overlap_-5dB", "overlap_-10dB")
ARMS = ("mixture", "projected", "contrast")
RATE = 16000
STEP = 8000
WIDTH = 12000


def serial(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate(row):
    starts = range(0, row["samples"] - WIDTH + 1, STEP)
    require(type(row["samples"]) is int and row["samples"] >= 3 * RATE, "Invalid native sample extent")
    require([window["start"] for window in row["windows"]] == list(starts), "A window escaped the fixed schedule")
    require(all(0 <= a < b <= row["samples"] for a, b in row["truth"]), "A labeled edit escaped the native sample extent")
    require((row["kind"] == "edited") == bool(row["truth"]), "Only edited files should carry edit ranges")
    require(row["role"] in {"selection", "diagnostic"} and row["condition"] in LEVELS and
            not (row["role"] == "selection" and row["kind"] == "edited"), "Unrecognized scoring role")
    require(row["kind"] in {"edited", "background", "ordinary", "ordinary_reprocessed"}, "Unrecognized control role")
    for window in row["windows"]:
        counts, scores = window["speech"], window["scores"]
        require(len(counts) == len(scores) == 3 and all(type(n) is int and 0 <= n <= WIDTH for n in counts),
                "Invalid native speech count")
        expected = [counts[0] * 2 >= WIDTH,
                    min(counts[:2]) * 2 >= WIDTH, min(counts) * 2 >= WIDTH]
        for score, admitted in zip(scores, expected, strict=True):
            require((score is not None) == admitted and (not admitted or
                    type(score) in (int, float) and math.isfinite(score)), "Reading or admission differs")


def pairs(row):
    views = {arm: [] for arm in ARMS}
    for window in row["windows"]:
        original, selected, remainder = window["scores"]
        attribute = (selected is not None and original is not None and selected > original and
                     (remainder is None or selected > remainder))
        views["mixture"].append(original)
        views["projected"].append(selected)
        views["contrast"].append(selected if attribute else None)
    pairs = {arm: {} for arm in ARMS}
    for arm, readings in views.items():
        for index, (left, right) in enumerate(zip(readings, readings[1:])):
            if left is not None and right is not None:
                pairs[arm][index * STEP] = min(left, right)
    return pairs


def operating_points(selection):
    result = {}
    for arm in ARMS:
        top_background = []
        control_groups = {}
        for row, readings in selection:
            if not readings[arm]:
                continue
            strongest = max(readings[arm].values())
            if row["kind"] == "background":
                top_background.append(strongest)
            else:
                group = row["group"]
                control_groups[group] = max(control_groups.get(group, strongest), strongest)
        ordered = sorted(control_groups.values())
        result[arm] = {"strict": max([ordered[-1], *top_background]) if len(ordered) >= 3 else None,
                       "wider": max([ordered[-2], *top_background]) if len(ordered) >= 3 else None,
                       "genuine_groups_assessed": len(ordered), "background_files_assessed": len(top_background)}
    return result


def totals(assessment, points, tier):
    counter = defaultdict(Counter)
    for row, readings in assessment:
        ranges = {}
        for arm, scores in readings.items():
            cutoff = points[arm][tier]
            ranges[arm] = {(start + offset, start + offset + WIDTH)
                           for start, response in scores.items() for offset in (0, STEP)
                           if cutoff is not None and response > cutoff}
        for arm in ("projected", "contrast"):
            ranges[f"union_{arm}"] = ranges["mixture"] | ranges[arm]
        kind = "ordinary" if row["kind"].startswith("ordinary") else row["kind"]
        for arm, selected in ranges.items():
            counts = counter[arm]
            counts[kind] += 1
            counts[f"{kind}_flags"] += bool(selected)
            if row["truth"]:
                located = matches(sorted(selected), row["truth"])
                counts["labels"] += len(row["truth"])
                counts["located"] += len(located)
                counts["wrong_time"] += bool(selected and not located)
    return {key: dict(v) for key, v in counter.items()}


def review(payload):
    require(payload["schema"] == SCHEMA and len(payload["rows"]) == 108, "Unexpected audio comparison cohort")
    rows = payload["rows"]
    require(hashlib.sha256(serial(rows)).hexdigest() == payload["canonical_readings_sha256"],
            "Scalar readings differ from their commitment")
    require(len({row["clip"] for row in rows}) == len(rows), "Duplicate case identity")
    cohorts = defaultdict(list)
    membership = defaultdict(set)
    for row in rows:
        validate(row)
        cohorts[row["condition"], row["role"]].append((row, pairs(row)))
        if row["kind"] != "background":
            membership[row["group"]].add(row["role"])
    require(all(len(roles) == 1 for roles in membership.values()), "A foreground group crosses source roles")
    require(all(sum(len(cohorts[level, role]) for role in ("selection", "diagnostic")) == 54 for level in LEVELS),
            "Incomplete matched condition count")
    points = {level: operating_points(cohorts[level, "selection"]) for level in LEVELS}
    comparisons = {level: {tier: totals(cohorts[level, "diagnostic"], points[level], tier)
                           for tier in ("strict", "wider")} for level in LEVELS}
    require(points == payload["points"] and comparisons == payload["diagnostic"],
            "Reported points or outcomes differ from the independent recount")
    return {"points": points, "diagnostic": comparisons}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    with gzip.open(args.receipt, "rt") as handle:
        result = review(json.load(handle))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
