# SPDX-License-Identifier: Apache-2.0
"""Replay source-aligned writing compression decisions without model weights."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

METHODS = ("raw_ratio", "word_order_gain", "conditional_word_order_loss")
ORDINARY = frozenset({"authentic", "within_source_exchange", "same_publisher_other_article"})


def _candidates(case, method):
    selected = [(r["score"], tuple(r["compact"])) for r in case["candidates"] if r["method"] == method]
    if (any(not math.isfinite(score) or not 0 <= start < end for score, (start, end) in selected)
            or len(set(bounds for _, bounds in selected)) != len(selected)):
        raise ValueError("Invalid candidate positions or scores")
    return selected


def _calibrate(cases, method):
    maxima = {}
    for row in cases:
        if row["truth"] is not None or row["role"] not in ORDINARY:
            raise ValueError("Only ordinary files are permitted in calibration")
        choices = _candidates(row, method)
        if choices:
            maxima[row["id"]] = max(maxima.get(row["id"], -math.inf), max(score for score, _ in choices))
    if len(maxima) < 8:
        raise ValueError("Not enough measured calibration groups")
    return sorted(maxima.values(), reverse=True)[1]


def _useful(bounds, truth):
    if truth is None:
        return False
    start, end = bounds
    left, right = truth
    overlap = max(0, min(end, right) - max(start, left))
    return 2 * overlap >= right - left and 10 * overlap >= 3 * (max(end, right) - min(start, left))


def _record(counter, role, selection, truth):
    counter[role, "files"] += 1
    counter[role, "flagged"] += bool(selection)
    counter[role, "localized"] += any(_useful(span, truth) for span in selection)
    counter[role, "misplaced_ranges"] += sum(not _useful(span, truth) for span in selection) if truth else 0


def _json_counts(result):
    return {role: {metric: count for (name, metric), count in result.items() if name == role}
            for role, _ in result}


def review(receipt):
    frozen, held = receipt["frozen"], receipt["held"]
    calibration = frozen["calibration"]
    thresholds = {method: _calibrate(calibration, method) for method in METHODS}
    if thresholds != frozen["thresholds"]:
        raise ValueError("Operating points do not match the frozen calibration")
    calibration_groups = {row["id"] for row in calibration}
    held_keys = {(row["id"], row["role"]) for row in held}
    if len(held_keys) != len(held) or any(group in calibration_groups for group, _ in held_keys):
        raise ValueError("Held files overlap calibration groups")

    alone = {method: Counter() for method in METHODS}
    combined = {method: Counter() for method in METHODS}
    current = Counter()
    random_expectation = 0.0
    for row in held:
        role, truth = row["role"], row["compact_truth"]
        truth = tuple(truth) if truth else None
        native = [tuple(bounds) for bounds in row["baseline_compact"]]
        _record(current, role, native, truth)
        for method in METHODS:
            options = _candidates(row, method)
            top = max((score for score, _ in options), default=-math.inf)
            top_spans = [span for score, span in options if score == top]
            chosen = top_spans if len(top_spans) == 1 and top > thresholds[method] else []
            _record(alone[method], role, chosen, truth)
            extra = [span for span in chosen if all(span[1] <= left or right <= span[0] for left, right in native)]
            _record(combined[method], role, native + extra, truth)
            if truth:
                alone[method][role, "unique_top_localized"] += len(top_spans) == 1 and top_spans[0] == truth
        if truth:
            candidates = _candidates(row, METHODS[0])
            random_expectation += sum(bounds == truth for _, bounds in candidates) / len(candidates)
    return {"thresholds": thresholds, "current": _json_counts(current),
            "alone": {k: _json_counts(v) for k, v in alone.items()},
            "combined": {k: _json_counts(v) for k, v in combined.items()},
            "expected_uniform_top_localizations": round(random_expectation, 6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as file:
        receipt = json.load(file)
    if receipt["schema"] != "chipmunk.model-free-compression-writing/1":
        raise ValueError("Expected the writing compression receipt")
    computed = review(receipt)
    if computed != receipt["replay"]:
        raise ValueError("Replayed counts differ from the receipt")
    print(json.dumps(computed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
