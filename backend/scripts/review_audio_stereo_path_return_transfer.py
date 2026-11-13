"""Recompute stereo path-return decisions from the anonymous scalar receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "chipmunk.audio-stereo-path-return/1"
ARMS = ("level_only", "joint")
LEVELS = (None, 5, 0, -5)
NEGATIVES = ("untouched", "room_only", "ordinary_same_path", "gradual_move")
KINDS = (*NEGATIVES, "ordinary_path_change", "synthetic_path_change", "synthetic_same_path")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    body = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def validate(rows, role):
    group_counts = {}
    ids, inputs = set(), set()
    for row in rows:
        group, kind, level, key = (row[field] for field in ("group", "kind", "level_db", "case"))
        require(row["role"] == role and kind in KINDS and level in LEVELS and
                key == f"{group}:{level}:{kind}" and key not in ids, "Unexpected row identity")
        ids.add(key)
        pin = row["input_sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", pin) is not None and pin not in inputs, "Waveform alias")
        inputs.add(pin)
        truth = row["truth"]
        require(truth in ([23200, 38560], [55200, 70560]) if "path" in kind else truth is None, "Edit clock differs")
        measured = row["measured_snr_db"]
        require(measured is None if level is None or kind == "room_only" else abs(measured - level) < .02,
                "Unbound SNR measurement")
        require(bool(row["donor_family"]) == kind.startswith("synthetic"), "Incorrect source label")
        require(set(row["findings"]) == set(ARMS), "Missing comparison route")
        for finding in row["findings"].values():
            if finding is None:
                continue
            start, stop = finding["range_samples"]
            require(9600 <= start < stop <= 86400 and start % 1600 == stop % 1600 == 0
                    and (stop-start) in (11200, 14400, 17600) and math.isfinite(finding["score"]) and finding["score"] >= 0,
                    "Unbound finding clock or response")
        group_counts.setdefault(group, set()).add((kind, level))
    for counts in group_counts.values():
        expected_kinds = {"untouched", "ordinary_same_path", "ordinary_path_change", "gradual_move"}
        if any(kind.startswith("synthetic") for kind, _ in counts):
            expected_kinds |= {"synthetic_same_path", "synthetic_path_change"}
        expected = {(kind, level) for kind in expected_kinds for level in LEVELS}
        expected |= {("room_only", level) for level in LEVELS if level is not None}
        require(expected == counts, "Missing condition-matched control")
    require(len(group_counts) == (3 if role == "development" else 6), "Unexpected source group count")
    return inputs


def points(rows):
    return {arm: max(row["findings"][arm]["score"] for row in rows
                     if row["kind"] in NEGATIVES and row["findings"][arm] is not None) for arm in ARMS}


def location(truth, found):
    a, b = truth
    c, d = found
    hit = max(0, min(b, d) - max(a, c))
    union = b-a + d-c - hit
    return hit * 2 >= b-a and 10 * hit >= 3 * union


def summaries(rows, operating):
    result = {}
    for level in LEVELS:
        result[str(level)] = {}
        for arm in ARMS:
            report = {}
            for kind in KINDS:
                eligible = [row for row in rows if row["kind"] == kind and row["level_db"] == level]
                assessed = [row for row in eligible if row["findings"][arm] is not None]
                flagged = [row for row in assessed if row["findings"][arm]["score"] > operating[arm]]
                supported = [row for row in assessed if row["truth"] is not None
                             and location(row["truth"], row["findings"][arm]["range_samples"])]
                localized = [row for row in flagged if row in supported]
                mae = [tuple(abs(a-b) for a,b in zip(row["truth"], row["findings"][arm]["range_samples"], strict=True))
                       for row in localized]
                report[kind] = {"eligible": len(eligible), "assessed": len(assessed), "flagged": len(flagged),
                                "best_range_at_edit": len(supported), "localized": len(localized),
                                "median_boundary_error_samples": [statistics.median(x) for x in zip(*mae, strict=True)]
                                if mae else None}
            result[str(level)][arm] = report
    return result


def review(receipt):
    lock = receipt["freeze"]
    dev, held = receipt["development"]["rows"], receipt["assessment"]
    require(lock["schema"] == SCHEMA and lock["clock_hz"] == 16000 and lock["waveform_samples"] == 96000
            and lock["levels_db"] == list(LEVELS) and len(dev) == 57 and len(held) == 146,
            "Study scope differs")
    require(not validate(dev, "development") & validate(held, "assessment"), "Source waveform alias across roles")
    # The native roster follows retained group order: held paired, development controls, held controls.
    native_order = held[:108] + dev + held[108:]
    require(digest([{key: value for key, value in row.items() if key != "findings"} for row in native_order]) ==
            lock["delivery_roster_sha256"], "Frozen waveform roster differs")
    operating = points(dev)
    require(operating == receipt["development"]["points"], "Development cutoff differs")
    return {"development_deliveries": len(dev), "assessment_deliveries": len(held),
            "points": operating, "assessment": summaries(held, operating)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    value = review(json.loads(args.receipt.read_bytes()))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
