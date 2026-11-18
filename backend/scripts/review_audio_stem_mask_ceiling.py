# SPDX-License-Identifier: MIT
"""Recount matched speech-mask measurements without recordings or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

SCHEMA = "chipmunk.audio-stem-mask-ceiling/1"
METHODS = ("recurrent", "recurrent_cap", "known_stems_cap", "known_stems_matched")
CAP_DB = -1
SEVERE_SNR_DB = -6
CONCURRENT_FRACTION = .75
EPSILON = 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(rows):
    output = {}
    for i, method in enumerate(METHODS):
        gains = [row["component_db"][i] for row in rows]
        deltas = [foreground - background for foreground, background in gains]
        remaining = [row["input_snr_db"] + delta for row, delta in zip(rows, deltas, strict=True)]
        groups = defaultdict(list)
        for row, delta in zip(rows, deltas, strict=True):
            groups[row["fixture"]].append(delta)
        group_deltas = [median(group) for group in groups.values()]
        output[method] = {
            "foreground_db": round(median(foreground for foreground, _ in gains), 5),
            "interference_db": round(median(background for _, background in gains), 5),
            "snr_change_db": round(median(deltas), 5),
            "resulting_snr_db": round(median(remaining), 5),
            "improved_intervals": sum(delta > 0 for delta in deltas),
            "positive_output_snr": sum(value > 0 for value in remaining),
            "improved_groups": sum(delta > 0 for delta in group_deltas),
            "source_group_median_snr_change_db": round(median(group_deltas), 5),
        }
    return {"intervals": len(rows), "input_snr_db": round(median(row["input_snr_db"] for row in rows), 5),
            "methods": output}


def review(receipt, prior):
    require(receipt["schema"] == SCHEMA, "Unknown diagnostic schema")
    require(prior["schema"] == "chipmunk.audio-recurrent-spectrum-opus/1", "Unknown parent capsule")
    source = {(row["id"], *span): powers for row in prior["confirmation"] if row["kind"] == "edited"
              for span, powers in zip(row["truth"], row["component_db"][1:], strict=True)}
    fixtures = {row["id"]: row["family"] for row in prior["confirmation"] if row["kind"] == "edited"}
    rows = receipt["rows"]
    require(len(rows) == len(source) == 75 and len(fixtures) == 11 and len(set(fixtures.values())) == 6,
            "The held study does not cover the full source panel")
    seen = set()
    for row in rows:
        key = (row["fixture"], *row["samples"])
        require(key in source and key not in seen, "An edit clock differs from the parent evidence")
        seen.add(key)
        require(math.isfinite(row["input_snr_db"]) and 0 <= row["concurrent_activity"] <= 1,
                "Invalid input component measurement")
        gains = row["component_db"]
        require(len(gains) == len(METHODS) and all(len(pair) == 2 and all(map(math.isfinite, pair)) for pair in gains),
                "Invalid output component measurement")
        original, capped, oracle, matched = gains
        require(all(abs(a - b) <= EPSILON for a, b in zip(original, source[key], strict=True)),
                "The recurrent sweep differs from the parent measurements")
        require(capped[0] >= CAP_DB - EPSILON and oracle[0] >= CAP_DB - EPSILON
                and matched[0] >= min(0, original[0]) - EPSILON, "The strength bound loses too much foreground")
        if original[0] >= CAP_DB:
            require(all(abs(a - b) <= EPSILON for a, b in zip(capped, original, strict=True)),
                    "Capping an already qualified mask changes its response")
    strata = {
        "all": rows,
        "severe": [row for row in rows if row["input_snr_db"] <= SEVERE_SNR_DB],
        "less_severe": [row for row in rows if row["input_snr_db"] > SEVERE_SNR_DB],
        "concurrent": [row for row in rows if row["concurrent_activity"] >= CONCURRENT_FRACTION],
    }
    return {"source_groups": len(fixtures), "generator_families": len(set(fixtures.values())),
            "comparisons": {name: summarize(values) for name, values in strata.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    contents = args.receipt.read_bytes()
    original = args.parent.read_bytes()
    receipt = json.loads(gzip.decompress(contents))
    require(hashlib.sha256(original).hexdigest() == receipt["parent_capsule_sha256"], "Parent capsule has changed")
    result = review(receipt, json.loads(gzip.decompress(original)))
    require(result == receipt["expected"], "Stored summary differs from the verified readings")
    print(json.dumps({"capsule_sha256": hashlib.sha256(contents).hexdigest(), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
