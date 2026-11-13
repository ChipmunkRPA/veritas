# SPDX-License-Identifier: Apache-2.0
"""Recount frozen cepstral localization, genuine flags and coverage without models."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-cepstral-partial-transfer/1"
ARMS = ("linear", "mel", "combined")
BASE = "incumbent_frames"
CANDIDATE = "periodic_excitation"
SHORT = "incumbent_two_second"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def useful(span, truth):
    if span is None or truth is None:
        return False
    a, b = span
    c, d = truth
    overlap = max(0, min(b, d) - max(a, c))
    return 2 * overlap >= d - c and 10 * overlap >= 3 * (b - a + d - c - overlap)


def counts(pairs, arm):
    flagged = [(row, item) for row, item in pairs if item["flag"]]
    false = [row for row, _ in flagged if row["truth"] is None]
    return {
        "synthetic": sum(row["truth"] is not None for row, _ in pairs),
        "genuine": sum(row["truth"] is None for row, _ in pairs),
        "true_flags": sum(row["truth"] is not None for row, _ in flagged),
        "genuine_false_flags": len(false),
        "genuine_false_groups": len({row["group"] for row in false}),
        "useful": sum(useful(item["range"], row["truth"]) for row, item in flagged),
        "threshold_free_useful": sum(useful(item["range"], row["truth"]) for row, item in pairs),
        "no_measurement": sum(item["range"] is None for _, item in pairs),
        "background_only_flags": sum(row["arm"] == "background_only" for row in false),
    }


def review(receipt):
    require(receipt["schema"] == SCHEMA and set(receipt["outcomes"]) == set(ARMS), "Unknown method or arm")
    rows = receipt["baseline"]
    require(len(rows) == len({(row["panel"], row["id"]) for row in rows}) == 216, "Case roster differs")
    roles = Counter((row["panel"], row["role"]) for row in rows)
    require(
        roles
        == {
            ("matched", "selection"): 64,
            ("matched", "confirmation"): 104,
            ("meeting", "selection"): 24,
            ("meeting", "confirmation"): 24,
        },
        "Case roles differ",
    )
    result = {}
    for arm in ARMS:
        point = receipt["points"][arm]
        selection = receipt["selection"][arm]
        require(
            len(selection) == 46
            and all(value is None or math.isfinite(value) for value in selection)
            and point == max(value for value in selection if value is not None),
            "Frozen genuine point differs",
        )
        readings = receipt["outcomes"][arm]
        cells = defaultdict(lambda: defaultdict(list))
        for row, check in zip(rows, readings, strict=True):
            size = 96000 if row["panel"] == "meeting" else 64000
            require(re.fullmatch("[0-9a-f]{64}", row["pcm_sha256"]), "Missing exact source commitment")
            require(
                row["truth"] is None or 0 <= row["truth"][0] < row["truth"][1] <= size,
                "Truth differs from source clock",
            )
            for key, item in (row["checks"] | {CANDIDATE: check}).items():
                span = item["range"]
                if span is not None:
                    a, b = span
                    require(
                        type(a) is type(b) is int
                        and 0 <= a < b <= size
                        and a % 3200 == 0
                        and b - a == (32000 if key == SHORT else 9600),
                        "Measured range is outside the fixed sample grid",
                    )
                require(
                    item["located"] is useful(span, row["truth"]),
                    "Location decision does not match sample bounds",
                )
                if key == CANDIDATE:
                    value = item["score"]
                    require(
                        (value is None) == (span is None)
                        and (value is None or math.isfinite(value))
                        and item["flag"] is (value is not None and value > point),
                        "Frozen decision differs",
                    )
                label = "/".join(row[k] for k in ("panel", "role", "condition"))
                cells[label][key].append((row, item))
        selected = {
            label: {key: counts(pairs, key) for key, pairs in cell.items()} for label, cell in cells.items()
        }
        for label, cell in selected.items():
            require(
                all(receipt["summary"][arm]["cells"][label][key] == value for key, value in cell.items()),
                "Reported cell differs",
            )
        result[arm] = selected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
