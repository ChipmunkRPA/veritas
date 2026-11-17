# SPDX-License-Identifier: Apache-2.0
"""Count actual JPEG regions and source-level false alerts without model imports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.jpeg-paired-boundary/1"
ARMS = ("plain", "paired", "boundary")
QUALITIES = (75, 90)
POSITIVE = frozenset(("sharp", "feather"))
BUDGETS = (0, 1, 2, 4)


def useful(record):
    overlap, truth, picked = (record.get(key, 0) for key in ("intersection", "truth", "selected"))
    return bool(truth and 2 * overlap >= truth and 10 * overlap >= 3 * (picked + truth - overlap))


def incumbent_hit(row):
    return any(x and useful(x) for x in row["current"].values())


def incumbent_flag(row):
    return any(x and x["selected"] > 0 for x in row["current"].values())


def measure(rows, arm, point):
    counts = Counter()
    false_sources = set()
    for row in rows:
        changed = row["role"] in POSITIVE
        original = incumbent_flag(row)
        located = bool(changed and incumbent_hit(row))
        candidate = row["checks"][arm]
        flag = candidate["score"] is not None and candidate["score"] > point
        hit = changed and useful(candidate)
        counts.update(
            files=1,
            changed=changed,
            controls=not changed,
            baseline_correct=located,
            baseline_control_flags=not changed and original,
            top_correct=hit,
            candidate_correct=hit and flag,
            added_correct=hit and flag and not located,
            added_feather=hit and flag and not located and row["role"] == "feather",
            candidate_wrong_location=flag and changed and not hit,
            candidate_control_flags=flag and not changed,
            added_control=flag and not changed and not original,
            ordinary_fill_flags=flag and row["family"] == "classical_fill",
        )
        if flag and not changed and not original:
            false_sources.add(row["source_group"])
    return dict(counts) | {"added_control_groups": len(false_sources)}


def freeze(rows):
    points, chosen = {}, {}
    for quality in QUALITIES:
        items = [row for row in rows if row["quality"] == quality]
        options = []
        points[str(quality)] = {}
        for arm in ARMS:
            maxima = {}
            for row in items:
                score = row["checks"][arm]["score"]
                if row["role"] not in POSITIVE and score is not None and not incumbent_flag(row):
                    key = row["source_group"]
                    maxima[key] = max(score, maxima.get(key, float("-inf")))
            highest = sorted(maxima.values(), reverse=True)
            if len(highest) <= max(BUDGETS):
                raise ValueError("Need separately sourced ordinary calibration groups")
            levels = points[str(quality)][arm] = {}
            for allowance in BUDGETS:
                point = highest[allowance]
                counts = measure(items, arm, point)
                levels[str(allowance)] = {"point": point, "counts": counts}
                order = (
                    counts["added_correct"],
                    counts["added_feather"],
                    -counts["added_control_groups"],
                    -counts["candidate_control_flags"],
                    -counts["candidate_wrong_location"],
                    -allowance,
                )
                options.append((order, {"arm": arm, "allowance": allowance, "point": point}))
        chosen[str(quality)] = max(options, key=lambda entry: entry[0])[1]
    return points, chosen


def summarize(rows, points, chosen):
    result = {}
    for quality in QUALITIES:
        entries = [row for row in rows if row["quality"] == quality]
        for name in ("all", *sorted({row["family"] for row in entries})):
            partition = entries if name == "all" else [row for row in entries if row["family"] == name]
            summary = {}
            for arm, levels in points[str(quality)].items():
                summary[arm] = {
                    level: measure(partition, arm, value["point"]) for level, value in levels.items()
                }
            selection = chosen[str(quality)]
            summary["selected"] = summary[selection["arm"]][str(selection["allowance"])]
            result[f"{quality}/{name}"] = summary
    return result


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def review(path, selection=None):
    data = read(path)
    if data["schema"] != SCHEMA or not data["rows"]:
        raise ValueError("Expected a native JPEG boundary result")
    rows = data["rows"]
    for row in rows:
        if set(row["checks"]) != set(ARMS) or row["quality"] not in QUALITIES:
            raise ValueError("A case is missing one of its matched JPEG comparisons")
        total = row["shape"][0] * row["shape"][1]
        for entry in row["checks"].values():
            selected, truth = entry["selected"], entry["truth"]
            if not 0 <= entry["intersection"] <= min(selected, truth) or max(selected, truth) > total:
                raise ValueError("Invalid exact pixel overlap")
    if selection is None:
        if data["freeze"] is not None or freeze(rows) != (data["points"], data["chosen"]):
            raise ValueError("The development measurements do not reproduce their operating points")
    else:
        review(selection)
        earlier = read(selection)
        if (
            hashlib.sha256(selection.read_bytes()).hexdigest() != data["freeze"]
            or any(earlier[key] != data[key] for key in ("model_pins", "source_method", "points", "chosen"))
            or {row["source_group"] for row in rows} & set(earlier["used_sources"])
        ):
            raise ValueError("The held rows cross the frozen method or development sources")
    result = summarize(rows, data["points"], data["chosen"])
    if result != data["outcomes"]:
        raise ValueError("The individual JPEG cases do not reproduce the summary")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.result, args.freeze), sort_keys=True, indent=2))
