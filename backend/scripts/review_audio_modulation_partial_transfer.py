# SPDX-License-Identifier: Apache-2.0
"""Reconstruct held localization and false-group counts from the public score ledger."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.audio-modulation-partial-transfer/1"
ARMS = ("incumbent", "modulation", "with_cepstral")
HOLDOUTS = {"matched_dry": (10, 20), "matched_loud_speech": (10, 34), "recorded_room_speech": (12, 40)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(span, truth):
    if span is None or truth is None:
        return False
    a, b = span
    c, d = truth
    shared = max(0, min(b, d) - max(a, c))
    return 2 * shared >= d - c and 10 * shared >= 3 * (b - a + d - c - shared)


def counts(rows, arm, *, cutoff=None):
    flagged = [row for row in rows if (row["checks"][arm]["flag"] if cutoff is None else
                                      row["checks"][arm]["score"] is not None and row["checks"][arm]["score"] > cutoff)]
    genuine = [row for row in rows if row["truth"] is None]
    false = [row for row in flagged if row["truth"] is None]
    return {"synthetic": len(rows) - len(genuine), "genuine": len(genuine), "true_flags": len(flagged) - len(false),
            "genuine_false_flags": len(false), "genuine_false_groups": len({row["group"] for row in false}),
            "useful": sum(useful(row["checks"][arm]["range"], row["truth"]) for row in flagged),
            "threshold_free_useful": sum(useful(row["checks"][arm]["range"], row["truth"]) for row in rows),
            "no_measurement": sum(row["checks"][arm]["range"] is None for row in rows),
            "background_only_flags": sum(row["arm"] == "background_only" for row in false)}


def review(data):
    require(data["schema"] == SCHEMA and re.fullmatch("[0-9a-f]{64}", data["freeze_sha256"]), "Invalid study freeze")
    points = data["points"]
    require(set(points) == set(data["selection_scores"]) == set(ARMS[1:]), "Unexpected operating-point arms")
    for arm, point in points.items():
        scores = data["selection_scores"][arm]
        require(len(scores) == 46 and all(s is None or math.isfinite(s) for s in scores)
                and point == max(s for s in scores if s is not None), "Independent genuine point differs")

    cases = data["cases"]
    require(len(cases) == len({r["pcm_sha256"] for r in cases}) == 126, "Unexpected case roster")
    require(Counter((r["condition"], r["truth"] is not None) for r in cases) == {
        (condition, synthetic): size for condition, sizes in HOLDOUTS.items()
        for synthetic, size in zip((True, False), sizes, strict=True)
    }, "Holdout counts differ")
    output = {}
    for name, expected in HOLDOUTS.items():
        rows = [row for row in cases if row["condition"] == name]
        span_size = 96000 if name == "recorded_room_speech" else 64000
        prior_point = data["incumbent_points"][name]
        for row in rows:
            require(set(row["checks"]) == set(ARMS) and re.fullmatch("[0-9a-f]{64}", row["pcm_sha256"]), "Invalid case binding")
            require((row["truth"] is not None) == (row["arm"] == "synthetic"), "Synthetic label and interval differ")
            if row["truth"] is not None:
                a, b = row["truth"]
                require(type(a) is type(b) is int and 0 <= a < b <= span_size, "Authored edit lies outside the source")
            for arm, item in row["checks"].items():
                score, span = item["score"], item["range"]
                require((score is None) == (span is None) and (score is None or math.isfinite(score)), "Incomplete reading")
                if span is not None:
                    a, b = span
                    width = 32000 if arm == "incumbent" and name == "recorded_room_speech" else 9600
                    require(type(a) is type(b) is int and 0 <= a < b <= span_size and a % 3200 == 0 and b - a == width,
                            "Reading differs from its fixed sample grid")
                require(item["located"] is useful(span, row["truth"]), "Reported overlap differs")
                point = prior_point if arm == "incumbent" else points[arm]
                require(item["flag"] is (score is not None and score > point), "Threshold decision differs")
        output[name] = {arm: counts(rows, arm) for arm in ARMS}
        group_maxima = {arm: {} for arm in ARMS}
        for row in rows:
            if row["truth"] is not None:
                continue
            for arm in ARMS:
                value = row["checks"][arm]["score"]
                if value is not None:
                    group = row["group"]
                    group_maxima[arm][group] = max(value, group_maxima[arm].get(group, -math.inf))
        output[name]["equal_false_groups"] = {
            str(n): {arm: counts(rows, arm, cutoff=sorted(maxima.values(), reverse=True)[n])
                     for arm, maxima in group_maxima.items()} for n in (0, 1, 2)
        }
        require(output[name]["incumbent"]["synthetic"] == expected[0], "Held positive count differs")
    held = output["recorded_room_speech"]
    gates = {}
    for arm in ARMS[1:]:
        added = sorted({row["family"] for row in cases if row["condition"] == "recorded_room_speech"
                        and row["checks"][arm]["flag"] and row["checks"][arm]["located"]
                        and not (row["checks"]["incumbent"]["flag"] and row["checks"]["incumbent"]["located"])})
        candidate, incumbent = held[arm], held["incumbent"]
        gates[arm] = {"added_families": added, "qualifies": len(added) >= 2 and candidate["useful"] >= incumbent["useful"] + 2
                      and candidate["genuine_false_groups"] <= incumbent["genuine_false_groups"] + 1
                      and candidate["background_only_flags"] == 0}
    report = {"conditions": output, "gates": gates}
    require(data["result"] == report, "Summary or advancement gate differs")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))
