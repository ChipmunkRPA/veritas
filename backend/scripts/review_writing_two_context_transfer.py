"""Replay sentence scope, position, and ordinary-control outcomes from public measurements."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-two-context-transfer/1"
CANDIDATES = ("stable", "stable_with_margin", "interior_old", "interior_position")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def winner(scores, point):
    ordered = sorted(((score, position) for position, score in scores.items()), reverse=True)
    if not ordered or ordered[0][0] <= point or len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
        return []
    return [ordered[0][1]]


def scores(case):
    units = case["units"]
    require(len(units) == 3 and all(0 <= a < b <= case["characters"] and 0 < words <= b - a
                                    for a, b, words in units)
            and all(a[1] < b[0] for a, b in zip(units, units[1:])), "Invalid source sentence ranges")
    eligible = 12 <= units[1][2] <= 65 and all(7 <= units[i][2] <= 80 for i in (0, 2))
    readings = case["readings"]
    labels = {"full0", "full1", "full2"} | ({"left1", "right1"} if eligible else set())
    require(set(readings) == labels, "Measured input scope differs")
    for row in readings.values():
        require(math.isfinite(row["value"]) and math.isfinite(row["incumbent"]) and 0 <= row["incumbent"] <= 1
                and row["center_tokens"] > 0 and row["context_tokens"] > 0
                and row["center_tokens"] + row["context_tokens"] <= 512
                and all(SHA256.fullmatch(row[key]) for key in ("input_sha256", "input_ids_sha256")), "Invalid native response")
    original = {i: readings[f"full{i}"]["value"] for i in range(len(units))}
    if eligible:
        stable = {1: min(readings[label]["value"] for label in ("left1", "right1"))}
        interior = {1: original[1]}
        position = {1: original[1] - max(original[0], original[2])}
    else:
        stable = interior = position = {}
    return {"prior": original, "stable": stable, "stable_with_margin": stable,
            "interior_old": interior, "interior_position": position}


def points(calibration):
    require(len(calibration) == 50 and {row["role"] for row in calibration} == {"human", "splice"},
            "Unexpected genuine calibration cases")
    maxima = {arm: {} for arm in CANDIDATES}
    keys = set()
    for case in calibration:
        name = (case["publisher"], case["group"])
        key = (*name, case["role"])
        require(key not in keys, "Duplicate calibration case")
        keys.add(key)
        for arm, values in scores(case).items():
            if arm in maxima and values:
                maxima[arm][name] = max(maxima[arm].get(name, -math.inf), *values.values())
    require(len({key[:2] for key in keys}) == 25 and all(len(groups) == 25 for groups in maxima.values()),
            "Calibration source group count differs")
    return {arm: sorted(values.values(), reverse=True)[1] for arm, values in maxima.items()}


def useful(target, start, end):
    a, b = target
    shared = max(0, min(end, b) - max(start, a))
    return 2 * shared >= b - a and 10 * shared >= 3 * (end - start + b - a - shared)


def outcome(case, levels, prior_point):
    ranked = scores(case)
    selected = {arm: winner(ranked[arm], level) for arm, level in levels.items()}
    selected["prior"] = winner(ranked["prior"], prior_point)
    selected["current"] = [i for i in range(3) if case["readings"][f"full{i}"]["incumbent"] >= 0.5]
    require(selected == case["selected"], "Recorded source locations differ")
    target = case["truth"]
    spans = case["nonspace"]
    require(len(spans) == 3 and all(0 <= a < b <= case["characters"] for a, b in spans)
            and all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "Invalid non-whitespace positions")
    if target is not None:
        require(target == case["units"][1][:2] and case["role"] in ("granite", "olmo"), "Known edit span differs")
    else:
        require(case["role"] in ("human", "ordinary"), "Generated case has no known span")
    result = {}
    for arm, indices in selected.items():
        ranges = [(spans[max(0, i - 1)][0], spans[min(i + 1, 2)][1]) if arm == "current" else spans[i] for i in indices]
        result[arm] = bool(target is not None and any(useful(spans[1], *location) for location in ranges))
    require(result == case["useful"], "Recorded source overlap differs")
    return selected, result


def review(receipt):
    require(receipt["schema"] == SCHEMA, "Unknown writing receipt")
    levels = points(receipt["calibration"])
    require(levels == receipt["points"] and receipt["prior_point"] == 0.8601455351555256, "Frozen review points differ")
    fresh = receipt["held"]
    require(Counter(row["role"] for row in fresh) == {"human": 9, "ordinary": 9, "granite": 9, "olmo": 2}
            and len({(row["group"], row["role"]) for row in fresh}) == len(fresh), "Fresh held roster differs")
    source_ids = {row["group"] for row in receipt["publisher_sources"]}
    require(len(source_ids) == 9 and all(row["group"] in source_ids and row["publisher"] == "f1000" for row in fresh)
            and not {row["text_sha256"] for row in receipt["calibration"]} & {row["text_sha256"] for row in fresh},
            "Unexpected overlap or publisher roster")
    results = [(row, *outcome(row, levels, receipt["prior_point"])) for row in fresh]
    baseline = {row["group"] for row, selection, _ in results if row["truth"] is None and selection["current"]}
    report = {}
    for arm in ("current", "prior", *CANDIDATES):
        flags = {row["group"] for row, selection, _ in results if row["truth"] is None and selection[arm]}
        report[arm] = {
            "ordinary_groups_flagged": len(flags),
            "additional_ordinary_groups": len(flags - baseline),
            "ordinary_revisions_flagged": sum(bool(selection[arm]) for row, selection, _ in results if row["role"] == "ordinary"),
            "originals_flagged": sum(bool(selection[arm]) for row, selection, _ in results if row["role"] == "human"),
            "useful_edits": sum(local[arm] for _, _, local in results),
            "additional_edits": sum(local[arm] and not local["current"] for _, _, local in results),
            "useful_olmo": sum(local[arm] for row, _, local in results if row["role"] == "olmo"),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        report = review(json.loads(args.receipt.read_bytes()))
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Review failed. {error}") from None
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
