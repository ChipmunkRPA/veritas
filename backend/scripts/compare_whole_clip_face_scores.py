"""Freeze and independently review paired six-frame face-swap score observations.

This tool consumes saved native-frame receipts. It performs no inference and
publishes only aggregate counts. A fit uses the genuine development clips only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

SCHEMA = "chipmunk.video-face-pooling/1"
FRACTION = 0.90
ARMS = ("reference_median", "candidate_median", "paired_max_median", "candidate_top_two")
FAMILIES = {"original_videos": False, "inswapper": True, "uniface": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _summaries(receipt):
    samples = receipt["samples"]
    require(len(samples) == 6 and len({row["index"] for row in samples}) == 6
            and all(a["index"] < b["index"] and a["pts"] < b["pts"] for a, b in zip(samples, samples[1:])),
            "Expected six distinct, ordered native-frame observations")
    scores = [row["score"] for row in samples if row["score"] is not None]
    require(all(type(value) in (float, int) and math.isfinite(value) for value in scores), "Invalid detector response")
    center = median(scores) if len(scores) >= 4 else None
    require(receipt["measured"] == len(scores) and receipt["median"] == center, "Saved pooling differs from its inputs")
    return center, mean(sorted(scores)[-2:]) if center is not None else None


def _frame_binding(receipt):
    return receipt["time_base"], [(s["index"], s["pts"], s["rgb_sha256"], s["box"]) for s in receipt["samples"]]


def observations(plan, reference, candidate):
    expected = [case["id"] for case in plan["cases"]]
    require(len(set(expected)) == len(expected), "Duplicate case IDs")
    require([r["id"] for r in reference] == [r["id"] for r in candidate] == expected,
            "Detector receipts do not cover the same frozen cases in order")
    for case, first, second in zip(plan["cases"], reference, candidate, strict=True):
        require(case["sha256"] == first["source"] == second["source"] and _frame_binding(first) == _frame_binding(second),
                "Paired detector inputs differ in source bytes, frames, source clock or face box")
        reference_median, _ = _summaries(first)
        candidate_median, candidate_top_two = _summaries(second)
        yield case, {"reference_median": reference_median, "candidate_median": candidate_median,
                     "candidate_top_two": candidate_top_two}


def _point(values):
    ordered = sorted(values)
    return ordered[math.ceil(FRACTION * len(ordered)) - 1]


def _scales(controls):
    values = {arm: [row[arm] for row in controls] for arm in ARMS[:2]}
    result = {}
    for arm, scores in values.items():
        center = median(scores)
        high = _point(scores)
        require(high > center, "Paired control tail cannot scale a constant detector")
        result[arm] = {"median": center, "upper_point": high}
    return result


def _combined(scores, scales):
    if any(scores[arm] is None for arm in scales):
        return None
    return max((scores[arm] - limits["median"]) / (limits["upper_point"] - limits["median"])
               for arm, limits in scales.items())


def fit(plan, reference, candidate):
    controls = [(case, scores) for case, scores in observations(plan, reference, candidate)
                if case["role"] == "development" and not case["positive"]]
    require(len(controls) == 10 and len({case["group"] for case, _ in controls}) == 10,
            "Expected ten paired development originals from distinct groups")
    require(all(value is not None for _, row in controls for value in row.values()),
            "Each fixed development original must have both measured inputs")
    raw = [row for _, row in controls]
    scales = _scales(raw)
    points = {arm: _point(row[arm] for row in raw) for arm in raw[0]}
    combined = [_combined(row, scales) for row in raw]
    points["paired_max_median"] = _point(combined)
    alarms = {arm: sum(value > points[arm] for value in
                      (combined if arm == "paired_max_median" else [row[arm] for row in raw])) for arm in ARMS}
    return {"schema": SCHEMA, "strict_exceedance": True, "control_quantile": FRACTION, "development_originals": len(raw),
            "scales": scales, "points": points, "development_false_flags": alarms}


def evaluate(locked, calibration, plan, reference, candidate):
    require(locked["schema"] == SCHEMA and locked["strict_exceedance"] is True
            and locked["control_quantile"] == FRACTION, "Unsupported frozen comparison")
    previous = calibration["cases"]
    require(not ({row["group"] for row in previous} & {row["group"] for row in plan["cases"]})
            and not ({row["sha256"] for row in previous} & {row["sha256"] for row in plan["cases"]}),
            "Confirmation overlaps the previous trial in group or matched file bytes")
    require(all(case["role"] == "confirmation" and type(case["positive"]) is bool for case in plan["cases"]),
            "Unexpected evaluation role or label")
    groups = defaultdict(list)
    for case in plan["cases"]:
        groups[(case.get("cohort", "confirmation"), case["group"])].append(case)
    require(groups and len({group for _, group in groups}) == len(groups)
            and all(len(items) == len(FAMILIES) and {item["family"]: item["positive"] for item in items} == FAMILIES
                    for items in groups.values()), "Each evaluation group needs its own matched original and both swaps")
    require(len({case["sha256"] for case in plan["cases"]}) == len(plan["cases"]), "Duplicate confirmation file bytes")
    disposition = defaultdict(Counter)
    counts = defaultdict(lambda: defaultdict(Counter))
    flagged_groups = defaultdict(lambda: defaultdict(set))
    for case, scores in observations(plan, reference, candidate):
        scores["paired_max_median"] = _combined(scores, locked["scales"])
        cohort = case.get("cohort", "confirmation")
        key = (cohort, case["family"])
        labels = {}
        for arm, score in scores.items():
            status = "unassessed" if score is None else "flagged" if score > locked["points"][arm] else "below"
            labels[arm] = status
            outcome = counts[key][arm]
            outcome.update({"cases": 1, status: 1})
            for label in ("flagged", "below", "unassessed"):
                outcome.setdefault(label, 0)
            if not case["positive"] and status == "flagged":
                flagged_groups[cohort][arm].add(case["group"])
        transition = f"{labels['reference_median']} to {labels['paired_max_median']}"
        disposition[key][transition] += 1
    return {"schema": SCHEMA, "cases": len(plan["cases"]), "support": "six frozen native-frame inputs per case per arm",
            "frozen_points": {k: v for k, v in locked.items() if k != "inputs"},
            "results": {"/".join(key): dict(value) for key, value in counts.items()},
            "paired_transitions": {"/".join(key): value for key, value in disposition.items()},
            "negative_flag_groups": {cohort: {arm: len(flagged_groups[cohort][arm]) for arm in ARMS}
                                     for cohort in sorted({cohort for cohort, _ in groups})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "readout"):
        child = sub.add_parser(command)
        if command == "readout":
            for name in ("lock", "calibration"):
                child.add_argument(name, type=Path)
        for name in ("plan", "reference", "candidate", "output"):
            child.add_argument(name, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("plan", "reference", "candidate")}
    plan, reference, candidate = read(args.plan), rows(args.reference), rows(args.candidate)
    inputs = {name: digest(path) for name, path in paths.items()}
    inputs["comparator_code"] = digest(Path(__file__))
    if args.command == "freeze":
        result = fit(plan, reference, candidate)
    else:
        locked = read(args.lock)
        require(locked["inputs"]["plan"] == digest(args.calibration)
                and locked["inputs"]["comparator_code"] == inputs["comparator_code"],
                "Historical trial or comparison code differs from the frozen fit")
        result = evaluate(locked, read(args.calibration), plan, reference, candidate)
        inputs["calibration_lock"] = digest(args.lock)
    write(args.output, {**result, "inputs": inputs})


if __name__ == "__main__":
    main()
