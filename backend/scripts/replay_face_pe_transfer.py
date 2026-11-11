"""Independently recalculate the published face PE transfer from anonymous clip scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ARMS = ("reference", "clip", "pe")
METHODS = {
    "reference_median": ("reference",),
    "clip_median": ("clip",),
    "pe_median": ("pe",),
    "reference_plus_clip": ("reference", "clip"),
    "reference_plus_pe": ("reference", "pe"),
    "reference_clip_pe": ARMS,
}
COHORTS = ("native_hd", "matched_delivery")
FAMILIES = ("original_videos", "inswapper", "uniface")
STATES = ("flagged", "below", "unassessed")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def upper_point(values):
    return sorted(values)[8]


def normalized_max(scores, arms, scales):
    if any(scores[arm] is None for arm in arms):
        return None
    if len(arms) == 1:
        return scores[arms[0]]
    return max((scores[arm] - scales[arm]["median"]) / (scales[arm]["point"] - scales[arm]["median"])
               for arm in arms)


def validate_roster(cases):
    groups = defaultdict(Counter)
    for case in cases:
        scores = case["scores"]
        require(set(scores) == set(ARMS)
                and all(v is None or (type(v) in (float, int) and math.isfinite(v)) for v in scores.values()),
                "Invalid response vector")
        groups[case["group"]][case["partition"], case["cohort"], case["family"]] += 1
    expected_shapes = Counter({
        frozenset({("development", "development", "original_videos")}): 10,
        frozenset(("cohort_transfer", "native_hd", family) for family in FAMILIES): 8,
        frozenset(("cohort_transfer", "matched_delivery", family) for family in FAMILIES): 12,
        frozenset(("delivery_transfer", cohort, family) for cohort in COHORTS for family in FAMILIES): 8,
    })
    require(all(all(value == 1 for value in shape.values()) for shape in groups.values())
            and Counter(frozenset(shape) for shape in groups.values()) == expected_shapes,
            "Anonymous source groups do not match the published roster")


def replay(fixture):
    cases = fixture["cases"]
    require(fixture["schema"] == "chipmunk.face-pe-transfer-medians/1" and len(cases) == 118,
            "Wrong scalar fixture")
    validate_roster(cases)
    development = [row["scores"] for row in cases if row["partition"] == "development"]
    require(all(None not in score.values() for score in development), "Unassessed development control")
    distributions = {arm: [score[arm] for score in development] for arm in ARMS}
    scales = {arm: {"median": median(values), "point": upper_point(values)} for arm, values in distributions.items()}
    require(all(s["point"] > s["median"] for s in scales.values()), "Invalid development scale")
    points = {name: upper_point([normalized_max(row, arms, scales) for row in development])
              for name, arms in METHODS.items()}
    development_flags = {name: sum(normalized_max(row, arms, scales) > points[name] for row in development)
                         for name, arms in METHODS.items()}
    states = defaultdict(lambda: {m: Counter(dict.fromkeys(STATES, 0)) for m in METHODS})
    transitions = defaultdict(lambda: defaultdict(Counter))
    distinct_flags = defaultdict(lambda: {m: set() for m in METHODS})
    pairs = defaultdict(dict)
    for case in cases:
        if case["partition"] == "development":
            require(case["pair"] is None, "Unexpected development pair")
            continue
        scores = {method: normalized_max(case["scores"], arms, scales) for method, arms in METHODS.items()}
        outcome = {method: "unassessed" if value is None else ("flagged" if value > points[method] else "below")
                   for method, value in scores.items()}
        partition, cohort, family = (case[name] for name in ("partition", "cohort", "family"))
        key = f"{partition}/{cohort}/{family}"
        for method, state in outcome.items():
            states[key][method][state] += 1
            transitions[key][method][f"{outcome['reference_median']} to {state}"] += 1
            if family == "original_videos" and state == "flagged":
                distinct_flags[partition][method].add(case["group"])
        if partition == "delivery_transfer":
            require(type(case["pair"]) is int and cohort not in pairs[case["pair"]], "Duplicate or missing delivery")
            pairs[case["pair"]][cohort] = (case["group"], family, outcome)
        else:
            require(case["pair"] is None, "Unexpected delivery pair")
    switches = defaultdict(lambda: defaultdict(Counter))
    require(len(pairs) == 24, "Incomplete delivery comparison")
    for pair in pairs.values():
        first_group, first_family, native = pair["native_hd"]
        second_group, second_family, matched = pair["matched_delivery"]
        require((first_group, first_family) == (second_group, second_family), "Delivery pair changed source or label")
        for method in METHODS:
            switches[first_family][method][f"{native[method]} to {matched[method]}"] += 1
    return {
        "points": points,
        "scales": scales,
        "development_flags": development_flags,
        "states": states,
        "from_reference": transitions,
        "native_to_matched": switches,
        "distinct_original_groups_flagged": {part: {m: len(g) for m, g in per_method.items()}
                                             for part, per_method in distinct_flags.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("medians", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    fixture_bytes = args.medians.read_bytes()
    expected = json.loads(args.result.read_bytes())
    require(expected["schema"] == "chipmunk.face-pe-transfer-result/1"
            and hashlib.sha256(fixture_bytes).hexdigest() == expected["anonymous_medians_sha256"],
            "Published fixture digest differs")
    actual = replay(json.loads(fixture_bytes))
    for key in ("points", "scales"):
        require(actual[key] == expected["calibration"][key], f"Recomputed {key} differs")
    require(actual["development_flags"] == expected["calibration"]["development_false_flags"],
            "Recomputed development false flags differ")
    for key in ("states", "from_reference", "native_to_matched", "distinct_original_groups_flagged"):
        require(actual[key] == expected[key], f"Recomputed {key} differs")
    print("Reproduced six development operating points and all 108 transfer decisions by cohort and family.")


if __name__ == "__main__":
    main()
