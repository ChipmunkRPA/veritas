# SPDX-License-Identifier: Apache-2.0
"""Independently replay base and instruction model article passage results."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ARMS = ("base", "instruct", "difference")
GENERATED = {"granite", "olmo"}
SCHEMA = "chipmunk.writing-instruct-base-passage-public/1"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checked_measurements(readings):
    result = {}
    for pin, measurement in readings.items():
        require(sha256(pin) and set(measurement) == {"base_hex", "instruct_hex", "target_tokens"}, "Invalid recorded input")
        base, instruct = (float.fromhex(measurement[key]) for key in ("base_hex", "instruct_hex"))
        require(math.isfinite(base) and math.isfinite(instruct) and max(base, instruct) <= 0
                and type(measurement["target_tokens"]) is int and 0 < measurement["target_tokens"] <= 256,
                "Invalid target likelihood or coverage")
        result[pin] = dict(zip(ARMS, (base, instruct, instruct - base), strict=True))
    return result


def read_window(row, previous_end):
    span = row["range"]
    require(len(span) == 2 and all(type(value) is int for value in span)
            and 0 <= previous_end <= span[0] < span[1], "Invalid passage position")
    require(sha256(row["input_sha256"]), "Invalid passage input digest")
    return span


def checked_overlap(value):
    shared, union, truth = value
    require(all(type(number) is int for number in value) and 0 <= shared <= truth <= union and union > 0,
            "Invalid known-edit overlap")
    return truth > 0 and 2 * shared >= truth and 10 * shared >= 3 * union


def replay(receipt):
    require(receipt["schema"] == SCHEMA and all(sha256(pin) for pin in receipt["bindings"].values()),
            "Invalid article passage commitments")
    lookup = checked_measurements(receipt["readings"])
    select, held = (receipt["cases"][part] for part in ("development", "held"))
    require(Counter(row["role"] for row in select) == {"human": 12, "splice": 7}
            and Counter(row["role"] for row in held) == {"original": 15, "ordinary": 15, "granite": 13, "olmo": 11}
            and len({row["group"] for row in select}) == 12 and len({row["group"] for row in held}) == 15,
            "Article groups or case counts changed")
    input_sets = []
    for cases in (select, held):
        inputs = set()
        for case in cases:
            require(sha256(case["text_sha256"]) and case["windows"], "Missing article commitment or target")
            previous_end = 0
            for window in case["windows"]:
                previous_end = read_window(window, previous_end)[1]
                inputs.add(window["input_sha256"])
        input_sets.append(inputs)
    require(not input_sets[0] & input_sets[1] and set(lookup) == input_sets[0] | input_sets[1] and len(lookup) == 449,
            "Recorded inputs differ from the disjoint planned cases")

    points, outcomes = {}, {}
    for arm in ARMS:
        group_maximum = {}
        for case in select:
            group = case["group"]
            score = max(lookup[w["input_sha256"]][arm] for w in case["windows"])
            group_maximum[group] = max(group_maximum.get(group, -math.inf), score)
        point = sorted(group_maximum.values(), reverse=True)[2]
        points[arm] = point.hex()
        counts, genuine, incumbent_genuine, gains, families = Counter(), set(), set(), set(), set()
        for case in held:
            group, role = case["group"], case["role"]
            ranked = sorted(((lookup[w["input_sha256"]][arm], i) for i, w in enumerate(case["windows"])), reverse=True)
            score, index = ranked[0]
            unique = len(ranked) == 1 or ranked[1][0] != score
            flagged = unique and score > point
            overlaps = [checked_overlap(w["overlap"]) for w in case["windows"]]
            prior = [checked_overlap(window) for window in case["incumbent_overlaps"]]
            if role in GENERATED:
                require(case["truth"] in [w["range"] for w in case["windows"]], "A known edit was not measured")
                top_useful = unique and overlaps[index]
                found = flagged and top_useful
                added = found and not any(prior)
                counts.update({f"known_{role}": 1, "current_useful": any(prior), "candidate_useful": found,
                               "top_useful_before_point": top_useful, "added_useful": added,
                               "wrong_location": flagged and not top_useful})
                if added:
                    gains.add(group)
                    families.add(role)
            else:
                counts.update(genuine_candidate_files=flagged, genuine_current_files=bool(prior))
                if flagged:
                    genuine.add(group)
                if prior:
                    incumbent_genuine.add(group)
        new = genuine - incumbent_genuine
        outcomes[arm] = dict(counts) | {"added_useful_groups": len(gains), "added_families": sorted(families),
                                      "genuine_candidate_groups": len(genuine), "genuine_current_groups": len(incumbent_genuine),
                                      "new_genuine_groups": len(new), "passed": len(gains) >= 2 and len(families) == 2 and len(new) <= 1}
    return {"points_hex": points, "outcomes": outcomes, "exact_inputs": len(lookup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    result = replay(receipt)
    require(result == receipt["summary"], "Published comparison differs from replay")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
