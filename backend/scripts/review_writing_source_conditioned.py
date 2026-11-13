"""Replay a source-conditioned writing comparison without media or model weights."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-source-conditioned/1"
FAMILIES = {"granite", "olmo"}
MODES = {"null", "matched", "mismatched"}
METHODS = ("lexical", "conditional", "source_specific")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def exact_float(value):
    parsed = float.fromhex(value)
    require(math.isfinite(parsed) and value == parsed.hex(), "Invalid exact measurement")
    return parsed


def useful(window):
    intersection, union, truth = window["overlap"]
    require(0 <= intersection <= min(union, truth) and 0 < truth <= union, "Invalid source overlap")
    return 10 * intersection >= 3 * union and 2 * intersection >= truth


def score(row, predictions):
    require(set(row["prompts"]) == MODES, "Missing source conditions")
    count = row["target_tokens"]
    require(type(count) is int and 1 <= count <= 512 and HEX_DIGEST.fullmatch(row["target_sequence_sha256"]),
            "Target token commitment changed")
    loss = {mode: exact_float(value) for mode, value in row["mean_nll"].items()}
    require(set(loss) == MODES and all(value >= 0 for value in loss.values()), "Invalid target likelihood")
    lexical = exact_float(row["lexical"])
    require(0 <= lexical <= 1, "Invalid lexical distance")
    features = {"lexical": lexical, "conditional": loss["null"] - loss["matched"],
                "source_specific": loss["mismatched"] - loss["matched"]}
    focus = row["focus"]
    require(len(focus) == 2 and 0 < focus[0] < focus[1], "Invalid destination range")
    selected = []
    previous = 0
    for window in row["windows"]:
        start, end = window["focus"]
        spans = window["spans"]
        require(0 <= previous <= start < end and [start, end] in spans and len(spans) in {2, 3}
                and all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
                "Unordered source windows")
        previous = end
        response = exact_float(predictions[window["input_sha256"]])
        require(0 <= response <= 1, "Invalid incumbent response")
        hit = useful(window)
        if response >= .5:
            selected.append(hit)
    row["metrics"] = features
    row["baseline_flag"] = bool(selected)
    row["baseline_useful"] = row["kind"] in FAMILIES and any(selected[:6])
    return row


def summarize(rows):
    development = [row for row in rows if row["cohort"] == "selection"]
    held = [row for row in rows if row["cohort"] == "held"]
    old_groups = {row["group"] for row in development}
    held_groups = {row["group"] for row in held}
    ordinary = {row["group"]: row for row in held if row["kind"] == "ordinary"}
    require(len(rows) == len(development) + len(held) and len(old_groups) == len(development) == 15
            and len(held_groups) == len(ordinary) == 14 and not old_groups & held_groups
            and all(row["kind"] == "ordinary" for row in development)
            and len({(row["group"], row["kind"]) for row in rows}) == len(rows), "Held or selection source roles changed")
    require(all({"ordinary", "self"} <= {row["kind"] for row in held if row["group"] == group} for group in held_groups),
            "A held work lacks its matched controls")
    denominators = Counter(row["kind"] for row in held)
    points = {name: sorted(row["metrics"][name] for row in development)[-2] for name in METHODS}
    baseline_ordinary = {row["group"] for row in held if row["kind"] not in FAMILIES and row["baseline_flag"]}

    def compare(method, point):
        generated = [row for row in held if row["kind"] in FAMILIES]
        originals = [row for row in generated if row["baseline_useful"]]
        additional = [row for row in generated if not row["baseline_useful"]
                      and method is not None and row["metrics"][method] > point]
        current_false = set(baseline_ordinary)
        if method is not None:
            current_false.update(group for group, row in ordinary.items() if row["metrics"][method] > point)
        return {"useful": len(originals) + len(additional), "added_useful_files": len(additional),
                "added_useful_groups": len({row["group"] for row in additional}),
                "added_families": sorted({row["kind"] for row in additional}), "ordinary_groups": len(current_false),
                "added_ordinary_groups": len(current_false - baseline_ordinary),
                "paired_wins": {family: sum(method is not None and row["kind"] == family
                                            and row["metrics"][method] > ordinary[row["group"]]["metrics"][method]
                                            for row in generated) for family in sorted(FAMILIES)},
                "useful_by_family": dict(Counter(row["kind"] for row in originals + additional))}

    baseline = compare(None, 0)
    methods = {method: compare(method, point) for method, point in points.items()}
    extent_options = {row["metrics"]["lexical"] for row in held} | {-1.0}
    extent = [compare("lexical", point) for point in extent_options]
    for method in ("conditional", "source_specific"):
        arm = methods[method]
        budget = arm["added_ordinary_groups"]
        match = max(item["added_useful_files"] for item in extent if item["added_ordinary_groups"] <= budget)
        arm["lexical_max_added_at_same_budget"] = match
        arm["passes"] = (arm["added_useful_groups"] >= 3 and set(arm["added_families"]) == FAMILIES
                         and all(denominators[family] >= 6 for family in FAMILIES) and budget <= 1
                         and arm["added_useful_files"] > match)
    return {"points": points, "baseline": baseline, "comparison": methods, "held_denominators": dict(denominators)}


def replay(receipt):
    require(receipt["schema"] == SCHEMA and receipt["confidence_weight"] == 0, "Unknown receipt or confidence scope")
    predictions = receipt["baseline_responses"]
    sources = {row["group"]: row for row in receipt["sources"]}
    require(len(sources) == len(receipt["sources"]) == 29, "Frozen source groups changed")
    rows = []
    for row in receipt["cases"]:
        source, donor = sources[row["group"]], sources[row["donor"]]
        require(source["role"] == donor["role"] == row["cohort"] and row["group"] != row["donor"],
                "Wrong-source condition crossed the frozen split")
        require(all(HEX_DIGEST.fullmatch(value) for value in
                    [row["target_sha256"], row["text_sha256"], *row["prompts"].values()]), "Invalid bound input digest")
        rows.append(score(dict(row), predictions))
    require(set(predictions) == {window["input_sha256"] for row in rows for window in row["windows"]},
            "Incumbent input ledger changed")
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    receipt = json.loads(parser.parse_args().receipt.read_bytes())
    result = replay(receipt)
    require(result == receipt["result"], "Source-conditioned comparison changed")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
