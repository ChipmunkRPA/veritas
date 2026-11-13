# SPDX-License-Identifier: Apache-2.0
"""Recompute paired center and context decisions from text-free local receipts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-paired-center-transfer/1"
ARMS = ("context", "calibrated_context", "center", "joint")
GENERATORS = {"granite", "olmo", "qwen", "smol2"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_span(spans):
    require(len(spans) == 3 and all(len(row) == 2 and all(type(p) is int for p in row) for row in spans),
            "Three exact source positions are required")
    require(0 <= spans[0][0] < spans[0][1] <= spans[1][0] < spans[1][1] <= spans[2][0] < spans[2][1],
            "Source positions must be ordered and disjoint")


def logit(hex_value):
    score = float.fromhex(hex_value)
    require(math.isfinite(score) and 0 <= score <= 1, "Invalid saved response")
    bounded = min(1 - 1e-7, max(1e-7, score))
    return bounded / (1 - bounded)


def read_rows(receipt):
    points = {arm: {} for arm in ("context", "center")}
    pin_sets = {}
    panels = receipt["cases"]
    expected = {
        "selection": Counter(before=9, edited=9),
        "journal": Counter(before=21, edited=21, qwen=16, smol2=16),
        "f1000": Counter(self=10, ordinary=10, granite=8, olmo=6),
    }
    require(panels.keys() == expected.keys(), "Panel names differ")
    for name, rows in panels.items():
        require(Counter(row["role"] for row in rows) == expected[name], "Panel roles differ")
        require(len({(row["group"], row["role"]) for row in rows}) == len(rows), "Repeated work variant")
        pin_sets[name] = set()
        earlier = {}
        for row in rows:
            later, old = row["pair"]
            group = row["group"]
            if group not in earlier:
                earlier[group] = old
            require(old == earlier[group], "The paired source version changed")
            require(later["neighbors"] == old["neighbors"] and len(old["neighbors"]) == 2, "Unchanged flanks differ")
            for side in (later, old):
                checked_span(side["spans"])
                require(all(SHA256.fullmatch(pin) for pin in
                            (side["source_sha256"], side["context"], side["center"], *side["neighbors"])),
                        "Invalid earlier or later source commitment")
                for arm in ("context", "center"):
                    pin_sets[name].add(side[arm])
            unchanged = row["role"] in {"before", "self"}
            require((later == old) == unchanged and (later["source_sha256"] == old["source_sha256"]) == unchanged,
                    "Ordinary unchanged control differs from its earlier version")
            require(all(type(row[key]) is bool for key in ("existing_useful", "existing_flagged", "native_app_eligible"))
                    and (not row["existing_useful"] or row["existing_flagged"])
                    and (not unchanged or not row["native_app_eligible"]), "Invalid incumbent or app-scope status")
    require(not pin_sets["selection"] & (pin_sets["journal"] | pin_sets["f1000"]), "Calibration and held inputs overlap")
    require(len(set().union(*pin_sets.values())) == 252, "Source input count differs")
    checkpoints = receipt["measurements"]
    require(set(checkpoints) == {"first", "second"}, "Both pinned response sets are required")
    for model, readings in checkpoints.items():
        require(set(readings) == set().union(*pin_sets.values()), "Response roster differs from exact native inputs")
        limit = {"first": 768, "second": 512}[model]
        for response in readings.values():
            require(type(response["tokens"]) is int and 0 < response["tokens"] <= limit, "Incomplete native model input")
            logit(response["score_hex"])
    for name, rows in panels.items():
        for index, row in enumerate(rows):
            later, old = row["pair"]
            for arm in points:
                points[arm][name, index] = min(math.log(logit(readings[later[arm]]["score_hex"])
                                                        / logit(readings[old[arm]]["score_hex"]))
                                                for readings in checkpoints.values())
    return panels, points


def replay(receipt):
    require(receipt["schema"] == SCHEMA, "Unknown paired writing receipt")
    require(set(receipt["bindings"]) == {"freeze", "first", "second", "panels"}
            and all(SHA256.fullmatch(value) for value in receipt["bindings"].values()), "Incomplete study commitments")
    require(receipt["forwards"] == {"first": 63, "second": 63}, "Incomplete model runs")
    panels, scores = read_rows(receipt)
    scores["joint"] = {k: min(scores["center"][k], scores["context"][k]) for k in scores["center"]}
    scores["calibrated_context"] = scores["context"]
    cutoffs = {"context": math.log(2)}
    for arm in ("calibrated_context", "center", "joint"):
        maxima = {}
        for i, row in enumerate(panels["selection"]):
            group = row["group"]
            maxima[group] = max(maxima.get(group, -math.inf), scores[arm]["selection", i])
        cutoffs[arm] = sorted(maxima.values(), reverse=True)[1]
    summary = {}
    for panel in ("journal", "f1000"):
        summary[panel] = {}
        for arm in ARMS:
            counts = Counter()
            gains, families, false_groups, prior_false = set(), set(), set(), set()
            for i, row in enumerate(panels[panel]):
                later, old = row["pair"]
                edited = later["source_sha256"] != old["source_sha256"]
                prior = edited and row["native_app_eligible"] and scores["context"][panel, i] >= cutoffs["context"]
                score = scores[arm][panel, i]
                flagged = edited and (score >= cutoffs[arm] if arm == "context" else score > cutoffs[arm])
                group, role = row["group"], row["role"]
                if role in GENERATORS:
                    incumbent = row["existing_useful"] or prior
                    added = flagged and not incumbent
                    counts.update(known=1, existing=incumbent, candidate=flagged, added=added,
                                  app_eligible_added=added and row["native_app_eligible"])
                    if added:
                        gains.add(group)
                        families.add(role)
                else:
                    counts.update(genuine=1, flagged=flagged)
                    if flagged:
                        false_groups.add(group)
                    if row["existing_flagged"] or prior:
                        prior_false.add(group)
            summary[panel][arm] = dict(counts) | {
                "added_works": len(gains), "families": sorted(families), "false": len(false_groups),
                "prior_false": len(prior_false), "added_false": len(false_groups - prior_false),
            }
    return {"points": cutoffs, "summary": summary}


def replay_boundary(receipt, result):
    geometry = receipt["boundary_batches"]
    require(len(geometry) == 22, "Native paragraph-break sample count differs")
    require(Counter(row["panel"] for row in geometry) == {"selection": 2, "journal": 15, "f1000": 5},
            "Native paragraph-break panel count differs")
    counts = Counter()
    checked = set()
    for row in geometry:
        panel, index = row["panel"], row["index"]
        require((panel, index) not in checked, "Repeated native paragraph-break case")
        checked.add((panel, index))
        case = receipt["cases"][panel][index]
        later, earlier = case["pair"]
        left, middle, right = row["paragraphs"]
        require(all(type(value) is int for value in (left, middle, right)) and 0 <= left <= middle <= right
                and right == left + 1 and row["change_ids"],
                "The replay requires exactly one preserved adjacent paragraph boundary")
        require(row["pins"] == [earlier["context"], later["context"], earlier["center"], later["center"]],
                "Batch inputs differ from the frozen exact ranges")
        values = []
        require(set(row["readings"]) == {"first", "second"}, "Both native batch checkpoints are required")
        for checkpoint, model in row["readings"].items():
            require(len(model["responses_hex"]) == 4 and model["tokens"] == [receipt["measurements"][checkpoint][key]["tokens"]
                    for key in row["pins"]], "The native batch did not measure each complete input")
            values.append([logit(raw) for raw in model["responses_hex"]])
        require(len(values) == 2 and not case["native_app_eligible"], "Wrong native boundary input cohort")
        context = min(math.log(data[1] / data[0]) for data in values)
        center = min(math.log(data[3] / data[2]) for data in values)
        joint_mark = min(context, center) > result["points"]["joint"]
        context_mark = context > result["points"]["calibrated_context"]
        require((joint_mark, context_mark) == (row["joint"], row["calibrated_context"]), "Native batch decision differs")
        if panel != "selection":
            role = "generated" if case["role"] in GENERATORS else "ordinary"
            counts.update({role: 1, f"{role}_joint": joint_mark, f"{role}_context": context_mark})
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    result = replay(receipt)
    require(result == receipt["result"], "Published paired writing outcomes differ from replay")
    boundary = replay_boundary(receipt, result)
    require(boundary == receipt["boundary_result"], "Published boundary outcomes differ from replay")
    result["boundary_result"] = boundary
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
