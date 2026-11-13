# SPDX-License-Identifier: Apache-2.0
"""Recompute earlier-copy mouth localization counts from retained scalar records."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-paired-native-mouth-review/1"
ARMS = ("luminance", "luminance_and_texture")
NEURAL_ROLES = {"source_time_neural", "same_time_neural"}
ORDINARY_ROLES = {"original", "compression", "tone"}
LOCAL_ROLES = {"resample_ordinary", "delayed_pixels_ordinary", "same_time_resample", "delayed_pixels"}
HASH = re.compile(r"[0-9a-f]{64}")


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval_iou(a, b):
    a_start, a_end = map(Fraction, a)
    b_start, b_end = map(Fraction, b)
    union = max(a_end, b_end) - min(a_start, b_start)
    return max(0, min(a_end, b_end) - max(a_start, b_start)) / union


def measure(row, arm, points, held):
    proposal = row["measurements"].get(arm)
    evidence = row["evidence"].get(arm)
    require(bool(evidence) == (held and proposal is not None and proposal["score"] > points[arm]),
            "Evidence does not match the frozen decision")
    if proposal is None:
        return False, False
    start, end = proposal["frames"]
    x0, y0, x1, y1 = proposal["tile"]
    width, height = (720, 480) if held else (360, 288)
    require(math.isfinite(proposal["score"]) and 5 <= start < end <= row["pictures"] - 5
            and end - start in {8, 12, 16, 20} and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
            "Invalid proposed time range or tile")
    if evidence is None:
        return False, False
    before, middle, after = evidence["anchors"]
    first, last = map(Fraction, evidence["interval"])
    tb = Fraction(row["time_base"])
    require([item["index"] for item in (before, middle, after)] == [start - 1, (start + end) // 2, end]
            and before["pts"] * tb < first <= middle["pts"] * tb < last == after["pts"] * tb
            and all(len(item["rgb"]) == 2 and all(HASH.fullmatch(value) for value in item["rgb"])
                    for item in (before, middle, after)), "Invalid source-clock anchors")
    if row["truth"] is None:
        require("pixels" not in evidence, "Unexpected edit-mask evidence")
        return False, False
    pixels = evidence["pixels"]
    changed, covered, tile = pixels["changed"], pixels["covered"], pixels["tile_pixels"]
    require(pixels["index"] == middle["index"] and tile == (x1 - x0) * (y1 - y0)
            and 0 <= covered <= min(changed, tile) and changed <= width * height, "Invalid native pixel overlap")
    overlaps_time = interval_iou(evidence["interval"], row["truth"]) >= Fraction(1, 2)
    return overlaps_time, overlaps_time and 10 * covered >= changed + tile - covered


def summarize(rows, points, held):
    seen = set()
    counts = Counter()
    results = {arm: Counter() for arm in ARMS}
    qualified_groups = {arm: set() for arm in ARMS}
    for row in rows:
        role, family = row["role"], row["family"]
        key = (row["case"], role)
        require(key not in seen and 40 <= row["pictures"] <= 100 and Fraction(row["time_base"]) > 0
                and all(HASH.fullmatch(row[name]) for name in ("pts_sha256", "earlier_sha256", "submitted_sha256")),
                "Invalid or duplicate picture comparison")
        seen.add(key)
        require(family in ({"lombard", "musetalk"} if held else {"grid"})
                and role in ORDINARY_ROLES | LOCAL_ROLES | NEURAL_ROLES | {"wrong_source"}
                and bool(row["truth"]) == (role in NEURAL_ROLES)
                and (not row["aligned"] or role in NEURAL_ROLES), "Unexpected source role")
        status = row["status"]
        require(status in {"measured", "no_shared_content", "insufficient_luma"}
                and row["measurements"].keys() == (set(ARMS) if status == "measured" else set())
                and row["evidence"].keys() <= set(ARMS), "Invalid measurement coverage")
        counts[role] += 1
        if role == "wrong_source" and status == "measured":
            counts["wrong_source_admitted"] += 1
        if row["aligned"]:
            counts[f"{family}_aligned"] += 1
        for arm in ARMS:
            overlaps_time, localized = measure(row, arm, points, held)
            result = results[arm]
            if row["measurements"].get(arm) is not None:
                result[f"{role}_assessed"] += 1
            if arm in row["evidence"]:
                result[f"{role}_flagged"] += 1
            if overlaps_time:
                result[f"{family}_time_overlap"] += 1
            if localized:
                result[f"{family}_localized"] += 1
                if row["aligned"]:
                    result[f"{family}_aligned_localized"] += 1
                    if family == "lombard":
                        qualified_groups[arm].add(row["group"])
    expected = {role: 17 if held else 8 for role in ORDINARY_ROLES}
    if held:
        expected |= {"source_time_neural": 11, "same_time_neural": 6, "resample_ordinary": 11,
                     "delayed_pixels_ordinary": 11, "same_time_resample": 6, "delayed_pixels": 6,
                     "musetalk_aligned": 1}
    require(all(counts[key] == value for key, value in expected.items())
            and 0 < counts["wrong_source"] <= expected["original"], "Missing comparison cases")
    if held:
        for arm, result in results.items():
            result["lombard_aligned_localized_groups"] = len(qualified_groups[arm])
            result["passes_frozen_rule"] = (
                result["lombard_localized"] + result["musetalk_localized"] >= 6
                and result["lombard_aligned_localized"] >= 4
                and len(qualified_groups[arm]) >= 3
                and result["musetalk_aligned_localized"] == 1
                and sum(result[f"{role}_flagged"] for role in ORDINARY_ROLES) <= 2
                and not counts["wrong_source_admitted"]
            )
    return {"counts": dict(counts), "arms": {arm: dict(result) for arm, result in results.items()}}


def review(receipt):
    require(receipt["schema"] == SCHEMA and all(HASH.fullmatch(value) for value in receipt["inputs"].values()),
            "Unknown ledger schema or input commitment")
    development = receipt["development"]
    points = {arm: max(.8, max([0., *[row["measurements"][arm]["score"] for row in development
                                      if row["measurements"].get(arm) is not None]]) + .15) for arm in ARMS}
    require(receipt["points"] == points, "Operating points do not match ordinary development controls")
    return {"points": points, "development": summarize(development, points, False),
            "held": summarize(receipt["held"], points, True)}


def public_receipt(run):
    inputs = {}
    for name in ("freeze", "development", "held", "points"):
        inputs[name] = hashlib.sha256((run / f"{name}.json").read_bytes()).hexdigest()
    development, held, points = [read(run / f"{name}.json") for name in ("development", "held", "points")]
    require(development["freeze_sha256"] == held["freeze_sha256"] == inputs["freeze"]
            and points["development_sha256"] == inputs["development"], "Source artifacts disagree")
    groups = {}
    for row in (*development["rows"], *held["rows"]):
        group = (row["family"], row["group"])
        row["group"] = groups.setdefault(group, len(groups))
    return {"schema": SCHEMA, "inputs": inputs, "points": points["points"],
            "development": development["rows"], "held": held["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble-from", type=Path)
    args = parser.parse_args()
    receipt = read(args.receipt) if args.assemble_from is None else public_receipt(args.assemble_from)
    result = review(receipt)
    if args.assemble_from is not None:
        with args.receipt.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
