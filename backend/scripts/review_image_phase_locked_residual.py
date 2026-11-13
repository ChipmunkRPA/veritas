# SPDX-License-Identifier: Apache-2.0
"""Audit a sealed image phase-localization ledger without model or image dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("period_four", "period_eight")
COUNTS = {"cal": 32, "held_lama": 72, "held_generated": 30, "screen": 36}
THRESHOLD = {"primary_feathered_gains": 2, "alternate_feathered_gains": 1,
             "extra_photo_groups": 2, "extra_screenshot_groups": 1, "mislocalized_edits": 2}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def evaluate(rows):
    counts = defaultdict(lambda: defaultdict(Counter))
    gains = {arm: defaultdict(set) for arm in ARMS}
    wrong = Counter()
    for row in rows:
        panel, group, role = row["key"]
        height, width = row["shape"]
        require(type(row["target"]) is int and 0 <= row["target"] <= height * width, "Invalid changed-pixel count")
        positive = row["target"] > 0
        require(positive == (role in {"edited", "feathered_edit", "edit"}), "The case role disagrees with its truth")
        require(set(row["responses"]) == {*ARMS, "sharp_boundary"}, "A candidate or incumbent response is missing")
        base = row["responses"]["sharp_boundary"]["flag"]
        for arm, response in row["responses"].items():
            box, value, accepted = response["box"], response["score"], response["flag"]
            assessed = value is not None
            require(type(accepted) is bool and (box is not None) == assessed, "Invalid scope or rectangle")
            require(not assessed or (type(value) in (int, float) and math.isfinite(value) and value >= 0), "Invalid response")
            reported = 0
            if box is not None:
                x0, y0, x1, y1 = box
                require(all(type(n) is int for n in box) and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
                        "A selected rectangle leaves its native image")
                if arm in ARMS:
                    require(all(n % 16 == 0 for n in box), "A native phase proposal is off its fixed grid")
                reported = (x1 - x0) * (y1 - y0)
            pixels = response["pixels"]
            intersection = pixels["overlap"]
            require(pixels["reported"] == reported and pixels["target"] == row["target"]
                    and type(intersection) is int and 0 <= intersection <= min(reported, row["target"]),
                    "Recorded overlap is outside the native geometry")
            useful = positive and intersection >= .5 * row["target"] and intersection >= .3 * (reported + row["target"] - intersection)
            scope = ("native" if role == "edited" else "feathered") if positive else ("interface" if panel == "screen" else "ordinary")
            counts[arm][scope].update(views=1, assessed=assessed, flagged=accepted, useful=accepted and useful, useful_before_point=useful)
            if arm not in ARMS or not accepted or base:
                continue
            if not positive:
                gains[arm]["extra_screenshot_groups" if panel == "screen" else "extra_photo_groups"].add((panel, group))
            elif not useful:
                wrong[arm] += 1
            elif role != "edited":
                gains[arm]["primary_feathered_gains" if panel == "held_lama" else "alternate_feathered_gains"].add(group)
    actual = {arm: {key: wrong[arm] if key == "mislocalized_edits" else len(gains[arm][key])
                    for key in THRESHOLD} for arm in ARMS}
    passes = {arm: all(actual[arm][key] >= bound if key.endswith("_gains") else actual[arm][key] <= bound
                       for key, bound in THRESHOLD.items()) for arm in ARMS}
    return {arm: dict(items) for arm, items in counts.items()}, actual, passes


def review(plan, calibration, held):
    manifest = json.loads(plan)
    calibrated = json.loads(calibration)
    measured = json.loads(held)
    require(calibrated["freeze_sha256"] == measured["freeze_sha256"] == sha(plan)
            and measured["calibration_sha256"] == sha(calibration), "A freeze or calibration commitment differs")
    inventory = manifest["cases"]
    require({name: len(rows) for name, rows in inventory.items()} == COUNTS, "A cohort is incomplete")
    maxima = {arm: defaultdict(float) for arm in ARMS}
    for observed, pin in zip(calibrated["rows"], inventory["cal"], strict=True):
        require(all(observed[key] == value for key, value in pin.items()) and not observed["target"], "Calibration view changed")
        for arm in ARMS:
            score = observed["responses"][arm]["score"]
            require(score is None or math.isfinite(score) and score >= 0, "Invalid calibration score")
            maxima[arm][observed["key"][1]] = max(maxima[arm][observed["key"][1]], score or 0)
    require(all(len(groups) == 8 for groups in maxima.values()), "Expected eight calibration sources")
    points = {arm: sorted(groups.values())[-2] for arm, groups in maxima.items()}
    require(points == calibrated["points"] == measured["points"], "A cutoff differs from its frozen ordinary controls")
    case_roster = [row for name in ("held_lama", "held_generated", "screen") for row in inventory[name]]
    require(len({tuple(row["key"]) for row in case_roster}) == len(case_roster), "A held case was duplicated")
    for observed, pin in zip(measured["rows"], case_roster, strict=True):
        require(all(observed[key] == value for key, value in pin.items()), "A scored case differs from the frozen view")
        for arm in ARMS:
            row = observed["responses"][arm]
            score = row["score"]
            require(row["flag"] is (score is not None and score > points[arm]), "A flag disagrees with its frozen cutoff")
    counts, gains, gates = evaluate(measured["rows"])
    require(gains == measured["summary"]["gains"] and gates == measured["summary"]["passes"], "Advancement decision differs")
    return {"freeze_sha256": sha(plan), "calibration_sha256": sha(calibration), "held_sha256": sha(held),
            "views": len(case_roster), "calibration_views": len(inventory["cal"]), "cutoffs": points,
            "counts": counts, "held_gate_counts": gains, "passed": gates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    files = ("freeze.json", "calibration.json", "held.json")
    result = review(*((args.run / file).read_bytes() for file in files))
    print(json.dumps(result, indent=2, sort_keys=True))
