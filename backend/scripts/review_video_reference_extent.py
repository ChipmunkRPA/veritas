# SPDX-License-Identifier: Apache-2.0
"""Audit time and pixel overlap from the published paired-video region measurements."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

ARMS = ("tile", "extent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spatial(row, arm):
    if "overlap" not in row:
        return 0.
    changed = row["overlap"]["changed"]
    shared = row["overlap"][arm]
    box = row[arm]
    area = box["w"] * box["h"]
    require(0 <= shared <= min(area, changed), "Invalid shared pixel count")
    return shared / (area + changed - shared)


def useful(row, arm):
    if row["truth"] is None or row["status"] != "local_difference":
        return False
    a, b = row["frames"]
    left, right = row["truth"]
    return 2 * (min(b, right) - max(a, left)) >= max(b, right) - min(a, left) and spatial(row, arm) >= .10


def counts(rows):
    return {arm: sum(useful(row, arm) for row in rows) for arm in ARMS}


def validate(rows):
    for row in rows:
        if row["status"] != "local_difference":
            require("anchors" not in row and "overlap" not in row, "Unmeasured result has picture evidence")
            continue
        a, b = row["frames"]
        require(b - a in (8, 12, 16, 20) and a >= 5 and b <= row["pictures"] - 5,
                "Invalid frozen picture range")
        require([row["anchors"][role]["index"] for role in ("before", "during", "after")] == [a - 1, (a+b)//2, b],
                "Picture anchors differ")
        for arm in ARMS:
            x, y, w, h = (row[arm][key] for key in ("x", "y", "w", "h"))
            require(min(x, y) >= 0 and min(w, h) > 0 and x + w <= 720 and y + h <= 480,
                    "Picture box is outside the source")
            spatial(row, arm)


def review(study, controls):
    require(study["schema"] == "chipmunk.video-reference-extent/1"
            and controls["schema"] == "chipmunk.video-reference-extent-controls/1", "Unexpected study schema")
    diagnostic, transfer, rows = study["diagnostic"], study["transfer"], controls["rows"]
    require(len(diagnostic) == 17 and Counter(row["role"] for row in transfer) ==
            {"ai_aligned": 10, "ai_earlier_320ms": 10, "original": 10, "resample": 10,
             "reencode": 10, "global_tone": 10, "wrong_source": 9}, "Transfer roster differs")
    require(len(rows) == 36 and {(r["source"], r["role"]) for r in rows} == {
        (source, role) for source in range(6) for role in ("original", "large", "small", "compression", "tone", "wrong")
    }, "Control roster differs")
    validate(diagnostic)
    validate(transfer)
    validate(rows)
    neural = [r for r in transfer if r["role"].startswith("ai_")]
    neutral = [r for r in transfer if r["role"] in {"original", "reencode", "global_tone"}]
    wrong_statuses = {"unverified_photometric_support", "insufficient_shared_picture_content"}
    result = {"diagnostic": counts(diagnostic),
              "qualified_second_family": counts([r for r in diagnostic if r["bank"] == "musetalk" and r["aligned"]]),
              "transfer": counts(neural),
              "extent_additions": sum(useful(r, "extent") and not useful(r, "tile") for r in neural),
              "extent_losses": sum(useful(r, "tile") and not useful(r, "extent") for r in neural),
              "extent_groups": len({r["source"] for r in neural if useful(r, "extent")}),
              "neutral_flags": sum(r["status"] == "local_difference" for r in neutral),
              "wrong_source_admissions": sum(r["status"] not in wrong_statuses for r in transfer if r["role"] == "wrong_source")}
    require(result == study["results"], "Transfer counts differ from published decisions")
    result["neural_gate_passed"] = (result["qualified_second_family"]["extent"] == 1
                                    and result["transfer"]["extent"] >= 15 and result["extent_groups"] >= 7
                                    and result["extent_additions"] >= 3 and result["extent_losses"] <= 1
                                    and result["neutral_flags"] <= 1 and result["wrong_source_admissions"] == 0)
    for role in ("large", "small"):
        edited = [r for r in rows if r["role"] == role]
        result[role] = counts(edited) | {"median_iou": {arm: median(spatial(r, arm) for r in edited) for arm in ARMS},
                                       "lost": sum(useful(r, "tile") and not useful(r, "extent") for r in edited)}
    result["ordinary_neutral_flags"] = sum(r["status"] == "local_difference" for r in rows
                                           if r["role"] in {"original", "compression", "tone"})
    result["ordinary_wrong_admissions"] = sum(r["status"] not in wrong_statuses for r in rows if r["role"] == "wrong")
    iou = result["large"]["median_iou"]
    result["ordinary_gate_passed"] = (result["large"]["extent"] >= 5 and iou["extent"] - iou["tile"] >= .15
                                     and result["small"]["extent"] >= 5 and result["small"]["lost"] <= 1
                                     and result["ordinary_neutral_flags"] <= 1 and result["ordinary_wrong_admissions"] == 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("controls", type=Path)
    args = parser.parse_args()
    values = review(json.loads(args.study.read_bytes()), json.loads(args.controls.read_bytes()))
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
