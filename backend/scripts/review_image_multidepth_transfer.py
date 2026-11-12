# SPDX-License-Identifier: Apache-2.0
"""Replay regional selections from exact source deliveries and saved responses."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_edge_spread_transfer as panel
from scripts.review_image_residual_sign_transfer import hits, selected_mask

FORMATS = ("png", "jpeg75")
ARMS = ("last_layer", "three_depths")
BASELINE = "format_reference"
GRID = 24
SIDES = {4, 6, 8}


def box_for_region(region, shape):
    y, x, side = region
    panel.require(type(y) is int and type(x) is int and type(side) is int
                  and side in SIDES and 0 <= x <= GRID - side and 0 <= y <= GRID - side, "Invalid source grid region")
    height, width = shape

    def start(index, size):
        return (index * size + GRID - 1) // GRID

    return [start(x, width), start(y, height), start(x + side, width), start(y + side, height)]


def readback(row, shape, truth, points):
    observed = {}
    ranked = {}
    for arm in (*ARMS, BASELINE):
        proposal = row["proposals"][arm]
        if arm == BASELINE:
            picked = bool(proposal and proposal["selected"])
            mask = selected_mask(shape, proposal, picked)
        else:
            panel.require(math.isfinite(proposal["score"]), "Non-finite recorded response")
            box = box_for_region(proposal["region"], shape)
            ranked_mask = selected_mask(shape, {"box": box}, True)
            count, target, overlap = int(ranked_mask.sum()), int(truth.sum()), int(np.count_nonzero(ranked_mask & truth))
            ranked[arm] = hits({"reported": count, "target": target, "overlap": overlap})
            panel.require(ranked[arm] == proposal["ungated_useful"], "Incorrect ungated region")
            mask = ranked_mask & (proposal["score"] > points[arm]["point"])
        target, area, overlap = int(truth.sum()), int(mask.sum()), int(np.count_nonzero(mask & truth))
        observed[arm] = {"target": target, "reported": area, "overlap": overlap,
                         "iou": overlap / (target + area - overlap) if target else None,
                         "recall": overlap / target if target else None}
    panel.require(observed == row["measurements"], "Native pixel overlaps differ")
    return observed, ranked


def review(folder, rows, points, expected):
    lookup = {panel._key(row): row for row in rows}
    panel.require(len(rows) == len(lookup) == 192, "Missing or duplicate delivery rows")
    keys = ("sharp", "feathered", "ordinary", "original", "extra_ordinary", "extra_feathered", "ungated_useful")
    counts = {fmt: {arm: Counter(dict.fromkeys(keys, 0)) for arm in (*ARMS, BASELINE)} for fmt in FORMATS}
    source_flags = {fmt: {arm: set() for arm in (*ARMS, BASELINE)} for fmt in FORMATS}
    total = 0
    for case, source, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(source, case["box"]).items():
            truth = np.any(pixels != source["original"], axis=-1) if role in panel.POSITIVE else np.zeros_like(known)
            for fmt in FORMATS:
                row = lookup.pop((case["slot"], case["generator"], role, fmt))
                body, _ = panel.paired.encoded(pixels, fmt)
                panel.require(hashlib.sha256(body).hexdigest() == row["input_sha256"]
                              and hashlib.sha256(truth.tobytes()).hexdigest() == row["truth_sha256"],
                              "Delivery or native truth does not match its commitment")
                measures, ranked = readback(row, truth.shape, truth, points[fmt])
                for arm, reading in measures.items():
                    record = counts[fmt][arm]
                    if role in panel.POSITIVE:
                        hit = hits(reading)
                        record["sharp" if role == "edited" else "feathered"] += hit
                        record["extra_feathered"] += role == "feathered_edit" and hit and not hits(measures[BASELINE])
                        record["ungated_useful"] += ranked.get(arm, False)
                    elif reading["reported"]:
                        record["ordinary"] += 1
                        record["original"] += role == "original"
                        if not measures[BASELINE]["reported"]:
                            record["extra_ordinary"] += 1
                            source_flags[fmt][arm].add(case["slot"])
                total += 1
    panel.require(not lookup and total == 192, "Rebuilt roster is incomplete")
    for fmt in FORMATS:
        for arm in (*ARMS, BASELINE):
            counts[fmt][arm]["extra_ordinary_sources"] = len(source_flags[fmt][arm])
            if arm == BASELINE:
                counts[fmt][arm]["ungated_useful"] = None
            panel.require(dict(counts[fmt][arm]) == expected[fmt][arm], "Independent outcome arithmetic differs")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "earlier", "later"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen, fit, results = (panel.read(args.run / file) for file in ("freeze.json", "points.json", "result.json"))
    freeze_sha = panel.sha(args.run / "freeze.json")
    panel.require(freeze_sha == fit["freeze_sha256"] == results["freeze_sha256"]
                  and panel.sha(args.run / "points.json") == results["points_sha256"], "Method or calibration commitment differs")
    for arms in fit["points"].values():
        for reading in arms.values():
            values = reading["source_maxima"]
            panel.require(len(values) == 12 and all(math.isfinite(value) for value in values)
                          and reading["point"] == sorted(values)[-2], "Saved cutoff is not the second-largest control maximum")
    views = 0
    for label, folder in (("earlier", args.earlier), ("later", args.later)):
        for name, value in frozen["intakes"][label].items():
            panel.require(panel.sha(folder / name) == value, "Source intake differs")
        journal = args.run / f"{label}-rows.json"
        panel.require(panel.sha(journal) == results["journals"][label], "Held responses differ")
        views += review(folder, panel.read(journal), fit["points"], results["results"][label])
    decisions = {}
    for fmt in FORMATS:
        arms = [values[fmt]["three_depths"] for values in results["results"].values()]
        decisions[fmt] = (min(row["extra_feathered"] for row in arms) >= 2
                          and max(row["extra_ordinary"] for row in arms) <= 4
                          and max(row["extra_ordinary_sources"] for row in arms) <= 3)
    panel.require(decisions == results["three_depths_advances"], "Recorded promotion decision differs")
    panel.write(args.run / "audit.json", {"status": "verified", "exact_deliveries": views,
                "native_overlap_records": views * 3, "result_sha256": panel.sha(args.run / "result.json"),
                "decisions": decisions})
    print(views, "deliveries and", views * 3, "native overlap records verified", flush=True)


if __name__ == "__main__":
    main()
