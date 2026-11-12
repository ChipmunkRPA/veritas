# SPDX-License-Identifier: Apache-2.0
"""Test whether deterministic JPEG recompression corroborates a native PNG crop."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from scripts import image_crop_overlap_transfer as original
from scripts import image_remote_crop_editor_transfer as remote

panel = remote.panel
source = remote.source
cohorts = ("earlier", "later", "fresh")
arms = ("native_remote", "paired_minimum", "native_with_fallback")


def binding(args):
    previous = panel.read(args.remote / "result.json")
    point_lock = panel.read(args.remote / "freeze.json")
    prior = panel.read(args.prior / "result.json")
    ledgers = {label: panel.sha(args.remote / f"{label}-rows.json") for label in cohorts}
    panel.require(ledgers == previous["journals"] and point_lock["points_sha256"] == remote.POINTS_SHA
                  and panel.sha(args.remote / "freeze.json") == previous["freeze_sha256"],
                  "Unexpected native crop transfer commitments")
    dev = args.prior / "development.json"
    panel.require(panel.sha(args.prior / "held.jsonl") == point_lock["prior_rows"] == prior["held_sha256"]
                  and panel.sha(args.prior / "freeze.json") == prior["freeze_sha256"],
                  "Earlier crop inference commitments differ")
    fit_lock = panel.read(args.prior / "freeze.json")
    panel.require(fit_lock["weight_sha256"] == point_lock["weight_sha256"] == source.WEIGHT_SHA,
                  "Crop checkpoints differ")
    fit = {name: panel.sha(args.development / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
    panel.require(fit == fit_lock["inputs"]["development"], "Fitting roster differs")
    for label in cohorts:
        intake = {name: panel.sha(getattr(args, label) / name) for name in fit}
        panel.require(intake == point_lock["intakes"][label], "Held roster differs")
    return {"schema": "chipmunk.image-crop-delivery-consistency/1", "held_ledgers": ledgers,
            "native_points": point_lock["points"], "native_point_receipt": previous["freeze_sha256"],
            "fitting_ledger": panel.sha(dev), "fitting_intake": fit, "held_intakes": point_lock["intakes"],
            "checkpoint": source.WEIGHT_SHA, "code_sha256": panel.sha(Path(__file__)),
            "rules": {"paired": "Select the largest per-crop minimum of within-view median-centered logits.",
                      "point": "Strictly above the second-highest of eight fitting-source control maxima.",
                      "fallback": "Use the native remote result when it has a flag, otherwise use the paired result.",
                      "advance": "Each twelve-source panel needs at least one added useful fallback location and at most two added ordinary flags. The seven-source transfer must add a useful fallback location and at most one ordinary flag."}}


def observe(label, slot, role, pixels, truth, rows):
    logits = []
    truth_hash = source.digest(truth.tobytes())
    for fmt in panel.FORMATS:
        row = rows.pop((label, slot, role, fmt))
        encoded, image = panel.paired.encoded(pixels, fmt)
        boxes, patches = zip(*remote.crops.crops(image), strict=True)
        panel.require((source.digest(encoded), truth_hash, [original.digest_grid(patch) for patch in patches])
                      == (row["input_sha256"], row["truth_sha256"], row["crop_sha256"]), "Prepared crop or truth bytes differ")
        values = row["logits"]
        panel.require(len(values) == len(boxes) == 25 and all(math.isfinite(value) for value in values),
                      "Expected 25 finite crop responses")
        logits.append(values)
    native, recompressed = logits
    native_center, recompressed_center = median(native), median(recompressed)
    responses = [min(a - native_center, b - recompressed_center) for a, b in zip(native, recompressed, strict=True)]
    index = max(range(len(responses)), key=responses.__getitem__)
    return boxes, logits, index, responses[index]


def fit(args):
    rows = {(row["cohort"], row["slot"], row["role"], row["format"]): row
            for row in panel.read(args.prior / "development.json")}
    plan, checks = original.input_plan(args.development, args.earlier, args.later)
    maxima = []
    for slot, _, pixels, _ in original.insertion.panels(args.development, plan, checks):
        scores = []
        for role in ("original", "processed"):
            image = pixels[role]
            truth = np.zeros(image.shape[:2], dtype=bool)
            scores.append(observe("development", slot, role, image, truth, rows)[3])
        maxima.append(max(scores))
    panel.require(not rows and len(maxima) == 8, "Incomplete fitting comparison")
    return {"control_group_maxima": maxima, "point": sorted(maxima)[-2], "input_bindings": 32 * 25}


def run_panel(label, args, native_points, point):
    rows = {tuple(row["key"]): row for row in panel.read(args.remote / f"{label}-rows.json")}
    for case, images, mask in panel.paired.inputs(getattr(args, label)):
        for role, pixels in panel.native_variants(images, case["box"]).items():
            truth = np.any(pixels != images["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(mask)
            boxes, (native, _), index, response = observe(label, case["slot"], role, pixels, truth, rows)
            native_result = remote.inspect(native, boxes, native_points)
            native_flag = native_result["remote_control"]
            paired_flag = response > point
            native_region = remote.measurement(truth.shape, truth, native_result["box"], native_flag)
            paired_region = remote.measurement(truth.shape, truth, boxes[index], paired_flag)
            yield {"slot": case["slot"], "editor": case["generator"], "role": role, "score": response,
                   "paired_crop": index, "native_remote": native_region, "paired_minimum": paired_region,
                   "native_with_fallback": native_region if native_flag else paired_region}
    panel.require(not rows, "Held case ledger contains unmatched rows")


def summarize(rows):
    metrics = defaultdict(Counter)
    for row in rows:
        original_hit = remote.hits(row["native_remote"])
        original_flag = row["native_remote"]["reported"] > 0
        for arm in arms:
            result = row[arm]
            hit, flag = remote.hits(result), result["reported"] > 0
            positives = row["role"] in panel.POSITIVE
            metrics[arm].update({"sharp": hit and row["role"] == "edited",
                                 "feathered": hit and row["role"] == "feathered_edit",
                                 "ordinary": flag and not positives,
                                 "untouched": flag and row["role"] == "original",
                                 "added_locations": hit and not original_hit,
                                 "lost_locations": original_hit and not hit,
                                 "added_ordinary": flag and not positives and not original_flag,
                                 "cleared_ordinary": original_flag and not positives and not flag})
    return {arm: dict(metrics[arm]) for arm in arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm", "verify"))
    for field in ("prior", "remote", "development", *cohorts, "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    lock = binding(args)
    frozen, fitted = (args.output / name for name in ("freeze.json", "points.json"))
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(frozen, lock)
        return
    panel.require(panel.read(frozen) == lock, "Crop consistency protocol differs")
    if args.stage == "fit":
        panel.write(fitted, fit(args))
        return
    points = panel.read(fitted)
    if args.stage == "verify":
        panel.require(points == fit(args), "Frozen fitting calculation differs")
    summaries, journals = {}, {}
    for label, expected in zip(cohorts, (96, 96, 56), strict=True):
        records = list(run_panel(label, args, lock["native_points"], points["point"]))
        panel.require(len(records) == expected, "Held case count differs")
        journal = args.output / f"{label}-rows.json"
        if args.stage == "confirm":
            panel.write(journal, records)
        else:
            panel.require(panel.read(journal) == records, "Rebuilt case outcomes differ")
        summaries[label], journals[label] = summarize(records), panel.sha(journal)
        print(label, expected, "source-paired outcomes", flush=True)
    advancement = {label: (counts[arms[2]]["added_locations"] >= 1 and
                          counts[arms[2]]["added_ordinary"] <= (1 if label == "fresh" else 2))
                   for label, counts in summaries.items()}
    result = {"freeze_sha256": panel.sha(frozen), "points_sha256": panel.sha(fitted), "journals": journals,
              "results": summaries, "advancement_by_panel": advancement, "advances": all(advancement.values()),
              "held_views": 496, "held_crop_bindings": 12400, "neural_forwards": 0}
    path = args.output / "result.json"
    if args.stage == "confirm":
        panel.write(path, result)
    else:
        panel.require(panel.read(path) == result, "Rebuilt aggregate differs")
        panel.write(args.output / "audit.json", {"result_sha256": panel.sha(path), "verified": True,
                    "fitting_crop_bindings": points["input_bindings"], "held_crop_bindings": 12400})


if __name__ == "__main__":
    main()
