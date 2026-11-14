"""Locate small image edits via blind pixel prediction and matched controls."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts import image_jpeg_signed_transitions as archive
from scripts import image_residual_third_order_transfer as earlier
from scripts.image_jpeg_small_bg_real_holdout import pixel_pairs
from scripts.image_matched_detail_transfer import pixels

SCHEMA = "chipmunk.image-crossfit-prediction/1"
ARMS = ("fixed", "crossfit", "joint")
CONTROL_ROLES = ("ordinary", "local_blur", "local_tone")
GATE = {"main_new_feathered_sources": 4, "cross_dataset_new_feathered_sources": 2,
        "max_new_control_groups_each": 3, "max_misplaced_each": 4}
DESIGN = {"cell": 8, "context_cells": 3, "ridge_variance_floor": 4.,
          "predictors": "horizontal and vertical adjacent-pixel averages; isolated checkerboard fitting and scoring",
          "comparison": "fixed predictor, cross-fitted residual change and orientation agreement, and both jointly",
          "fit": "source-balanced ridge on exact positive and matched genuine/blur/tone cells; no baseline predictions in fit",
          "point": "strictly exceed the second highest ordinary per-source maximum per format/arm",
          "location": "cover at least one half of changed pixels and 0.30 IoU",
          "held_sources": "historically exposed overall, source-disjoint from this method's fit and point",
          "advance": GATE, "confidence_weight": 0}


def describe(rgb):
    h, w = (size // earlier.CELL for size in rgb.shape[:2])
    cell = earlier.CELL
    color = rgb.astype(np.float32)
    red, green, blue = color.transpose(2, 0, 1)
    phase = ((np.arange(h * cell)[:, None] + np.arange(w * cell)[None, :]) % 2).astype(np.float32)

    def local(value):
        tiles = value[:h * cell, :w * cell].reshape(h, cell, w, cell).mean(axis=(1, 3))
        return cv2.blur(tiles, (3, 3))

    fixed, fitted = [], []
    for plane in (0.299 * red + .587 * green + .114 * blue, red - green, blue - green):
        pad = np.pad(plane, 1, mode="reflect")
        x = (pad[1:-1, :-2] + pad[1:-1, 2:]) * .5
        y = (pad[:-2, 1:-1] + pad[2:, 1:-1]) * .5
        avg, delta = (x + y) * .5, (x - y) * .5
        err = (plane - avg)[:h * cell, :w * cell]
        delta = delta[:h * cell, :w * cell]
        squared, axis_disagreement, cross = err ** 2, delta ** 2, err * delta
        base, spread, relation = local(squared), local(axis_disagreement), local(cross)
        fixed.extend([np.log1p(base), np.log1p(spread), np.log((base + 4.) / (spread + 4.))])
        errors, weights = [], []
        for fold in (phase, 1. - phase):
            train_delta = 2 * local(axis_disagreement * fold)
            train_relation = 2 * local(cross * fold)
            weight = np.clip(train_relation / (train_delta + 4.), -1., 1.)
            test_delta = 2 * spread - train_delta
            test_relation = 2 * relation - train_relation
            test_error = 2 * base - 2 * local(squared * fold)
            errors.append(np.maximum(0, test_error - 2 * weight * test_relation + weight ** 2 * test_delta))
            weights.append(weight)
        holdout = (errors[0] + errors[1]) * .5
        fitted.extend([np.log((holdout + 4.) / (base + 4.)),
                       np.abs(weights[0] - weights[1]), np.abs((weights[0] + weights[1]) * .5)])
    a, b = np.stack(fixed, axis=-1), np.stack(fitted, axis=-1)
    return a, b, np.concatenate((a, b), axis=-1)


def fit(pairs):
    moments = defaultdict(list)
    for pair in pairs:
        cells = np.argwhere(archive.occupancy(pair[-1]) >= .75)
        if len(cells) < 3:
            continue
        y, x = cells[np.linspace(0, len(cells) - 1, min(len(cells), 48), dtype=int)].T
        groups = defaultdict(dict)
        for row, body, _ in earlier.inputs([pair], "fit"):
            groups[row["format"]][row["role"]] = tuple(field[y, x] for field in describe(earlier.decoded(body)))
        for fmt, group in groups.items():
            for index, arm in enumerate(ARMS):
                edited = np.concatenate([group[role][index] for role in earlier.POSITIVE])
                ordinary = np.concatenate([group[role][index] for role in CONTROL_ROLES])
                diff = edited.mean(axis=0) - ordinary.mean(axis=0)
                center = (edited.mean(axis=0) + ordinary.mean(axis=0)) * .5
                a, b = edited - center, ordinary - center
                moments[fmt, arm].append((diff, (a.T @ a / len(a) + b.T @ b / len(b)) * .5))
    weights = defaultdict(dict)
    for (fmt, arm), rows in moments.items():
        archive.require(len(rows) >= 12, "Insufficient covered image groups")
        differences, covariances = zip(*rows, strict=True)
        covariance = np.mean(covariances, axis=0)
        penalty = .1 * np.diag(np.diag(covariance).clip(.001))
        weights[fmt][arm] = np.linalg.solve(covariance + penalty, np.mean(differences, axis=0)).tolist()
    return weights


def measure(pairs, part, weights, baseline):
    for i, (row, body, changed) in enumerate(earlier.inputs(pairs, part), 1):
        positive = row["role"] in earlier.POSITIVE
        truth = changed if positive else np.zeros_like(changed)
        checks = {}
        for arm, field in zip(ARMS, describe(earlier.decoded(body)), strict=True):
            box = earlier.proposal(field @ np.asarray(weights[row["format"]][arm]))
            checks[arm] = box | {"overlap": pixels(box["box"], truth)} if box else None
        if baseline is not None:
            matched = baseline[identity(row)]
            archive.require(row == {k: matched[k] for k in row}, "Comparator used a different upload, source or mask")
            checks["incumbent"] = matched["checks"]["incumbent"]
        yield row | {"checks": checks}
        if i % 72 == 0:
            print(f"{part}: measured {i} deliveries", flush=True)


def points(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        source = row["source"]
        for arm, result in row["checks"].items():
            scores = groups[row["format"]][arm]
            scores[source] = max(scores.get(source, 0.), result["score"] if result else 0.)
    archive.require(all(len(scores) == 24 for arms in groups.values() for scores in arms.values()), "Point roster changed")
    return {fmt: {arm: sorted(scores.values())[-2] for arm, scores in arms.items()} for fmt, arms in groups.items()}


def identity(row):
    return row["source"], row["format"], row["role"]


def advance(summary):
    passing = []
    for fmt in earlier.FORMATS:
        for arm in ARMS:
            main = summary["tgif"][f"{fmt}/{arm}"]
            external = summary["bgreal"][f"{fmt}/{arm}"]
            if (main["new_feathered_groups"] >= GATE["main_new_feathered_sources"]
                    and external["new_feathered_groups"] >= GATE["cross_dataset_new_feathered_sources"]
                    and all(entry["new_ordinary_groups"] <= GATE["max_new_control_groups_each"]
                            and entry.get("misplaced", 0) <= GATE["max_misplaced_each"] for entry in (main, external))):
                passing.append(f"{fmt}/{arm}")
    return passing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "calibrate", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path, default=Path("backend/data/localization-eval"))
    parser.add_argument("--bg", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("docs/evaluation-panels/image-residual-third-order-transfer-2026-11-13.json.gz"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    groups = archive.sources(args)
    pairs = {part: list(archive.archive_pairs(groups, part)) for part in ("fit", "calibration", "held")}
    pairs["tgif"] = pairs.pop("held")
    bg = (args.bg / "plan.json").read_bytes()
    archive.require(archive.digest(bg) == earlier.BG_PLAN, "Unexpected BG-REAL source roster")
    pairs["bgreal"] = list(pixel_pairs(args.bg, json.loads(bg)))
    sources = [row[0] for rows in pairs.values() for row in rows]
    archive.require(len(sources) == len(set(sources)) == 108, "Fit, point, and test groups must be disjoint")
    freeze = {"schema": SCHEMA, "design": DESIGN, "script_sha256": archive.digest(Path(__file__).read_bytes()),
              "baseline_sha256": archive.digest(args.baseline.read_bytes()),
              "inputs": {part: [row for row, *_ in earlier.inputs(rows, part)] for part, rows in pairs.items()}}
    path = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        earlier.write(path, freeze)
        print({part: len(rows) for part, rows in freeze["inputs"].items()}, flush=True)
        return
    archive.require(freeze == earlier.read(path), "Analysis code or source input differs from the freeze")
    calibration = args.run / "calibration.json"
    if args.phase == "calibrate":
        weights = fit(pairs["fit"])
        rows = list(measure(pairs["calibration"], "calibration", weights, None))
        state = {"weights": weights, "points": points(rows), "calibration": rows}
        earlier.write(calibration, state)
        print(state["points"], flush=True)
        return
    state = earlier.read(calibration)
    archive.require(state["points"] == points(state["calibration"]), "Expected unchanged operating points")
    with gzip.open(args.baseline, "rt") as stream:
        baseline = json.load(stream)
    results = {}
    for part in ("tgif", "bgreal"):
        index = {identity(row): row for row in baseline[part]}
        results[part] = list(measure(pairs[part], part, state["weights"], index))
    # Use the exact predecessor counting method with this runner's independent arms.
    earlier.ARMS = ARMS
    summary = {part: earlier.summarize(rows, state["points"]) for part, rows in results.items()}
    earlier.write(args.run / "receipt.json", {"freeze": freeze, **state, **results, "summary": summary, "advance": advance(summary)})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
