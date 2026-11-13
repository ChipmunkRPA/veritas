# SPDX-License-Identifier: Apache-2.0
"""Test matched local changes in coarse/fine gradient direction for JPEG edits."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts import image_masked_context_new_editors as external
from scripts import image_masked_context_transfer as previous

SCHEMA = "chipmunk.image-directional-structure-transfer/1"
ARMS = ("perpendicular", "paired")
CELL, STEP, STRIP = 16, 3, 2
SIDES = (3, 4, 5, 6, 8, 10, 12)
STRATA, MIN_COMMON, MIN_IN_STRATUM = 12, 2, 12
DESIGN = {
    "cue": "At the same source pixel, measure fine-gradient perpendicular energy share and reversal relative to the broad gradient.",
    "gaussian_sigmas": [.8, 4.], "sample_stride": STEP, "cell_px": CELL,
    "matching": "Four coarse orientation bins and three magnitude bins. Require same strata in core and both opposed flanks.",
    "gradient_support": {"fine_min": 1.6, "coarse_min": 1., "coarse_breaks": [2.5, 6.], "unclipped": [5, 250]},
    "native_sides": list(SIDES), "flank_cells": STRIP, "arms": list(ARMS),
    "ranking": "For each feature use the smaller excess above both flanks; the paired arm requires both features on one shared axis.",
    "gates": {"extra_feathered_groups": 3, "editor_families": 2, "extra_ordinary_groups": 2, "misplaced": 4},
    "point": "Strict exceedance of second-largest group maximum on eight ordinary-only calibration sources.",
    "useful": previous.prior.DESIGN["useful"], "confidence_weight": 0,
}
read, write, require, sha = previous.read, previous.write, previous.require, previous.sha


def gradients(rgb):
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def derivative(scale):
        smooth = cv2.GaussianBlur(gray, (0, 0), scale)
        return tuple(cv2.Sobel(smooth, cv2.CV_32F, dx, dy, ksize=3, scale=.125)
                     for dx, dy in ((1, 0), (0, 1)))

    fx, fy = derivative(.8)
    cx, cy = derivative(4.)
    fine, coarse = np.hypot(fx, fy), np.hypot(cx, cy)
    cosine = np.clip((fx*cx + fy*cy) / np.maximum(1e-6, fine*coarse), -1., 1.)
    orient = np.minimum(3, np.floor(np.mod(np.arctan2(cy, cx), np.pi) * 4 / np.pi).astype(np.int8))
    bands = np.searchsorted((2.5, 6.), coarse)
    valid = (fine >= 1.6) & (coarse >= 1.) & (gray >= 5.) & (gray <= 250.)
    sampled = np.zeros((height, width), bool)
    sampled[1::STEP, 1::STEP] = True
    width, height = width // CELL * CELL, height // CELL * CELL
    return (orient + 4*bands)[:height, :width], (sampled & valid)[:height, :width], cosine[:height, :width]


def integral_fields(rgb):
    strata, valid, cosine = gradients(rgb)
    height, width = strata.shape

    def cell_sum(plane):
        return plane.reshape(height//CELL, CELL, width//CELL, CELL).sum(axis=(1, 3), dtype=np.float64)

    fields = []
    for category in range(STRATA):
        eligible = (strata == category) & valid
        fields.append(np.stack((cell_sum(eligible), cell_sum(np.where(eligible, 1 - cosine**2, 0.)),
                                cell_sum(eligible & (cosine < -.25))), axis=-1))
    combined = np.stack(fields, axis=-2)
    return np.pad(combined.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0), (0, 0)))


def rectangle(table, x, y, width, height):
    return table[y+height, x+width] + table[y, x] - table[y, x+width] - table[y+height, x]


def contrast(core, near, far):
    supported = (np.minimum.reduce((core[..., 0], near[..., 0], far[..., 0])) >= MIN_IN_STRATUM)
    weights = np.where(supported, np.minimum.reduce((core[..., 0], near[..., 0], far[..., 0])), 0.)
    samples = weights.sum(axis=-1)
    share = core[..., 1:] / np.maximum(1., core[..., :1])
    offsets = [share - strip[..., 1:] / np.maximum(1., strip[..., :1]) for strip in (near, far)]
    excess = np.minimum(*[(d * weights[..., None]).sum(axis=-2) / np.maximum(1., samples[..., None])
                          for d in offsets])
    return np.where((supported.sum(axis=-1) >= MIN_COMMON)[..., None], excess, -np.inf)


def proposals(rgb):
    table = integral_fields(rgb)
    rows, cols = (size - 1 for size in table.shape[:2])
    best = {arm: {"box": None, "score": None} for arm in ARMS}
    for h in SIDES:
        for w in SIDES:
            if max(h, w) > 2*min(h, w) or h*w > .10*rows*cols or h + 2*STRIP > rows or w + 2*STRIP > cols:
                continue
            x, y = np.meshgrid(np.arange(STRIP, cols - w - STRIP + 1), np.arange(STRIP, rows - h - STRIP + 1))
            core = rectangle(table, x, y, w, h)
            left, right = rectangle(table, x-STRIP, y, STRIP, h), rectangle(table, x+w, y, STRIP, h)
            top, bottom = rectangle(table, x, y-STRIP, w, STRIP), rectangle(table, x, y+h, w, STRIP)
            left_right, top_bottom = contrast(core, left, right), contrast(core, top, bottom)
            axes = np.stack((left_right, top_bottom), axis=-2)
            axial = axes[..., 0].max(axis=-1)
            joint = (axes * [1., 2.]).min(axis=-1).max(axis=-1)
            for arm, field in zip(ARMS, (axial, joint), strict=True):
                j, i = np.unravel_index(field.argmax(), field.shape)
                score = float(field[j, i] * (h*w/9)**.10)
                if np.isfinite(score) and (best[arm]["score"] is None or score > best[arm]["score"]):
                    top, left = int(y[j, i])*CELL, int(x[j, i])*CELL
                    best[arm] = {"box": [left, top, left+w*CELL, top+h*CELL], "score": score}
    return best


def synthetic():
    # Two equally supported edge strata. Move the fine direction toward perpendicular and reverse in the core.
    ring = np.array([[[64., 4., 3.], [40., 2., 1.]]])
    core = np.array([[[64., 48., 24.], [40., 30., 12.]]])
    baseline, change = contrast(ring, ring, ring), contrast(core, ring, ring)
    require(np.array_equal(baseline, np.zeros((1, 2))) and np.all(change > .25), "Directional measurement control failed")
    flat = proposals(np.full((384, 384, 3), 120, dtype=np.uint8))
    require(not any(hit["box"] for hit in flat.values()), "Flat image has no matched edge structure")
    return {"unchanged": baseline.tolist(), "altered_direction": change.tolist(), "flat_regions": 0}


def bindings(args, lock):
    for role, views in previous.inputs(args, lock):
        yield role, ((key, body, mask, old["arms"].get("current_luma")) for old, key, body, mask in views)
    plan, known = external.source_plan(args, lock)
    saved = read(args.external / "plan.json")
    require(all(saved[key] == value for key, value in plan.items()), "Fresh-editor transfer source plan differs")
    old_output = args.output
    args.output = args.external
    try:
        yield "external", ((key, body, mask, None) for key, body, mask in external.views(args, plan, known))
    finally:
        args.output = old_output


def measure(rows, points=None):
    for key, body, truth, incumbent in rows:
        require(key["input_sha256"] == sha(body) and key["truth_sha256"] == sha(truth.tobytes()), "Exact source differs")
        rgb = previous.prior.base.decode(body)
        hits = proposals(rgb)
        if points is not None:
            if incumbent is None:
                reference = previous.prior.base.comparison.format_reference("jpeg75", body, rgb)
                box = reference["box"] if reference and reference["selected"] else None
                hits["current_luma"] = {"box": box, "score": None}
            else:
                hits["current_luma"] = {"box": incumbent["box"], "score": None}
        for arm, hit in hits.items():
            overlap = previous.prior.overlap(truth, hit["box"])
            hit.update(overlap=overlap, useful=previous.prior.base.geometry.useful(overlap))
            if points is not None:
                hit["flagged"] = bool(hit["box"]) and (arm == "current_luma" or hit["score"] > points[arm])
        yield key | {"arms": hits}


def points(rows):
    maxima = {arm: defaultdict(lambda: -np.inf) for arm in ARMS}
    for row in rows:
        for arm, groups in maxima.items():
            score = row["arms"][arm]["score"]
            groups[row["source"]] = max(groups[row["source"]], -np.inf if score is None else score)
    require(all(len(group) == 8 for group in maxima.values()), "Eight ordinary calibration groups required")
    return {arm: max(0., sorted(groups.values())[-2]) for arm, groups in maxima.items()}


def summarize(rows):
    stats, ordinary, soft = defaultdict(Counter), defaultdict(set), defaultdict(set)
    wrong = Counter()
    for row in rows:
        role, source, family = (row[key] for key in ("role", "source", "family"))
        positive = role in previous.prior.HELD_POSITIVE
        for arm, hit in row["arms"].items():
            useful = hit["useful"] and hit["flagged"]
            misplaced = positive and hit["flagged"] and not useful
            stats[f"{family}/{role if positive else 'ordinary'}/{arm}"].update(
                files=1, assessed=bool(hit["box"]), flags=hit["flagged"], useful=useful, useful_before=hit["useful"],
                misplaced=misplaced)
            wrong[arm] += misplaced
            if useful and role == "feathered":
                soft[arm].add((family, source))
            if not positive and hit["flagged"]:
                ordinary[arm].add(source)
    result = {}
    for arm in ARMS:
        gains = soft[arm] - soft["current_luma"]
        result[arm] = {"extra_feathered_groups": len(gains), "editor_families": len({f for f, _ in gains}),
                       "extra_ordinary_groups": len(ordinary[arm] - ordinary["current_luma"]), "misplaced": wrong[arm]}
    return {"strata": stats, "advancement": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    lock, _, _ = previous.fresh.previous(args)
    paths = (args.prior_probe / "calibration.json", args.output_fresh / "measurements.json",
             args.external / "plan.json", args.external / "intake.json")
    freeze = {"schema": SCHEMA, "design": DESIGN, "commitments": {f"{path.parent.name}/{path.name}": external.roster.sha(path)
              for path in (*paths, Path(__file__))}, "synthetic": synthetic()}
    require(freeze["commitments"][f"{paths[0].parent.name}/calibration.json"] ==
            read(args.prior_probe / "result.json")["commitments"]["calibration.json"], "Calibration source journal differs")
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, freeze)
        return
    require(read(freeze_path) == freeze, "Frozen study differs")
    reports, cutoffs = {}, None
    start = time.monotonic()
    for phase, deliveries in bindings(args, lock):
        rows = list(measure(deliveries, cutoffs))
        require(len(rows) == {"calibration": 40, "confirmation": 96, "external": 64}[phase], "Study input count differs")
        path = args.output / f"{phase}.json"
        if args.stage == "run":
            write(path, rows)
        else:
            require(rows == read(path), "Actual measurement replay differs")
        if phase == "calibration":
            cutoffs = points(rows)
        else:
            reports[phase] = summarize(rows)
        print(phase, len(rows), flush=True)
    confirmation = reports["confirmation"]["advancement"]
    gate = DESIGN["gates"]
    passing = {arm: all(row[key] >= limit if key in {"extra_feathered_groups", "editor_families"} else row[key] <= limit
                        for key, limit in gate.items()) for arm, row in confirmation.items()}
    result = freeze | {"points": cutoffs, "results": reports, "passes": passing}
    if args.stage == "run":
        write(args.output / "result.json", result)
    else:
        require(result == read(args.output / "result.json"), "Numerical summaries differ on replay")
        write(args.publication, result | {"exact_native_replays": 200, "replay_seconds": round(time.monotonic() - start, 2)})
    print(confirmation, round(time.monotonic() - start, 2), flush=True)


if __name__ == "__main__":
    main()
