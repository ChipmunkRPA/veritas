# SPDX-License-Identifier: Apache-2.0
"""Test cross-fitted local color-line errors on recompressed small edits."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts import image_directional_structure_transfer as panel

SCHEMA = "chipmunk.image-color-line-transfer/1"
ARMS = ("fine_excess", "cross_scale")
CELL, POOL, RING = 16, 2, 3
SIDES = (3, 4, 5, 6, 8, 10, 12)
MIN_SUPPORT, COARSE_SIGMA = .50, 1.8
LUMA = np.array([.2126, .7152, .0722])
LEVELS = np.arange(256, dtype=np.float64) / 255.
LINEAR_RGB = np.where(LEVELS <= .04045, LEVELS / 12.92, ((LEVELS + .055) / 1.055)**2.4)
DESIGN = {
    "signal": "Held-out linear RGB distance from the fitted local color axis at native and blurred scales.",
    "arms": {
        "fine_excess": "Independent fine minus coarse axis errors with test-scale normalization.",
        "cross_scale": "Fine error using the fitted coarse axis minus fine error using the fitted fine axis.",
    },
    "cell_px": CELL, "pool_px": POOL, "gaussian_sigma_px": COARSE_SIGMA,
    "cross_fit": "Train and test on opposite checkerboard halves of pooled samples, then swap.",
    "eligibility": {"coarse_rms": .012, "fine_rms": .015, "max_clipped_fraction": .15,
                    "max_coarse_error": .32, "min_support": MIN_SUPPORT, "min_matched_cells": 4,
                    "coarse_luma_edges": [.12, .36], "coarse_strength_edges": [.035, .10]},
    "candidate_geometry": {"native_cell_sides": list(SIDES), "max_aspect_ratio": 2, "max_area_fraction": .10,
                           "surrounding_strips": RING, "max_one_per_arm": True},
    "point": panel.DESIGN["point"], "gates": panel.DESIGN["gates"], "useful": panel.DESIGN["useful"],
    "confidence_weight": 0,
}


def cells(rgb):
    """Return complete native blocks of 2×2-cell-averaged fine and blurred colors."""
    height, width = (size // CELL * CELL for size in rgb.shape[:2])
    native = LINEAR_RGB[rgb]
    blurred = cv2.GaussianBlur(native, (0, 0), COARSE_SIGMA)
    edge = CELL // POOL

    def blocks(samples):
        inner = samples[:height, :width]
        pooled = inner.reshape(height//POOL, POOL, width//POOL, POOL, 3).mean(axis=(1, 3))
        return pooled.reshape(height//CELL, edge, width//CELL, edge, 3).transpose(0, 2, 1, 3, 4).reshape(
            height//CELL, width//CELL, edge*edge, 3,
        )

    clipped = np.any((rgb[:height, :width] <= 3) | (rgb[:height, :width] >= 252), axis=-1)
    clipped_fraction = clipped.reshape(height//CELL, CELL, width//CELL, CELL).mean(axis=(1, 3))
    return blocks(native), blocks(blurred), clipped_fraction


def color_axis(samples):
    mean = samples.mean(axis=-2, keepdims=True)
    centered = samples - mean
    covariance = centered.swapaxes(-1, -2) @ centered
    _, axes = np.linalg.eigh(covariance)
    return mean, axes[..., -1]


def fractions(fine, coarse):
    """Measure held-out off-axis energy relative to total color variation."""
    parity = np.indices((CELL//POOL, CELL//POOL)).sum(axis=0).reshape(-1) % 2
    measurements, norms = [], []
    for half in range(2):
        train = parity == half
        fine_mean, fine_axis = color_axis(fine[..., train, :])
        coarse_mean, coarse_axis = color_axis(coarse[..., train, :])
        fine_test = fine[..., ~train, :] - fine_mean
        coarse_test = coarse[..., ~train, :] - coarse_mean
        fine_total = np.square(fine_test).sum(axis=(-2, -1))
        coarse_total = np.square(coarse_test).sum(axis=(-2, -1))

        def off_axis(samples, axis, total):
            parallel = np.square(np.einsum("...sc,...c->...s", samples, axis)).sum(axis=-1)
            return np.clip(1. - parallel / np.maximum(total, 1e-12), 0., 1.)

        fine_error = off_axis(fine_test, fine_axis, fine_total)
        coarse_error = off_axis(coarse_test, coarse_axis, coarse_total)
        transfer_error = off_axis(fine_test, coarse_axis, fine_total)
        measurements.append(np.stack((fine_error, coarse_error, transfer_error), axis=-1))
        norms.append(np.sqrt(np.stack((fine_total, coarse_total), axis=-1) / np.count_nonzero(~train)))
    return np.stack(measurements, axis=-2), np.stack(norms, axis=-2)


def maps(rgb):
    fine, coarse, clipped = cells(rgb)
    errors, rms = fractions(fine, coarse)
    valid = ((rms[..., 0].min(axis=-1) >= .015) & (rms[..., 1].min(axis=-1) >= .012)
             & (errors[..., 1].max(axis=-1) <= .32) & (clipped < .15))
    means = coarse.mean(axis=-2)
    brightness = np.searchsorted((.12, .36), means @ LUMA)
    strength = np.searchsorted((.035, .10), rms[..., 1].mean(axis=-1))
    observed = np.stack(((errors[..., 0]-errors[..., 1]).min(axis=-1),
                         (errors[..., 2]-errors[..., 0]).min(axis=-1)), axis=-1)
    return observed, valid, brightness + 3*strength


def integral_maps(values, valid, categories):
    indicators = (categories[..., None] == np.arange(9)) & valid[..., None]
    fields = np.concatenate((indicators[..., None], values[..., None, :] * indicators[..., None]), axis=-1)
    return np.pad(fields.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0), (0, 0)))


def compare(core, context):
    weight = np.minimum(core[..., 0], context[..., 0])
    differences = core[..., 1:] / np.maximum(1., core[..., :1]) - context[..., 1:] / np.maximum(1., context[..., :1])
    count = weight.sum(axis=-1)
    contrast = (differences * weight[..., None]).sum(axis=-2) / np.maximum(count[..., None], 1.)
    return contrast, count


def proposals(rgb):
    values, valid, categories = maps(rgb)
    table = integral_maps(values, valid, categories)
    rows, columns = valid.shape
    best = {arm: {"box": None, "score": None} for arm in ARMS}
    rectangle = panel.rectangle
    for h in SIDES:
        for w in SIDES:
            if max(h, w) > 2*min(h, w) or h*w > .10*rows*columns or h + 2*RING > rows or w + 2*RING > columns:
                continue
            x, y = np.meshgrid(np.arange(RING, columns-w-RING+1), np.arange(RING, rows-h-RING+1))
            core = rectangle(table, x, y, w, h)
            left = rectangle(table, x-RING, y, RING, h)
            right = rectangle(table, x+w, y, RING, h)
            top = rectangle(table, x, y-RING, w, RING)
            bottom = rectangle(table, x, y+h, w, RING)
            opposite = []
            for first, second in ((left, right), (top, bottom)):
                a, a_support = compare(core, first)
                b, b_support = compare(core, second)
                matched = np.minimum(a_support, b_support) >= max(4, h*w*MIN_SUPPORT)
                opposite.append(np.where(matched[..., None], np.minimum(a, b), -np.inf))
            score = np.maximum(*opposite) * (h*w/9)**.10
            for index, arm in enumerate(ARMS):
                field = score[..., index]
                j, i = np.unravel_index(field.argmax(), field.shape)
                value = float(field[j, i])
                if np.isfinite(value) and (best[arm]["score"] is None or value > best[arm]["score"]):
                    left_px, top_px = int(x[j, i])*CELL, int(y[j, i])*CELL
                    best[arm] = {"box": [left_px, top_px, left_px+w*CELL, top_px+h*CELL], "score": value}
    return best


def synthetic():
    """Check a one-axis field and the same coarse field with a localized extra axis."""
    y, x = np.mgrid[:384, :384]
    ramp = np.sin(x*np.pi/10) * .08 + np.cos(y*np.pi/18) * .0336
    baseline = np.stack((.34+ramp, .40+ramp*.8, .30+ramp*.5), axis=-1)
    changed = baseline.copy()
    delta = (np.sin(x[144:240, 144:240]*np.pi/3) * np.cos(y[144:240, 144:240]*np.pi/3)) * .22
    changed[144:240, 144:240] += delta[..., None] * np.array([1., -.75, -.8])

    def srgb(linear):
        return np.rint((1.055 * linear**(1/2.4) - .055) * 255).astype(np.uint8)

    base, edited = proposals(srgb(baseline)), proposals(srgb(changed))
    mask = np.zeros((384, 384), bool)
    mask[144:240, 144:240] = True
    for arm in ARMS:
        score = edited[arm]["score"]
        useful = panel.previous.prior.base.geometry.useful(panel.previous.prior.overlap(mask, edited[arm]["box"]))
        panel.require(useful and score > max(.02, base[arm]["score"]), "Known color-line defect should localize")
    flat = proposals(np.full((384, 384, 3), 120, dtype=np.uint8))
    panel.require(not any(hit["box"] for hit in flat.values()), "Flat input lacks a supported color line")
    return {"baseline": base, "altered": edited, "flat": flat}


def measure(rows, points=None):
    base = panel.previous.prior.base
    for key, body, truth, incumbent in rows:
        panel.require(key["input_sha256"] == panel.sha(body) and key["truth_sha256"] == panel.sha(truth.tobytes()),
                      "Exact source or mask differs")
        rgb = base.decode(body)
        hits = proposals(rgb)
        if points is not None:
            if incumbent is None:
                reference = base.comparison.format_reference("jpeg75", body, rgb)
                incumbent = {"box": reference["box"] if reference and reference["selected"] else None}
            hits["current_luma"] = {"box": incumbent["box"], "score": None}
        for arm, hit in hits.items():
            overlap = panel.previous.prior.overlap(truth, hit["box"])
            hit.update(overlap=overlap, useful=base.geometry.useful(overlap))
            if points is not None:
                hit["flagged"] = bool(hit["box"]) and (arm == "current_luma" or hit["score"] > points[arm])
        yield key | {"arms": hits}


def summarize(rows):
    stats, ordinary, soft = defaultdict(Counter), defaultdict(set), defaultdict(set)
    wrong = Counter()
    for row in rows:
        role, source, family = (row[key] for key in ("role", "source", "family"))
        positive = role in panel.previous.prior.HELD_POSITIVE
        for arm, hit in row["arms"].items():
            useful = hit["useful"] and hit["flagged"]
            misplaced = positive and hit["flagged"] and not useful
            stats[f"{family}/{role if positive else 'ordinary'}/{arm}"].update(
                files=1, assessed=bool(hit["box"]), flags=hit["flagged"], useful=useful,
                useful_before=hit["useful"], misplaced=misplaced,
            )
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
    destination = args.output
    cv2.setNumThreads(1)
    lock, _, _ = panel.previous.fresh.previous(args)
    paths = (args.prior_probe / "calibration.json", args.output_fresh / "measurements.json",
             args.external / "plan.json", args.external / "intake.json", Path(panel.__file__), Path(__file__))
    commit = {f"{path.parent.name}/{path.name}": panel.external.roster.sha(path) for path in paths}
    freeze = {"schema": SCHEMA, "design": DESIGN, "commitments": commit, "synthetic": synthetic()}
    panel.require(commit[f"{paths[0].parent.name}/calibration.json"] ==
                  panel.read(args.prior_probe / "result.json")["commitments"]["calibration.json"], "Calibration source journal differs")
    lock_path = destination / "freeze.json"
    if args.stage == "freeze":
        destination.mkdir(mode=0o700)
        panel.write(lock_path, freeze)
        return
    panel.require(panel.read(lock_path) == freeze, "Frozen study differs")
    reports, cutoffs = {}, None
    start = time.monotonic()
    for phase, inputs in panel.bindings(args, lock):
        rows = list(measure(inputs, cutoffs))
        panel.require(len(rows) == {"calibration": 40, "confirmation": 96, "external": 64}[phase], "Study input count differs")
        output = destination / f"{phase}.json"
        if args.stage == "run":
            panel.write(output, rows)
        else:
            panel.require(rows == panel.read(output), "Numerical replay differs")
        if phase == "calibration":
            maxima = {arm: {} for arm in ARMS}
            for row in rows:
                for arm in ARMS:
                    score = row["arms"][arm]["score"]
                    maxima[arm][row["source"]] = max(maxima[arm].get(row["source"], -np.inf), -np.inf if score is None else score)
            panel.require(all(len(groups) == 8 for groups in maxima.values()), "Eight separate calibration groups required")
            cutoffs = {arm: max(0., sorted(groups.values())[-2]) for arm, groups in maxima.items()}
        else:
            reports[phase] = summarize(rows)
        print(phase, len(rows), flush=True)
    limits = DESIGN["gates"]
    confirmation = reports["confirmation"]["advancement"]
    passes = {arm: all(row[key] >= limit if key in {"extra_feathered_groups", "editor_families"} else row[key] <= limit
                       for key, limit in limits.items()) for arm, row in confirmation.items()}
    result = freeze | {"points": cutoffs, "results": reports, "passes": passes}
    if args.stage == "run":
        panel.write(destination / "result.json", result)
    else:
        panel.require(result == panel.read(destination / "result.json"), "Replay totals differ")
        panel.write(args.publication, result | {"exact_native_replays": 200, "replay_seconds": round(time.monotonic()-start, 2)})
    print(confirmation, round(time.monotonic()-start, 2), flush=True)


if __name__ == "__main__":
    main()
