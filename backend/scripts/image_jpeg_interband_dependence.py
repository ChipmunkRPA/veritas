# SPDX-License-Identifier: Apache-2.0
"""Test native JPEG local cross-scale association at fixed marginal distributions."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts import image_jpeg_contour_gap_transfer as panels
from scripts import image_residual_encoder_transfer as base
from scripts.review_image_residual_encoder_transfer import overlap

SCHEMA = "chipmunk.image-jpeg-interband-dependence/1"
ARMS = ("amplitude", "polarity", "agreement")
CELL, RING = 16, 2
SIDES = (3, 4, 5, 6, 8, 10, 12, 14)
SIGMAS = (1.0, 2.0, 4.0)
FLOORS = (.12, .12)
POINT_FLOOR = .01
DESIGN = {
    "jpeg_quality": list(base.QUALITIES), "subsampling": "4:2:0", "cell_px": CELL,
    "directional_gaussian_sigmas_px": list(SIGMAS), "native_sampling": "horizontal and vertical scale-normalized derivatives",
    "joint_statistics": "Kendall tau-b on absolute responses and signed polarity coupling with marginal centering",
    "paired_control": "Subtract mean coupling of the exact same sampled coarse values cyclically shifted by 2 or 3 positions on either axis.",
    "support": "Both scale pairs need enough cells in core and ring. Exclude cells with 15% clipped or mean parent band activity under 1.5.",
    "deficit": "Parent-child coupling lower than the surrounding two-cell ring, with a minimum deficit in both adjacent scale pairs.",
    "arms": list(ARMS), "search_sides_cells": list(SIDES), "maximum_aspect_ratio": 2,
    "maximum_area_share": .10, "min_spread_floor": list(FLOORS), "min_point": POINT_FLOOR,
    "calibration": "Separate operating points on 12 ordinary source groups. Use strict exceedance of second largest group maximum or minimum point.",
    "held": "Separate retained 12- and 16-photo F5 panels, previously exposed under other methods. Original, same-site local blur, tone, copy and opaque overlay, global resize and second JPEG, plus matched local/global processing when provided.",
    "comparator": "Current JPEG aligned luma on exact same encoded bytes.",
    "useful": {"minimum_iou": .30, "minimum_recall": .50}, "confidence_weight": 0,
    "advance": "At least three useful added feathered groups across both editor families, at most two additional ordinary groups and four misplaced edit flags per quality.",
}
SAMPLE_AXIS = np.arange(1, CELL, 2)
_rng = np.random.default_rng(3921)
_left, _right = np.triu_indices(len(SAMPLE_AXIS) ** 2, k=1)
_selected = _rng.choice(len(_left), 192, replace=False)
PAIR_I, PAIR_J = _left[_selected], _right[_selected]
read, write, require, sha = base.read, base.write, base.require, base.sha


def cells(plane):
    rows, cols = (size // CELL for size in plane.shape)
    return plane[:rows*CELL, :cols*CELL].reshape(rows, CELL, cols, CELL).transpose(0, 2, 1, 3)


def differences(cube):
    return np.sign(cube[..., PAIR_I] - cube[..., PAIR_J]).astype(np.int8)


def joint(first, second):
    rows, cols = first.shape[:2]
    flat = first[..., SAMPLE_AXIS[:, None], SAMPLE_AXIS].reshape(rows, cols, -1)
    coarse = second[..., SAMPLE_AXIS[:, None], SAMPLE_AXIS]
    second_views = [coarse] + [np.roll(coarse, step, axis=axis) for axis in (-2, -1) for step in (2, 3)]
    amp_a = differences(np.abs(flat))
    sign_a = np.sign(flat)
    anorm = np.square(amp_a, dtype=np.int8).sum(axis=-1)
    a_center = sign_a - sign_a.mean(axis=-1, keepdims=True)
    a_pow = np.square(a_center).sum(axis=-1)
    measures = []
    for view in second_views:
        flat_b = view.reshape(rows, cols, -1)
        amp_b = differences(np.abs(flat_b))
        bnorm = np.square(amp_b, dtype=np.int8).sum(axis=-1)
        kendall = (amp_a * amp_b).sum(axis=-1) / np.sqrt(np.maximum(1, anorm * bnorm))
        sign_b = np.sign(flat_b)
        b_center = sign_b - sign_b.mean(axis=-1, keepdims=True)
        signed = (a_center * b_center).sum(axis=-1) / np.sqrt(np.maximum(1., a_pow * np.square(b_center).sum(axis=-1)))
        measures.append(np.stack((kendall, signed), axis=-1))
    return measures[0] - np.mean(measures[1:], axis=0)


def plane_statistics(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source = cells(gray)
    valid = ((source < 4) | (source > 251)).mean(axis=(-2, -1)) < .15
    luma = [cv2.GaussianBlur(gray, (0, 0), sigma) for sigma in SIGMAS]
    results = []
    for orientation in ((1, 0), (0, 1)):
        bands = [cells(cv2.Sobel(p, cv2.CV_32F, *orientation, ksize=3, scale=.125 * sigma))
                 for p, sigma in zip(luma, SIGMAS, strict=True)]
        direction = []
        for a, b in zip(bands[:-1], bands[1:], strict=True):
            support = valid & (np.abs(b).mean(axis=(-2, -1)) >= 1.5)
            direction.append(np.where(support[..., None], joint(a, b), np.nan))
        results.append(np.stack(direction, axis=2))
    return np.stack(results, axis=3)


def rectangle(ii, x, y, w, h):
    return ii[y+h, x+w] + ii[y, x] - ii[y+h, x] - ii[y, x+w]


def integral(values):
    pads = ((1, 0), (1, 0)) + ((0, 0),) * (values.ndim - 2)
    return np.pad(values.cumsum(0).cumsum(1), pads)


def proposals(field):
    rows, cols = field.shape[:2]
    valid = np.isfinite(field[..., 0])
    count = integral(valid.astype(np.int32))
    sums = integral(np.nan_to_num(field))
    support = integral(np.nan_to_num(field)**2)
    best = {arm: None for arm in ARMS}
    for h in SIDES:
        for w in SIDES:
            area = h * w
            if max(h, w) > 2 * min(h, w) or area > .10 * rows * cols or h + 2*RING > rows or w + 2*RING > cols:
                continue
            x, y = np.meshgrid(np.arange(RING, cols - w - RING + 1), np.arange(RING, rows - h - RING + 1))
            nc = rectangle(count, x, y, w, h)
            nr = rectangle(count, x-RING, y-RING, w+2*RING, h+2*RING) - nc
            core = rectangle(sums, x, y, w, h)
            ring = rectangle(sums, x-RING, y-RING, w+2*RING, h+2*RING) - core
            square_ring = rectangle(support, x-RING, y-RING, w+2*RING, h+2*RING) - rectangle(support, x, y, w, h)
            m = ring / np.maximum(1, nr[..., None])
            spread = np.sqrt(np.maximum(0, square_ring / np.maximum(1, nr[..., None]) - m**2))
            z = (m - core / np.maximum(1, nc[..., None])) / np.maximum(np.asarray(FLOORS), spread)
            usable = (nc >= max(6, area*.5)) & (nr >= 12) & (m[..., 0] >= .05)
            coherent = np.min(np.where(usable[..., None], z, -np.inf), axis=2)
            # Either orientation may carry a measurable deficit, but agreement uses one common orientation.
            values = [np.max(coherent[..., channel], axis=2) for channel in (0, 1)]
            values.append(np.max(np.min(coherent, axis=-1), axis=2))
            for arm, value in zip(ARMS, values, strict=True):
                value *= (area / 9) ** .12
                j, i = np.unravel_index(value.argmax(), value.shape)
                score = float(value[j, i])
                if np.isfinite(score) and (best[arm] is None or score > best[arm]["score"]):
                    a, b = int(x[j, i]), int(y[j, i])
                    best[arm] = {"box": [CELL*a, CELL*b, CELL*(a+w), CELL*(b+h)], "score": score}
    return {arm: result or {"box": None, "score": None} for arm, result in best.items()}


def views(args, frozen, phase):
    for (item, native, exact), check in zip(panels.groups(args, phase), frozen["sources"][phase], strict=True):
        require(panels.identity(item, native, exact) == check, "Source does not reproduce frozen identity")
        ordinary, positives = base.authored(native, exact, item["recipient"])
        ordinary["annotation_overlay"] = panels.overlay(native["original"], exact, item["recipient"])
        for role, rgb in (ordinary | positives).items():
            mask = np.any(rgb != native["original"], axis=2) if role in base.POSITIVE else np.zeros(rgb.shape[:2], bool)
            for quality in base.QUALITIES:
                body = base.encode(rgb, quality)
                yield {"source": item["recipient"], "phase": phase, "family": item["generator"], "role": role,
                       "quality": quality, "input_sha256": sha(body), "truth_sha256": sha(mask.tobytes())}, body, mask


def fixed_points(rows):
    maxima = defaultdict(dict)
    for row in rows:
        for arm, response in row["result"].items():
            group = maxima[str(row["quality"]), arm]
            value = response["score"]
            group[row["source"]] = max(group.get(row["source"], POINT_FLOOR), value if value is not None else POINT_FLOOR)
    require(all(len(group) == 12 for group in maxima.values()), "Calibration groups changed")
    points = defaultdict(dict)
    for (quality, arm), group in maxima.items():
        points[quality][arm] = sorted(group.values())[-2]
    return points


def assess(args, frozen, phase, points=None):
    rows = []
    for item, body, mask in views(args, frozen, phase):
        rgb = base.decode(body)
        result = proposals(plane_statistics(rgb))
        if points is not None:
            old = base.comparison.format_reference(f"jpeg{item['quality']}", body, rgb)
            result["current_luma"] = {"box": old["box"] if old and old["selected"] else None, "score": None}
        for arm, hit in result.items():
            hit["overlap"] = overlap(mask, hit["box"])
            hit["useful"] = base.geometry.useful(hit["overlap"])
            if points is not None:
                hit["flagged"] = bool(hit["box"]) and (arm == "current_luma" or hit["score"] > points[str(item["quality"])][arm])
        rows.append(item | {"result": result})
        if len(rows) % 72 == 0:
            print(phase, len(rows), flush=True)
    return rows


def summarize(rows):
    sections = defaultdict(Counter)
    outcomes = defaultdict(dict)
    for quality in base.QUALITIES:
        selected = [r for r in rows if r["quality"] == quality]
        ordinary = defaultdict(set)
        gain, misplaced = defaultdict(set), Counter()
        for row in selected:
            source, role, family = (row[key] for key in ("source", "role", "family"))
            for arm, hit in row["result"].items():
                useful = hit["flagged"] and hit["useful"]
                sections[f"{quality}/{row['phase']}/{family}/{role}/{arm}"].update(
                    views=1, ranked=bool(hit["box"]), useful_before_point=hit["useful"], flagged=hit["flagged"], useful=useful)
                if role not in base.POSITIVE and hit["flagged"]:
                    ordinary[arm].add(source)
                if role in base.POSITIVE and hit["flagged"]:
                    misplaced[arm] += not useful
                    if role == "feathered" and useful and not row["result"]["current_luma"]["useful"]:
                        gain[arm].add((family, source))
        for arm in ARMS:
            actual = {"added_feathered_groups": len(gain[arm]), "families": len({family for family, _ in gain[arm]}),
                      "ordinary_groups": len(ordinary[arm]), "current_ordinary_groups": len(ordinary["current_luma"]),
                      "extra_ordinary_groups": len(ordinary[arm] - ordinary["current_luma"]), "wrong_edit_views": misplaced[arm]}
            outcomes[str(quality)][arm] = actual | {"passes": actual["added_feathered_groups"] >= 3 and actual["families"] == 2
                                                   and actual["extra_ordinary_groups"] <= 2 and actual["wrong_edit_views"] <= 4}
    return {"outcomes": outcomes, "strata": sections}


def audit(args, frozen, phase, recorded, points=None):
    checks = 0
    for (item, _, mask), row in zip(views(args, frozen, phase), recorded, strict=True):
        require(all(row[key] == value for key, value in item.items()), "Delivered JPEG or exact truth differs")
        for arm, result in row["result"].items():
            metrics = overlap(mask, result["box"])
            useful = metrics["iou"] is not None and metrics["iou"] >= .30 and metrics["recall"] >= .50
            require(result["overlap"] == metrics and result["useful"] == useful, "Recorded native overlap differs")
            if points is not None:
                flagged = bool(result["box"]) and (arm == "current_luma" or result["score"] > points[str(item["quality"])][arm])
                require(result["flagged"] == flagged, "Flag differs from frozen point")
            checks += 1
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for key in ("archive", "encoder", "earlier", "later", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.rosters = panels.roster.DATA
    cv2.setNumThreads(1)
    profile = panels.plan(args)
    locked = {"schema": SCHEMA, "design": DESIGN, "source_plan": profile, "runner_sha256": sha(Path(__file__).read_bytes()),
              "numpy": np.__version__, "opencv": cv2.__version__}
    folder = args.output
    if args.stage == "freeze":
        folder.mkdir(mode=0o700)
        write(folder / "freeze.json", locked)
        print(sha((folder / "freeze.json").read_bytes()), flush=True)
        return
    require(locked == read(folder / "freeze.json"), "Frozen method, sources or environment differ")
    if args.stage == "run":
        start = time.monotonic()
        cal = assess(args, profile, "calibration")
        write(folder / "calibration.json", cal)
        points = fixed_points(cal)
        write(folder / "points.json", points)
        print("points", dict(points), flush=True)
        held = [row for phase in ("retained_a", "retained_b") for row in assess(args, profile, phase, points)]
        write(folder / "held.json", held)
        result = {"schema": SCHEMA, "design": DESIGN, "points": points, "held_rows": len(held), "calibration_rows": len(cal),
                  "dependencies": {key: locked[key] for key in ("numpy", "opencv", "runner_sha256")},
                  "commitments": {name: sha((folder / name).read_bytes()) for name in ("freeze.json", "held.json", "calibration.json", "points.json")},
                  "seconds": round(time.monotonic() - start, 3), **summarize(held)}
        write(folder / "result.json", result)
        print(result["outcomes"], flush=True)
        return
    result = read(folder / "result.json")
    require(all(sha((folder / name).read_bytes()) == expected for name, expected in result["commitments"].items()), "Result artifact differs")
    cal, held, points = (read(folder / file) for file in ("calibration.json", "held.json", "points.json"))
    require(result["points"] == points == fixed_points(cal), "Ordinary-only operating points changed")
    count = audit(args, profile, "calibration", cal)
    count += sum(audit(args, profile, phase, [r for r in held if r["phase"] == phase], points) for phase in ("retained_a", "retained_b"))
    require(summarize(held) == {key: result[key] for key in ("outcomes", "strata")}, "Outcome arithmetic changed")
    result["readback"] = {"input_and_mask_bindings": len(cal) + len(held), "native_overlaps": count}
    require(args.publication is not None, "A review destination is required")
    write(args.publication, result)
    print(result["readback"], flush=True)


if __name__ == "__main__":
    main()
