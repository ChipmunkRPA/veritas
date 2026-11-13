# SPDX-License-Identifier: Apache-2.0
"""Test compact interruptions of a corroborated row and column residual pattern."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from scripts import image_chroma_interpolation_transfer as chronologies
from scripts import image_edge_spread_transfer as panel
from scripts import image_residual_color_transfer as comparison

SCHEMA = "chipmunk.image-line-fingerprint-transfer/1"
FORMATS = ("png", "jpeg100", "jpeg90")
CONTROL = (*panel.CONTROLS, "local_copy", "global_resize")
CELL = 16
GAP = CELL
SIDE = 2 * CELL
TRIM = CELL // 2
MIN_REFERENCE = .16
MIN_ENERGY = .035
POINT_FLOOR = .08
sha, read, write, require = panel.sha, panel.read, panel.write, panel.require


def luma_residual(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    low = cv2.GaussianBlur(gray, (5, 5), .85)
    difference = gray - low
    activity = cv2.blur(np.abs(difference), (9, 9))
    return np.where((gray > 8) & (gray < 247), np.clip(difference / (activity + .8), -3., 3.), 0.)


def line_vectors(integral, low, high, first, length):
    indices = first[:, None] + np.arange(length)[None, :]
    means = (integral[indices, high[:, None]] - integral[indices, low[:, None]]) / (high - low)[:, None]
    # Remove energy that simply repeats at any phase of the final eight-pixel JPEG grid.
    phase = means.reshape(len(means), -1, 8)
    phase -= phase.mean(axis=1, keepdims=True)
    return means


def corr(first, second, min_energy=MIN_ENERGY):
    e0, e1 = np.square(first).mean(axis=1), np.square(second).mean(axis=1)
    numerator = (first * second).mean(axis=1)
    return np.where((e0 >= min_energy ** 2) & (e1 >= min_energy ** 2),
                    numerator / np.maximum(np.sqrt(e0 * e1), 1e-12), np.nan)


def axis_scores(integral, x, y, width, height):
    starts, length = y + TRIM, height - 2 * TRIM
    center = line_vectors(integral, x + TRIM, x + width - TRIM, starts, length)
    before = line_vectors(integral, x - GAP - SIDE, x - GAP, starts, length)
    after = line_vectors(integral, x + width + GAP, x + width + GAP + SIDE, starts, length)
    flank = corr(before, after)
    # Absence of center energy is a measured dropout only once both references agree.
    primary = np.maximum(np.nan_to_num(corr(center, before), nan=0.), np.nan_to_num(corr(center, after), nan=0.))
    return flank, primary


def best_region(rgb):
    height, width = rgb.shape[:2]
    if min(height, width) < 256 or height * width > 1_048_576:
        return None
    residual = luma_residual(rgb)
    horizontal, vertical = (np.pad(plane.cumsum(axis=1), ((0, 0), (1, 0))) for plane in (residual, residual.T))
    selected = None
    for shape_h, shape_w in chronologies.SIZES:
        w, h = shape_w * CELL, shape_h * CELL
        xx, yy = np.meshgrid(np.arange(GAP + SIDE, width - w - GAP - SIDE + 1, CELL),
                             np.arange(GAP + SIDE, height - h - GAP - SIDE + 1, CELL))
        if not xx.size:
            continue
        x, y = xx.ravel(), yy.ravel()
        flank_rows, core_rows = axis_scores(horizontal, x, y, w, h)
        flank_columns, core_columns = axis_scores(vertical, y, x, h, w)
        eligible = (flank_rows >= MIN_REFERENCE) & (flank_columns >= MIN_REFERENCE)
        gap = np.minimum(flank_rows - core_rows, flank_columns - core_columns)
        score = np.where(eligible, np.maximum(gap, 0.), -np.inf)
        index = int(score.argmax())
        if np.isfinite(score[index]) and (selected is None or score[index] > selected["score"]):
            selected = {"box": [int(x[index]), int(y[index]), int(x[index] + w), int(y[index] + h)],
                        "score": float(score[index]), "flank": [float(flank_rows[index]), float(flank_columns[index])],
                        "core": [float(core_rows[index]), float(core_columns[index])]}
    return selected


def geometric_controls(original, box):
    from PIL import Image

    x0, y0, x1, y1 = box
    copy = panel.blend.local(original, np.roll(original, (y1 - y0, x1 - x0), axis=(0, 1)), box, feather=True)
    photo = Image.fromarray(original)
    smaller = photo.resize(tuple(round(side * .8) for side in photo.size), Image.Resampling.LANCZOS)
    resized = np.asarray(smaller.resize(photo.size, Image.Resampling.LANCZOS))
    return {"local_copy": copy, "global_resize": resized}


def delivery(rgb, fmt):
    if fmt == "png":
        body, image = panel.paired.encoded(rgb, fmt)
        return body, np.asarray(image)
    body = chronologies.encode(rgb, int(fmt.removeprefix("jpeg")), 2)
    return body, chronologies.decode(body)[0]


def fit(folder):
    maxima = {fmt: [] for fmt in FORMATS}
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    for _, _, images, _ in panel.scoring.panels(folder, plan, attempts):
        original = images["original"]
        height, width = original.shape[:2]
        box = (width // 3, height // 3, width * 7 // 12, height * 7 // 12)
        controls = {**panel.development_controls(original, images["processed"]), **geometric_controls(original, box)}
        for fmt in FORMATS:
            proposals = [best_region(delivery(image, fmt)[1]) for image in controls.values()]
            maxima[fmt].append(max((proposal["score"] for proposal in proposals if proposal), default=0.))
    require(all(len(scores) == 8 for scores in maxima.values()), "Need eight development source groups")
    return {fmt: {"point": max(POINT_FLOOR, sorted(values)[-2]), "source_maxima": values} for fmt, values in maxima.items()}


def sources(folder):
    for row, images, _ in panel.paired.inputs(folder):
        original = images["original"]
        box = tuple(row["box"])
        views = {**panel.native_variants(images, box), **geometric_controls(original, box)}
        for role, image in views.items():
            truth = np.any(image != original, axis=-1) if role in panel.POSITIVE else np.zeros(original.shape[:2], bool)
            if role in panel.POSITIVE:
                x0, y0, x1, y1 = box
                require(truth.any() and truth[y0:y1, x0:x1].sum() == truth.sum(), "Exact authored truth escaped its region")
            yield row, role, image, truth


def confirm(folder, points):
    digest = panel.paired.shared.digest
    for case, role, image, truth in sources(folder):
        for fmt in FORMATS:
            body, decoded = delivery(image, fmt)
            proposal = best_region(decoded)
            baseline = comparison.format_reference(fmt, body, decoded)
            masks = {}
            for arm, item, selected in (("candidate", proposal, proposal and proposal["score"] > points[fmt]["point"]),
                                        ("baseline", baseline, baseline and baseline["selected"]),
                                        ("ranked", proposal, proposal)):
                masks[arm] = panel.paired.boxed(item["box"] if item else (), truth.shape, bool(selected))
            yield {"slot": case["slot"], "generator": case["generator"], "format": fmt, "role": role,
                   "input_sha256": digest(body), "truth_sha256": digest(truth.tobytes()), "proposal": proposal,
                   "baseline": baseline, "measurements": {key: panel.scoring.measure(mask, truth) for key, mask in masks.items()}}


def summarize(rows):
    output = {}
    for fmt in FORMATS:
        section = [row for row in rows if row["format"] == fmt]
        edited = [row for row in section if row["role"] in panel.POSITIVE]
        ordinary = [row for row in section if row["role"] in CONTROL]
        require(len(edited) == 24 and len(ordinary) == 96, "Held panel lost a source variant")
        hits = {name: {role: sum(panel.useful(row["measurements"][name]) for row in edited if row["role"] == role)
                       for role in panel.POSITIVE} for name in ("candidate", "baseline", "ranked")}
        flags = {name: {role: sum(bool(row["measurements"][name]["reported"]) for row in ordinary if row["role"] == role)
                        for role in CONTROL} for name in ("candidate", "baseline")}
        added = [row for row in ordinary if row["measurements"]["candidate"]["reported"] and not row["measurements"]["baseline"]["reported"]]
        extra = sum(row["role"] == "feathered_edit" and panel.useful(row["measurements"]["candidate"])
                    and not panel.useful(row["measurements"]["baseline"]) for row in edited)
        groups = len({row["slot"] for row in added})
        output[fmt] = {"useful": hits, "ordinary_flags": flags, "added_ordinary_views": len(added),
                       "added_ordinary_groups": groups, "added_feathered": extra,
                       "candidate_assessed": sum(row["proposal"] is not None for row in section),
                       "advances": extra >= 2 and len(added) <= 4 and groups <= 3}
    return output


def freeze(paths):
    input_records, ids = {}, {}
    for role, path in paths.items():
        input_records[role] = {name: sha(path / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
        ids[role] = [{row[key] for row in read(path / "plan.json")["cases"]}
                     for key in ("recipient", "flickr_owner", "flickr_photo_id")]
    require(all(not any(x & y for x, y in zip(ids[a], ids[b], strict=True))
                for a, b in (("development", "earlier"), ("development", "later"), ("earlier", "later"))),
            "Recorded source, photo or photographer overlaps across roles")
    code = (Path(__file__), Path(chronologies.__file__), Path(panel.__file__), Path(comparison.__file__),
            Path(panel.paired.__file__), Path(panel.blend.__file__), Path(comparison.jpeg_dq.__file__))
    return {"schema": SCHEMA, "source_records": input_records, "code": {path.name: sha(path) for path in code},
            "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version},
            "method": {"formats": list(FORMATS), "cell": CELL, "gap": GAP, "flank": SIDE, "trim": TRIM,
                       "shapes_in_cells": [list(size) for size in chronologies.SIZES], "minimum_flank_correlation": MIN_REFERENCE,
                       "minimum_line_rms": MIN_ENERGY, "point_floor": POINT_FLOOR, "held_controls": list(CONTROL),
                       "point_rule": "Strict exceedance of the second-highest of eight ordinary source maxima, with a fixed floor.",
                       "advancement": "At least two added useful feathered locations per held panel for each delivery, with no more than four added ordinary flags in at most three source groups.",
                       "useful_iou": .30, "useful_recall": .50, "confidence_weight": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for field in ("development", "earlier", "later", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    frozen = freeze({key: getattr(args, key) for key in ("development", "earlier", "later")})
    lock = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(lock, frozen)
        return
    require(read(lock) == frozen, "Frozen sources or code changed")
    cv2.setNumThreads(1)
    points_path = args.output / "points.json"
    if args.stage == "fit":
        write(points_path, {"freeze_sha256": sha(lock), "points": fit(args.development)})
        return
    points = read(points_path)
    require(points["freeze_sha256"] == sha(lock), "Calibration is unbound")
    started = time.monotonic()
    rows, summary = {}, {}
    for key in ("earlier", "later"):
        rows[key] = list(confirm(getattr(args, key), points["points"]))
        summary[key] = summarize(rows[key])
        print(key, summary[key], flush=True)
    write(args.output / "result.json", {"freeze_sha256": sha(lock), "points_sha256": sha(points_path),
                                        "rows": rows, "summary": summary, "elapsed_s": round(time.monotonic() - started, 3)})


if __name__ == "__main__":
    main()
