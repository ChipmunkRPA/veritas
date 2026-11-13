# SPDX-License-Identifier: Apache-2.0
"""Measure compact native-grid phase agreement on exact masked image edits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow

from app.detectors.image import sharp_boundary
from scripts import image_matched_detail_transfer as common

SCHEMA = "chipmunk.image-phase-locked-residual/1"
ARMS = ("period_four", "period_eight")
CHANNELS = ("luma", "red_green", "blue_green")
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
PERIODS = (4, 8)
CELL = 16
GATE = {"minimum_primary_feathered_gains": 2, "minimum_alternate_feathered_gains": 1,
        "maximum_extra_photo_groups": 2, "maximum_extra_screenshot_groups": 1,
        "maximum_mislocalized_edits": 2}
SCOPE = {"cell_size": CELL, "periods": list(PERIODS), "channels": list(CHANNELS),
         "directions": list(map(list, DIRECTIONS)), "gaussian_sigma": .9, "texture_floor": .85,
         "minimum_luma_sd": 1.5, "maximum_luma_clipped_fraction": .15,
         "minimum_quadrant_coverage": .60, "minimum_ring_cells": 8,
         "minimum_in_phase_quadrant_ratio": .33, "ring_penalty": .65, "area_power": .20,
         "rectangle_lengths_cells": list(common.SIDES), "authorship_confidence_weight": 0,
         "calibration": "Strictly exceed the second-highest of eight ordinary source-level maxima.",
         "useful": {"minimum_native_iou": .30, "minimum_native_recall": .50}, "gate": GATE}
read, write, require, file_sha = common.read, common.write, common.require, common.file_sha


def maps(rgb):
    """Signed coefficients aligned to the common source grid, normalized for nearby texture."""
    raw = rgb.astype(np.float32)
    r, g, b = np.moveaxis(raw, -1, 0)
    luma = .299 * r + .587 * g + .114 * b
    y, x = np.mgrid[:CELL, :CELL]
    values = np.stack((luma, r - g, b - g), axis=-1)
    low = cv2.GaussianBlur(values, (0, 0), SCOPE["gaussian_sigma"], borderType=cv2.BORDER_REFLECT_101)
    residual = values - low
    mean_square = cv2.blur(residual ** 2, (9, 9), borderType=cv2.BORDER_REFLECT_101)
    scale = np.sqrt(np.maximum(mean_square, 0.))
    scaled = residual / np.maximum(scale, SCOPE["texture_floor"])
    tiles = np.stack([common.windows(ch) for ch in np.moveaxis(scaled, -1, 0)], axis=-1)
    luma_cells = common.windows(luma)
    valid = (luma_cells.std(axis=(-1, -2)) >= SCOPE["minimum_luma_sd"]) & (
        ((luma_cells < 3) | (luma_cells > 252)).mean(axis=(-1, -2)) < SCOPE["maximum_luma_clipped_fraction"])
    yyxx = np.stack([dy * y + dx * x for dy, dx in DIRECTIONS])
    terms = [np.exp((-2j * np.pi / period) * yyxx) for period in PERIODS]
    coefficients = np.einsum("hwyxc,pdyx->phwcd", tiles, np.stack(terms), optimize=True) / CELL ** 2
    coefficients = coefficients.reshape((len(PERIODS), *valid.shape, -1))
    require(np.isfinite(coefficients).all(), "Native phase coefficients must remain finite")
    return coefficients * valid[..., None], valid


def select(values, valid):
    height, width = valid.shape
    signal, count = common.integral(values), common.integral(valid.astype(int))
    best = {"score": None, "box": None, "channel": None, "direction": None,
            "amplitude": None, "ring_amplitude": None, "quadrant_in_phase": None}
    rect = common.rectangle
    for top, left, bottom, right, area in common.shapes(height, width):
        middle_y, middle_x = (top + bottom) // 2, (left + right) // 2
        inside = rect(signal, top, left, bottom, right)
        n = rect(count, top, left, bottom, right)
        ring = rect(signal, top - 1, left - 1, bottom + 1, right + 1) - inside
        ring_n = rect(count, top - 1, left - 1, bottom + 1, right + 1) - n
        core = inside / np.maximum(n[..., None], 1)
        amplitude = np.abs(core)
        ring_amplitude = np.abs(ring) / np.maximum(ring_n[..., None], 1)
        phase = np.conj(core) / np.maximum(amplitude, 1e-10)
        weakest = np.full_like(amplitude, np.inf)
        enough = ring_n >= SCOPE["minimum_ring_cells"]
        for a, b, c, d in ((top, left, middle_y, middle_x), (top, middle_x, middle_y, right),
                            (middle_y, left, bottom, middle_x), (middle_y, middle_x, bottom, right)):
            total = rect(count, a, b, c, d)
            enough &= total >= SCOPE["minimum_quadrant_coverage"] * (c - a) * (d - b)
            concordance = np.real(rect(signal, a, b, c, d) * phase) / np.maximum(total[..., None], 1)
            weakest = np.minimum(weakest, concordance)
        good = enough[..., None] & (weakest >= SCOPE["minimum_in_phase_quadrant_ratio"] * amplitude)
        value = np.maximum(0., weakest - SCOPE["ring_penalty"] * ring_amplitude) * (area / 9) ** SCOPE["area_power"]
        values = np.where(good, value, -np.inf)
        i, j, channel = np.unravel_index(values.argmax(), values.shape)
        score = float(values[i, j, channel])
        if np.isfinite(score) and (best["score"] is None or score > best["score"]):
            position = i, j, channel
            best = {"score": score, "box": [int(box[i, j] * CELL) for box in (left, top, right, bottom)],
                    "channel": CHANNELS[channel // len(DIRECTIONS)], "direction": list(DIRECTIONS[channel % len(DIRECTIONS)]),
                    "amplitude": float(amplitude[position]), "ring_amplitude": float(ring_amplitude[position]),
                    "quadrant_in_phase": float(weakest[position])}
    return best


def freeze(args):
    previous = common.freeze(args)
    return {**previous, "schema": SCHEMA, "design": SCOPE,
            "method_sha256": file_sha(Path(__file__)), "runtime": {"opencv": cv2.__version__,
            "numpy": np.__version__, "pillow": pillow}}


def observations(args, phase, lock, points=None):
    for (_, _, _, rgb, mask), source in common.joined_cases(args, phase, lock):
        native, valid = maps(rgb)
        responses = {}
        for arm, field in zip(ARMS, native, strict=True):
            result = select(field, valid)
            responses[arm] = common.response_summary(result, mask, points[arm] if points is not None else None)
        if points is not None:
            h, w = mask.shape
            sharp = sharp_boundary.measure(rgb) if sharp_boundary.supported_size(w, h) else None
            box = list(sharp.box) if sharp else None
            responses["sharp_boundary"] = {"score": sharp.score if sharp else None, "box": box,
                "pixels": common.pixels(box, mask), "flag": bool(sharp and sharp.score > sharp_boundary.REVIEW_POINT)}
        yield {**source, "target": int(mask.sum()), "shape": list(mask.shape), "responses": responses}


def points(rows):
    control = {arm: defaultdict(float) for arm in ARMS}
    for row in rows:
        for arm in ARMS:
            group = row["key"][1]
            control[arm][group] = max(control[arm][group], row["responses"][arm]["score"] or 0.)
    require(all(len(items) == 8 for items in control.values()), "Calibration source roster differs")
    return {arm: sorted(items.values())[-2] for arm, items in control.items()}


def aggregate(rows):
    counts = defaultdict(lambda: defaultdict(Counter))
    gates = {arm: Counter() for arm in ARMS}
    gains = {arm: defaultdict(set) for arm in ARMS}
    for row in rows:
        panel, group, role = row["key"]
        for arm, item in row["responses"].items():
            counts[f"{panel}/{role}"][arm].update(cases=1, assessed=item["score"] is not None, flagged=item["flag"],
                useful=bool(item["flag"] and common.useful(item["pixels"])),
                useful_before_point=common.useful(item["pixels"]))
        base = row["responses"]["sharp_boundary"]
        for arm in ARMS:
            tested = row["responses"][arm]
            if base["flag"] or not tested["flag"]:
                continue
            if not row["target"]:
                gains[arm]["extra_screenshot_groups" if panel == "screen" else "extra_photo_groups"].add((panel, group))
            elif not common.useful(tested["pixels"]):
                gates[arm]["mislocalized_edits"] += 1
            elif role in {"feathered_edit", "edit"}:
                gains[arm]["primary_feathered_gains" if panel == "held_lama" else "alternate_feathered_gains"].add(group)
    for arm in ARMS:
        gates[arm].update({key: len(groups) for key, groups in gains[arm].items()})
    measured = {arm: {key.split("_", 1)[1]: gates[arm][key.split("_", 1)[1]] for key in GATE} for arm in ARMS}
    passes = {arm: all(measured[arm][name] >= bound if kind == "minimum" else measured[arm][name] <= bound
                  for label, bound in GATE.items() for kind, name in (label.split("_", 1),)) for arm in ARMS}
    return {"counts": {key: dict(values) for key, values in counts.items()}, "gains": measured, "passes": passes}


def main():
    local, tile = common.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held", "review"))
    parser.add_argument("run", type=Path)
    defaults = {"tile-run": tile, "commons": local / "image-affine-native-scale-2026-11-11",
                "primary": local / "image-bg-real-acquisition-v225", "backup": local / "image-bg-real-backup-intake-v262",
                "legacy": local / "image-residual-patch-held-2026-11-11", "previous": local / "image-crop-layout-2026-11-12",
                "alternate": None}
    for key, value in defaults.items():
        parser.add_argument("--" + key, type=Path, default=value, required=value is None)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args)
    plan = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(plan, lock)
        print("Frozen", file_sha(plan), flush=True)
        return
    require(lock == read(plan), "Scope differs from the frozen inputs or method")
    calibration = args.run / "calibration.json"
    if args.stage == "calibrate":
        rows = list(observations(args, common.CALIBRATION_PANELS, lock))
        sealed = points(rows)
        write(calibration, {"freeze_sha256": file_sha(plan), "points": sealed, "rows": rows})
        print("Sealed points", sealed, flush=True)
        return
    cal = read(calibration)
    require(cal["freeze_sha256"] == file_sha(plan) and cal["points"] == points(cal["rows"]), "Calibration differs")
    if args.stage == "review":
        require(cal["rows"] == list(observations(args, common.CALIBRATION_PANELS, lock)), "Calibration replay differs")
    rows = list(observations(args, common.HELD_PANELS, lock, cal["points"]))
    recorded = {"schema": SCHEMA, "freeze_sha256": file_sha(plan), "calibration_sha256": file_sha(calibration),
                "points": cal["points"], "summary": aggregate(rows), "rows": rows}
    result = args.run / "held.json"
    if args.stage == "review":
        require(recorded == read(result), "Held replay differs")
    else:
        write(result, recorded)
    print(json.dumps({**recorded["summary"]["gains"], "passes": recorded["summary"]["passes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
