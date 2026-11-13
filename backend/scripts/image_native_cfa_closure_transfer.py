# SPDX-License-Identifier: Apache-2.0
"""Test whether an intact bilinear interpolation stencil localizes native-grid edits."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import image_rawnind_phase_v239 as raw_source
from scripts.localization_benchmark import mask_metrics

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "docs/evaluation-panels/image-native-cfa-cc0-roster-2026-11-13.json"
SCHEMA = "chipmunk.image-native-cfa-closure-transfer/1"
CELL = 32
SCOPE = (1536, 2048)
POSITIVE = ("odd_copy", "classical_fill")
ORDINARY = ("original", "even_copy", "local_tone", "local_blur", "global_gain")
OUT_OF_SCOPE = ("resampled", "eight_bit", "adaptive_demosaic")
FOLD_CLOSURE = .58
PHASE_SEPARATION = .18
MIN_SAMPLES = 36
CLOSURE_TOLERANCE = 12
ENDPOINT_SEPARATION = 48
REGION_SIDES = (2, 3, 4, 5, 6)
ARMS = ("phase_loss", "phase_shift")
DESIGN = {
    "supported": "Three-channel uint16 RGB sensor-grid raster. Require two spatial folds to agree on strong alternating bilinear closure in red and blue. Evaluation uses fixed linear no-resize RAW development into a centered 2048 by 1536 crop with even origin. Test resampling, 8-bit conversion and alternate adaptive demosaicing as scope challenges.",
    "cell": CELL, "crop_height_width": list(SCOPE), "minimum_closure_fraction": FOLD_CLOSURE,
    "minimum_phase_margin": PHASE_SEPARATION, "minimum_samples_per_stencil": MIN_SAMPLES,
    "closure_tolerance_in_uint16_codes": CLOSURE_TOLERANCE,
    "minimum_two_neighbour_difference_in_uint16_codes": ENDPOINT_SEPARATION,
    "selection": "Scan single compact rectangles of 2, 3, 4, 5 or 6 complete cells per side and at least 65% usable coverage. Rank positive cell evidence by sum over square root of candidate area.",
    "cutoff": "Strictly above the second-highest development-camera ordinary maximum or 0.25, whichever is greater. Edits excluded from fit.",
    "positives": "Only conventional odd-pixel copy and classical Telea fill. No AI origin results.",
    "useful": "Changed-pixel IoU at least 0.30 and recall at least 0.50.",
    "confidence_weight": 0,
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def write(path, doc):
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")


def raw_render(decoder, path, adaptive=False):
    with decoder.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        raw_source.require(pattern is not None and pattern.shape == (2, 2)
                           and sorted(raw.color_desc[index] for index in pattern.flat) == sorted(b"RGGB"),
                           "A verified original 2x2 Bayer capture is required")
        dimensions = raw.sizes.height, raw.sizes.width
        algorithm = decoder.DemosaicAlgorithm.AHD if adaptive else decoder.DemosaicAlgorithm.LINEAR
        pixels = raw.postprocess(demosaic_algorithm=algorithm, use_camera_wb=False, use_auto_wb=False,
                                 user_wb=[1., 1., 1., 1.], output_color=decoder.ColorSpace.raw,
                                 output_bps=16, gamma=(1., 1.), no_auto_bright=True, user_flip=0,
                                 fbdd_noise_reduction=decoder.FBDDNoiseReductionMode.Off)
    raw_source.require(pixels.dtype == np.uint16 and pixels.shape[:2] == dimensions,
                       "Native RAW geometry or integer format changed")
    h, w = SCOPE
    y, x = (dimensions[0] - h) // 4 * 2, (dimensions[1] - w) // 4 * 2
    raw_source.require(y >= 0 and x >= 0, "The RAW is too small for the frozen crop")
    return np.ascontiguousarray(pixels[y:y+h, x:x+w]), (x, y)


def cell_sums(matrix):
    height, width = matrix.shape
    return matrix.reshape(height // CELL, CELL, width // CELL, CELL).sum(axis=(1, 3))


def stencil(rgb):
    values = rgb.astype(np.int32)
    height, width = rgb.shape[:2]
    channels = []
    for index in (0, 2):
        channel = values[..., index]
        orientations = []
        for axis in (1, 0):
            left = np.roll(channel, 1, axis=axis)
            right = np.roll(channel, -1, axis=axis)
            diff = np.abs(2 * channel - left - right)
            eligible = (np.abs(left - right) >= ENDPOINT_SEPARATION)
            eligible &= (np.minimum(left, right) > 1024) & (np.maximum(left, right) < 64511)
            eligible[[0, -1], :] = False
            eligible[:, [0, -1]] = False
            rates = []
            for parity in range(4):
                p = np.s_[parity // 2:height:2, parity % 2:width:2]
                evaluated = np.zeros_like(eligible)
                evaluated[p] = eligible[p]
                count = cell_sums(evaluated)
                matched = cell_sums(evaluated & (diff <= CLOSURE_TOLERANCE))
                rates.append(np.divide(matched, np.maximum(count, 1), where=count >= MIN_SAMPLES,
                                       out=np.full_like(count, np.nan, dtype=np.float64)))
            orientations.append(rates)
        channels.append(orientations)
    red, blue = channels
    return np.stack([np.mean([red[0][p ^ 1], red[1][p ^ 2], blue[0][p ^ 2], blue[1][p ^ 1]], axis=0)
                     for p in range(4)], axis=-1)


def learn(rates):
    cells = np.isfinite(rates).all(axis=-1)
    y, x = np.indices(cells.shape)
    folds = [rates[cells & ((x + y) % 2 == n)] for n in range(2)]
    if min(map(len, folds)) < 24:
        return None
    profiles = np.stack([np.median(fold, axis=0) for fold in folds])
    ranked = np.sort(profiles, axis=1)
    winners = np.argmax(profiles, axis=1)
    if winners[0] != winners[1] or ranked[:, -1].min() < FOLD_CLOSURE or (ranked[:, -1] - ranked[:, -2]).min() < PHASE_SEPARATION:
        return None
    return int(winners[0]), float(ranked[:, -1].min()), cells


def best_region(score, usable):
    score = np.where(usable, np.maximum(score, 0), 0)
    if not np.any(score):
        return {"score": 0., "box": None}
    rect = cv2.integral(score)
    support = cv2.integral(usable.astype(np.uint8))
    result = {"score": 0., "box": None}
    for h in REGION_SIDES:
        for w in REGION_SIDES:
            if max(h, w) > 2 * min(h, w):
                continue
            summed = rect[h:, w:] + rect[:-h, :-w] - rect[h:, :-w] - rect[:-h, w:]
            enough = support[h:, w:] + support[:-h, :-w] - support[h:, :-w] - support[:-h, w:]
            ranking = np.where(enough >= .65 * h * w, summed / np.sqrt(h * w), 0.)
            y, x = np.unravel_index(np.argmax(ranking), ranking.shape)
            if ranking[y, x] > result["score"]:
                result = {"score": float(ranking[y, x]), "box": [int(i) * CELL for i in (x, y, x+w, y+h)]}
    return result


def measure(rgb):
    if rgb.dtype != np.uint16 or rgb.shape != (*SCOPE, 3):
        return {"status": "unsupported_raster", "phase": None, "arms": {}}
    rates = stencil(rgb)
    reference = learn(rates)
    if reference is None:
        return {"status": "no_repeatable_stencil", "phase": None, "arms": {}}
    phase, expected, eligible = reference
    observed = rates[..., phase]
    strongest_other = np.max(rates[..., [p for p in range(4) if p != phase]], axis=-1)
    return {"status": "assessed", "phase": phase, "eligible_cells": int(eligible.sum()), "reference_closure": expected,
            "arms": {"phase_loss": best_region(expected - observed - .12, eligible),
                     "phase_shift": best_region(strongest_other - observed - .20, eligible & (strongest_other >= .55))}}


def edits(original, ordinal):
    height, width = original.shape[:2]
    y = (12 + (ordinal * 7) % 23) * CELL
    x = (15 + (ordinal * 9) % 30) * CELL
    target = np.s_[y:y+128, x:x+128]
    patch = original[target]
    for role in ("original", "even_copy", "odd_copy", "local_tone", "local_blur", "classical_fill",
                 "global_gain", "resampled", "eight_bit"):
        if role in {"original", "global_gain", "resampled", "eight_bit"}:
            if role == "original":
                modified = original
            elif role == "global_gain":
                modified = np.rint(original * .82 + 1200).astype(np.uint16)
            elif role == "resampled":
                shrunk = cv2.resize(original, (width * 3 // 4, height * 3 // 4), interpolation=cv2.INTER_AREA)
                modified = cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                modified = (original >> 8).astype(np.uint8)
            yield role, modified, np.zeros((height, width), dtype=bool)
            continue
        if role in {"even_copy", "odd_copy"}:
            shift = int(role == "odd_copy")
            replacement = original[5 * CELL:5 * CELL+128, 4 * CELL+shift:4 * CELL+128+shift]
        elif role == "local_tone":
            replacement = np.rint(patch * .88 + 2050).astype(np.uint16)
        elif role == "local_blur":
            replacement = cv2.GaussianBlur(patch, (5, 5), 1.1)
        else:
            region = original[y-20:y+148, x-20:x+148]
            mask = np.zeros(region.shape[:2], np.uint8)
            mask[20:148, 20:148] = 255
            replacement = np.stack([cv2.inpaint(channel.astype(np.float32), mask, 4, cv2.INPAINT_TELEA)
                                    for channel in region.transpose(2, 0, 1)], axis=-1)[20:148, 20:148]
            replacement = np.clip(np.rint(replacement), 0, 65535).astype(np.uint16)
        modified = original.copy()
        modified[target] = replacement
        truth = np.zeros((height, width), dtype=bool)
        truth[target] = np.any(replacement != patch, axis=-1)
        raw_source.require(truth.any(), "Local edit mask is empty")
        yield role, modified, truth


def source_cases(cached_raw, archive):
    sources = []
    for i, source in enumerate(raw_source.frozen_roster(cached_raw)):
        path = raw_source.raw_path(cached_raw, i, source)
        body = raw_source.read_private(path, source["bytes"])
        raw_source.verify_raw(body, source)
        sources.append({"path": path, "sha256": raw_source.sha256(body), "split": "development", "key": f"rawnind-{i}",
                        "license": "CC-BY-SA-4.0"})
    for row in read(ROSTER):
        path = archive / f"sample-{row['id']}{Path(row['url']).suffix.lower()}"
        raw_source.require(row["license"] == "CC0-1.0" and sha(path) == row["sha256"], "Unqualified raw.pixls.us source")
        sources.append({"path": path, "sha256": row["sha256"], "split": row["split"], "key": str(row["id"]),
                        "license": row["license"]})
    raw_source.require(len({row["sha256"] for row in sources}) == len(sources) == 14,
                       "Need 14 distinct publisher captures")
    return sources


def lock(cached_raw, archive):
    return {"schema": SCHEMA, "design": DESIGN, "runner_sha256": sha(Path(__file__)),
            "wheel_sha256": raw_source.WHEEL_SHA, "cc0_roster_sha256": sha(ROSTER),
            "source_cases": [{key: val for key, val in row.items() if key != "path"}
                             for row in source_cases(cached_raw, archive)],
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__}}


def readout(row, role, rgb, truth, origin):
    measured = measure(rgb)
    for arm in measured["arms"].values():
        selected = np.zeros_like(truth)
        if arm["box"]:
            x, y, stop_x, stop_y = arm["box"]
            selected[y:stop_y, x:stop_x] = True
        arm["metrics"] = mask_metrics(selected, truth)
    return {"case": row["key"], "role": role, "input_rgb_sha256": digest(rgb), "truth_sha256": digest(truth),
            "truth_pixels": int(truth.sum()), "native_crop_origin": list(origin), **measured}


def observe(cached_raw, archive, split):
    wheel = raw_source.verified_wheel(cached_raw / raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-native-cfa-") as directory:
        raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            raw_source.require("rawpy" not in sys.modules, "Fresh isolated decoder process required")
            decoder = importlib.import_module("rawpy")
            raw_source.require(Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()),
                               "The pinned isolated RAW decoder must be used")
            for ordinal, row in enumerate(source_cases(cached_raw, archive)):
                if row["split"] != split:
                    continue
                original, origin = raw_render(decoder, row["path"])
                for role, view, truth in edits(original, ordinal):
                    if split == "held" or role in ORDINARY:
                        yield readout(row, role, view, truth, origin)
                if split == "held":
                    adaptive, origin = raw_render(decoder, row["path"], adaptive=True)
                    yield readout(row, "adaptive_demosaic", adaptive, np.zeros(SCOPE, dtype=bool), origin)
                print(split, row["key"], flush=True)
        finally:
            sys.path.pop(0)


def thresholds(rows):
    maxima = {arm: {} for arm in ARMS}
    for row in rows:
        for arm, maxima_by_source in maxima.items():
            value = row["arms"].get(arm, {}).get("score", 0.)
            maxima_by_source[row["case"]] = max(value, maxima_by_source.get(row["case"], 0.))
    return {arm: max(.25, sorted(values.values(), reverse=True)[1]) for arm, values in maxima.items()}


def summary(rows, points):
    total = Counter()
    for row in rows:
        role = row["role"]
        total[role, "files"] += 1
        total[role, row["status"]] += 1
        for arm in ARMS:
            if arm not in row["arms"]:
                continue
            result = row["arms"][arm]
            metrics = result["metrics"]
            useful = metrics["iou"] >= .30 and (metrics["recall"] or 0.) >= .50
            total[role, arm + "_ungated_useful"] += useful
            total[role, arm + "_flags"] += result["score"] > points[arm]
            total[role, arm + "_useful"] += useful and result["score"] > points[arm]
    return {role: {key: val for (name, key), val in total.items() if role == name} for role in ORDINARY+POSITIVE+OUT_OF_SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "held", "review"))
    parser.add_argument("cached_raw", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    args.cached_raw = args.cached_raw.resolve()
    cv2.setNumThreads(1)
    frozen = lock(args.cached_raw, args.archive)
    freeze_path = args.output / "freeze.json"
    fit_path = args.output / "fit.json"
    held_path = args.output / "held.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, frozen)
        return
    raw_source.require(read(freeze_path) == frozen, "Frozen input commitment differs")
    split = "development" if args.phase == "fit" else "held"
    rows = list(observe(args.cached_raw, args.archive, split))
    if args.phase == "fit":
        write(fit_path, {"freeze_sha256": sha(freeze_path), "points": thresholds(rows), "rows": rows})
        return
    fit = read(fit_path)
    raw_source.require(fit["freeze_sha256"] == sha(freeze_path) and fit["points"] == thresholds(fit["rows"]),
                       "Frozen development points changed")
    result = {"fit_sha256": sha(fit_path), "rows": rows}
    if args.phase == "held":
        write(held_path, result)
    else:
        raw_source.require(read(held_path) == result, "Held geometry, score or hash replay differs")
        if args.receipt:
            write(args.receipt, {key: val for key, val in frozen.items() if key != "source_cases"}
                  | {"freeze_sha256": sha(freeze_path), "fit_sha256": sha(fit_path), "held_sha256": sha(held_path),
                     "points": fit["points"], "held_sources": 8, "development_sources": 6,
                     "exact_held_replays": len(rows), "held": summary(rows, fit["points"])})
    print(json.dumps(summary(rows, fit["points"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
