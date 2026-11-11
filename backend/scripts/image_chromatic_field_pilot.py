"""Replay a camera-lens chromatic field on pinned native camera captures."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from scripts import image_rawnind_phase_v239 as prior

SCHEMA = "chipmunk.image-chromatic-field-native-pilot/1"
PROFILE = {
    "grid": [0.15, 0.29, 0.43, 0.57, 0.71, 0.85],
    "tile_short_side_fraction": .125,
    "neutral_range": 32,
    "intensity_interval": [12, 242],
    "minimum_neutral_support_fraction": .2,
    "minimum_masked_gradient_mean": 2.,
    "bandpass_sigmas_px": [0.8, 2.4],
    "minimum_alignment_correlation": .65,
    "maximum_translation_px": 2.5,
    "maximum_ecc_iterations": 25,
    "ecc_epsilon": 1e-5,
    "minimum_fold_tiles": 6,
    "minimum_fold_quadrants": 3,
    "minimum_field_extent_px": .10,
    "minimum_null_gain_px": .015,
    "minimum_null_fractional_gain": .12,
    "maximum_median_error_to_extent": .70,
    "robust_scale_floor_px": .04,
    "minimum_native_captures_for_transfer": 3,
    "minimum_jpeg75_captures_for_transfer": 2,
}
CHANNELS = {"red": 0, "blue": 2}
VIEWS = ("native", "jpeg90", "jpeg75")
RAW_RECEIPT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/image-rawnind-phase-complete-v239.json"


def commitment(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_document() -> dict:
    return {
        "schema": SCHEMA,
        "profile": PROFILE,
        "implementation_sha256": commitment(Path(__file__)),
        "decoder_implementation_sha256": commitment(Path(prior.__file__)),
        "source_receipt_sha256": commitment(RAW_RECEIPT),
        "source_roster_sha256": prior.ROSTER_SHA,
        "rawpy_wheel_sha256": prior.WHEEL_SHA,
        "synthetic_seed": 611,
        "synthetic_target_tile": [3, 3],
        "synthetic_shift_px": [1.10, -.80],
        "views": list(VIEWS),
    }


def _window(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    color_range = tile.max(axis=2) - tile.min(axis=2)
    green = np.ascontiguousarray(tile[:, :, 1], dtype=np.float32)
    lo, hi = PROFILE["intensity_interval"]
    mask = (color_range <= PROFILE["neutral_range"]) & (green > lo) & (green < hi)
    mask[:6] = mask[-6:] = False
    mask[:, :6] = mask[:, -6:] = False
    if np.mean(mask) < PROFILE["minimum_neutral_support_fraction"]:
        return None
    dx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    dy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    if float(np.hypot(dx[mask], dy[mask]).mean()) < PROFILE["minimum_masked_gradient_mean"]:
        return None
    return green, mask.astype(np.uint8)


def _band(image: np.ndarray) -> np.ndarray:
    lo, hi = PROFILE["bandpass_sigmas_px"]
    image = np.ascontiguousarray(image, dtype=np.float32)
    return cv2.GaussianBlur(image, (0, 0), lo) - cv2.GaussianBlur(image, (0, 0), hi)


def _translation(reference: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> tuple | None:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                PROFILE["maximum_ecc_iterations"], PROFILE["ecc_epsilon"])
    try:
        corr, warp = cv2.findTransformECC(reference, _band(moving), warp,
                                          cv2.MOTION_TRANSLATION, criteria, mask, 3)
    except cv2.error:
        return None
    shift = warp[:, 2].astype(np.float64)
    if (not np.isfinite(shift).all() or not np.isfinite(corr)
            or corr < PROFILE["minimum_alignment_correlation"]
            or np.linalg.norm(shift) > PROFILE["maximum_translation_px"]):
        return None
    return tuple(map(float, shift)), float(corr)


def _positions(pixels: np.ndarray) -> tuple:
    h, w = pixels.shape[:2]
    grid = PROFILE["grid"]
    side = int(round(min(h, w) * PROFILE["tile_short_side_fraction"]))
    if side < 32:
        raise ValueError("The source is smaller than a usable native measurement tile")
    half = side // 2
    return (side, [(j, i, int(round(y * (h - 1))), int(round(x * (w - 1))))
                   for j, y in enumerate(grid) for i, x in enumerate(grid)], half)


def observations(pixels: np.ndarray) -> tuple[int, dict]:
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Expected native uint8 RGB")
    side, positions, half = _positions(pixels)
    results = {channel: [] for channel in CHANNELS}
    h, w = pixels.shape[:2]
    scale = min(h, w)
    for j, i, y, x in positions:
        tile = pixels[y-half:y-half+side, x-half:x-half+side]
        valid = _window(tile)
        if valid is None:
            continue
        green, mask = valid
        reference = _band(green)
        for channel, plane in CHANNELS.items():
            measured = _translation(reference, tile[:, :, plane], mask)
            if measured is None:
                continue
            shift, corr = measured
            results[channel].append({
                "tile": (j, i), "fold": (i + j) % 2, "quadrant": (j >= 3, i >= 3),
                "xy": ((x - (w - 1) / 2) / scale, (y - (h - 1) / 2) / scale),
                "shift": shift, "corr": corr,
            })
    return side, results


def _design(xy: np.ndarray) -> np.ndarray:
    design = np.zeros((len(xy), 2, 3))
    design[:, 0, 0] = design[:, 1, 1] = 1.
    design[:, :, 2] = xy
    return design


def _radial_fit(xy: np.ndarray, shift: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = _design(xy)
    weights = confidence.copy()
    for _ in range(4):
        root = np.sqrt(weights[:, None, None])
        estimate = np.linalg.lstsq((design * root).reshape(-1, 3),
                                   (shift * root[:, :, 0]).reshape(-1), rcond=None)[0]
        residual = np.linalg.norm(design @ estimate - shift, axis=1)
        scale = max(PROFILE["robust_scale_floor_px"], 1.5 * float(np.median(residual)))
        weights = confidence * np.minimum(1., scale / np.maximum(residual, 1e-12))
    null = np.median(shift, axis=0)
    return estimate, null


def summarize_channel(rows: list[dict]) -> tuple[dict, dict]:
    coverage = [sum(row["fold"] == fold for row in rows) for fold in (0, 1)]
    info = {"measured_tiles": len(rows), "fold_tiles": coverage, "status": "insufficient_tiles"}
    if min(coverage) < PROFILE["minimum_fold_tiles"]:
        return info, {}
    if min(len({row["quadrant"] for row in rows if row["fold"] == fold})
           for fold in (0, 1)) < PROFILE["minimum_fold_quadrants"]:
        return {**info, "status": "insufficient_spatial_coverage"}, {}
    xy = np.array([row["xy"] for row in rows])
    shift = np.array([row["shift"] for row in rows])
    weight = np.array([row["corr"] for row in rows])
    folds = np.array([row["fold"] for row in rows])
    errors = np.zeros(len(rows))
    null_errors = np.zeros(len(rows))
    extents = np.zeros(len(rows))
    slopes = []
    for fold in (0, 1):
        test = folds == fold
        model, null = _radial_fit(xy[~test], shift[~test], weight[~test])
        slopes.append(float(model[2]))
        errors[test] = np.linalg.norm(_design(xy[test]) @ model - shift[test], axis=1)
        null_errors[test] = np.linalg.norm(null - shift[test], axis=1)
        extents[test] = np.linalg.norm(xy[test] * model[2], axis=1)
    error = float(np.median(errors))
    null_error = float(np.median(null_errors))
    extent = float(np.quantile(extents, .90))
    gain = null_error - error
    supported = (slopes[0] * slopes[1] > 0 and extent >= PROFILE["minimum_field_extent_px"]
                 and gain >= PROFILE["minimum_null_gain_px"]
                 and gain >= PROFILE["minimum_null_fractional_gain"] * null_error
                 and error <= PROFILE["maximum_median_error_to_extent"] * extent)
    info.update(status="measurable" if supported else "field_not_supported",
                median_radial_error_px=error, median_constant_error_px=null_error, field_extent_px=extent,
                fold_slope_px=slopes)
    return info, {row["tile"]: float(e) for row, e in zip(rows, errors, strict=True)}


def analyze(pixels: np.ndarray) -> tuple[dict, dict]:
    tile, measured = observations(pixels)
    summaries, residuals = {}, {}
    for channel, rows in measured.items():
        summaries[channel], residuals[channel] = summarize_channel(rows)
    return {"tile_px": tile, "channels": summaries,
            "status": "measurable" if any(item["status"] == "measurable" for item in summaries.values())
            else "field_not_supported"}, residuals


def authored() -> dict:
    rng = np.random.default_rng(611)
    texture = rng.normal(128., 35., (576, 768)).astype(np.float32)
    texture = np.uint8(np.clip(np.rint(cv2.GaussianBlur(texture, (0, 0), .75)), 0, 255))
    h, w = texture.shape
    center = (w / 2, h / 2)

    def scale(k):
        warp = cv2.getRotationMatrix2D(center, 0., 1. + k)
        return cv2.warpAffine(texture, warp, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    baseline = np.stack([scale(.0024), texture, scale(-.0020)], axis=2)
    manipulated = baseline.copy()
    side, positions, half = _positions(baseline)
    _, _, y, x = next(row for row in positions if row[:2] == (3, 3))
    extent = side + 32
    start_y, start_x = y - extent // 2, x - extent // 2
    support = np.s_[start_y:start_y+extent, start_x:start_x+extent]
    dy, dx = np.indices((extent, extent))
    distance = np.minimum.reduce((dy, dx, extent - 1 - dy, extent - 1 - dx))
    alpha = np.clip(distance / 12, 0., 1.)
    original = baseline[:, :, 0]
    moved = cv2.warpAffine(original, np.float32([[1, 0, 1.10], [0, 1, -.80]]),
                           (w, h), borderMode=cv2.BORDER_REFLECT_101)
    manipulated[support[0], support[1], 0] = np.uint8(np.rint((1-alpha) * original[support] + alpha * moved[support]))
    before, before_residual = analyze(baseline)
    after, after_residual = analyze(manipulated)
    key = (3, 3)
    return {
        "baseline": before, "feathered_control": after,
        "target_before_px": before_residual["red"].get(key),
        "target_after_px": after_residual["red"].get(key),
        "target_rank": (1 + sum(value > after_residual["red"][key]
                                for value in after_residual["red"].values())) if key in after_residual["red"] else None,
        "exact_changed_pixels": int(np.count_nonzero(np.any(baseline != manipulated, axis=2))),
        "source_extent": [w, h], "target_tile_pixels": side, "mask_margin_px": (extent - side) // 2 + half - side // 2,
    }


def run(source: Path, freeze: Path) -> dict:
    started = time.monotonic()
    expected = freeze_document()
    if json.loads(freeze.read_text()) != expected:
        raise ValueError("Use the original pre-measurement method and source freeze")
    roster = prior.frozen_roster(source)
    published = json.loads(RAW_RECEIPT.read_bytes())
    if len(roster) != published["captures"] or published["roster_sha256"] != prior.ROSTER_SHA:
        raise ValueError("The exact previously verified native sources are required")
    wheel = prior.verified_wheel(source / prior.WHEEL)
    cv2.setNumThreads(1)
    control = authored()
    results = {name: [] for name in VIEWS}
    with tempfile.TemporaryDirectory(prefix="chromatic-pinned-rawpy-") as decoder:
        prior.unpack_decoder(wheel, decoder)
        sys.path.insert(0, decoder)
        try:
            if "rawpy" in sys.modules:
                raise ValueError("Run this in a fresh isolated decoder process")
            rawpy = importlib.import_module("rawpy")
            if not Path(rawpy.__file__).resolve().is_relative_to(Path(decoder).resolve()):
                raise ValueError("Pinned decoder was not loaded")
            for slot, entry in enumerate(roster):
                path = prior.raw_path(source, slot, entry)
                prior.verify_raw(prior.read_private(path, entry["bytes"]), entry)
                pixels = prior.decode(rawpy, path)
                for name, view in islice(prior.ordinary_views(pixels), 3):
                    measured, _ = analyze(view)
                    results[name].append(measured)
                print(f"Measured capture {slot+1}/{len(roster)}", file=sys.stderr, flush=True)
        finally:
            sys.path.pop(0)
    counts = {name: dict(Counter(row["status"] for row in rows)) for name, rows in results.items()}
    ready = (counts["native"].get("measurable", 0) >= PROFILE["minimum_native_captures_for_transfer"]
             and counts["jpeg75"].get("measurable", 0) >= PROFILE["minimum_jpeg75_captures_for_transfer"])
    return {"schema": SCHEMA, "freeze_sha256": commitment(freeze), "authored_channel_shift": control,
            "native_sources": len(roster), "new_neural_calls": 0, "views": results, "coverage_counts": counts,
            "advance_to_archived_positive_comparison": ready, "elapsed_s": time.monotonic() - started}


def write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze = modes.add_parser("freeze")
    freeze.add_argument("output", type=Path)
    replay = modes.add_parser("run")
    for name in ("source", "freeze", "output"):
        replay.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        write(args.output, freeze_document())
    else:
        write(args.output, run(args.source, args.freeze))


if __name__ == "__main__":
    main()
