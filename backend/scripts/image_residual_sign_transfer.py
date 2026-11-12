# SPDX-License-Identifier: Apache-2.0
"""Compare texture-conditioned residual sign triplets on fixed image inserts."""
from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from scripts import image_edge_spread_transfer as panel
from scripts import image_local_resampling_transfer as common
from scripts import image_residual_color_transfer as previous

SCHEMA = "chipmunk.image-residual-sign-transfer/1"
ARMS = ("fine_triplets", "wide_triplets", "two_scale")
BASELINE = previous.BASELINE
CELL = panel.CELL
STRIDES = (1, 2)
JPEG_PERIOD = 8
MIN_RESIDUAL = 1.0
ACTIVITY_SPLIT = 4.0
MIN_STRATUM = 32
POINT_FLOOR = .002
MIN_EXTRA_FEATHERED = 2
MAX_EXTRA_CONTROL_GROUPS = 3
MAX_EXTRA_CONTROL_VIEWS = 4


def symbol_counts(gray, stride, compressed):
    """Count four sign-transition patterns in each orientation and activity stratum."""
    rows, cols = (size // CELL for size in gray.shape)
    counts = np.zeros((rows, cols, 4, 4), dtype=np.float64)
    for direction in range(2):
        if direction:
            counts_view, values = counts.transpose(1, 0, 2, 3), gray.T
        else:
            counts_view, values = counts, gray
        width = values.shape[1]
        x = np.arange(stride, width - stride - 2)
        if compressed:
            phase = x % JPEG_PERIOD
            x = x[(phase >= stride) & (phase + 2 + stride < JPEG_PERIOD)]
        residual = cv2.filter2D(values, -1, np.array([[1., *([0.] * (stride - 1)), -2.,
                                                      *([0.] * (stride - 1)), 1.]], np.float32),
                                 borderType=cv2.BORDER_REFLECT_101)
        activity = cv2.blur(np.abs(residual), (9, 9))[:, x] > ACTIVITY_SPLIT
        first, second, third = (residual[:, x + offset] for offset in range(3))
        reliable = (np.abs(first) >= MIN_RESIDUAL) & (np.abs(second) >= MIN_RESIDUAL) & (np.abs(third) >= MIN_RESIDUAL)
        symbol = 2 * ((first < 0) != (second < 0)) + ((second < 0) != (third < 0))
        row_cells, col_cells = counts_view.shape[:2]
        cell = np.arange(values.shape[0])[:, None] // CELL * col_cells + x[None, :] // CELL
        code = 8 * cell + 4 * activity + symbol
        counts_view[:, :, direction * 2:direction * 2 + 2] = np.bincount(
            code[reliable], minlength=row_cells * col_cells * 8).reshape(row_cells, col_cells, 2, 4)
    return counts


def distance(a, b):
    """Return supported, stratum-weighted squared Hellinger distance."""
    na, nb = a.sum(axis=-1), b.sum(axis=-1)
    both = (na >= MIN_STRATUM) & (nb >= MIN_STRATUM)
    wa = (a + .5) / (na[..., None] + 2)
    wb = (b + .5) / (nb[..., None] + 2)
    difference = np.maximum(0., 1. - np.sqrt(wa * wb).sum(axis=-1))
    weights = np.where(both, np.minimum(na, nb), 0.)
    enough = both.reshape(-1, 2, 2).any(axis=-1).all(axis=-1)
    return np.where(enough, (weights * difference).sum(axis=-1) / np.maximum(1., weights.sum(axis=-1)), np.nan)


def contrast(counts, regions):
    cumulative = common.integral(counts)
    inside, *sides = [panel.sums(cumulative, region) for region in regions]
    inner = np.stack([distance(inside, side) for side in sides], axis=-1)
    outer = np.stack([distance(a, b) for a, b in combinations(sides, 2)], axis=-1)
    available = np.isfinite(inner).all(axis=-1) & np.isfinite(outer).all(axis=-1)
    foreground = np.sort(inner, axis=-1)[:, 1]
    background = np.median(outer, axis=-1)
    return np.where(available, np.maximum(foreground - background, 0.), -np.inf), foreground, background


def response(rgb, fmt):
    height, width = rgb.shape[:2]
    panel.require(rgb.dtype == np.uint8 and rgb.shape == (height, width, 3) and
                  min(height, width) >= 256 and height * width <= 1_048_576, "Require native RGB up to one megapixel")
    boxes = panel.candidate_boxes(height // CELL, width // CELL)
    if not len(boxes):
        return dict.fromkeys(ARMS)
    gray = cv2.cvtColor(rgb[:height // CELL * CELL, :width // CELL * CELL], cv2.COLOR_RGB2GRAY).astype(np.float32)
    regions = previous._regions(boxes)
    bands = [contrast(symbol_counts(gray, stride, fmt == "jpeg75"), regions) for stride in STRIDES]
    fine, wide = (band[0] for band in bands)
    reports = {}
    for arm, scores in zip(ARMS, (fine, wide, np.minimum(fine, wide)), strict=True):
        i = int(np.argmax(scores))
        if not np.isfinite(scores[i]):
            reports[arm] = None
            continue
        indices = {"fine_triplets": (0,), "wide_triplets": (1,), "two_scale": (0, 1)}[arm]
        reports[arm] = {"box": (boxes[i] * CELL).tolist(), "score": float(scores[i]),
                        "bands": [{"core_to_flank": float(bands[b][1][i]), "among_flanks": float(bands[b][2][i])}
                                  for b in indices]}
    return reports


def profile(development, earlier, later, boundary):
    base = previous.profile(development, earlier, later, boundary)
    return {**base, "schema": SCHEMA, "implementation_sha256": {
                **base["implementation_sha256"],
                **{str(path.relative_to(panel.ROOT)): panel.sha(path)
                   for path in (Path(__file__), Path(previous.__file__), Path(common.__file__))}},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version},
            "method": {"arms": list(ARMS), "formats": list(panel.FORMATS), "strides": list(STRIDES),
                       "native_cell": CELL, "candidate_sizes": [list(size) for size in panel.BOX_SIZES],
                       "jpeg_support": "Use triplets whose complete residual footprints remain within one eight-pixel block along the sample axis.",
                       "strata": "Orientation and local mean absolute residual above or below 4 gray levels.",
                       "support": {"minimum_absolute_residual": MIN_RESIDUAL, "minimum_stratum_count": MIN_STRATUM,
                                   "each_axis_required": True},
                       "region_score": "Third largest core-to-flank Hellinger distance minus median distance between flanks, floored at zero.",
                       "point_rule": "Strictly above second highest of eight development control-source maxima or 0.002, whichever is greater.",
                       "fit_roles": list(panel.DEV_CONTROLS), "held_roles": [*panel.POSITIVE, *panel.CONTROLS],
                       "baseline": "Existing PNG sharp-boundary check and aligned JPEG double-quantization region.",
                       "useful": "At least 0.30 native changed-pixel IoU and 0.50 recall.",
                       "advance": {"arm": "two_scale", "scope": "each held panel and delivery separately",
                                   "additional_feathered": MIN_EXTRA_FEATHERED,
                                   "additional_ordinary_views": MAX_EXTRA_CONTROL_VIEWS,
                                   "additional_ordinary_groups": MAX_EXTRA_CONTROL_GROUPS},
                       "confidence_contribution": 0}}


def fit(folder):
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    for _, _, images, _ in panel.scoring.panels(folder, plan, attempts):
        controls = panel.development_controls(images["original"], images["processed"])
        for fmt in panel.FORMATS:
            measured = [response(np.asarray(panel.paired.encoded(image, fmt)[1]), fmt) for image in controls.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max((row[arm]["score"] for row in measured if row[arm] is not None), default=0.))
    panel.require(all(len(values) == 8 for arms in maxima.values() for values in arms.values()), "Incomplete development sources")
    return {fmt: {arm: {"source_maxima": values, "point": max(POINT_FLOOR, sorted(values)[-2])}
                  for arm, values in arms.items()} for fmt, arms in maxima.items()}


def confirm(folder, journal, points):
    expected = {panel._key(row): row for row in panel.read(journal)}
    panel.require(len(expected) == 192, "Require exact previous encoded views")
    for case, images, known in panel.paired.inputs(folder):
        for role, image in panel.native_variants(images, case["box"]).items():
            truth = np.any(image != images["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            truth_sha = panel.paired.shared.digest(truth.tobytes())
            for fmt in panel.FORMATS:
                body, decoded = panel.paired.encoded(image, fmt)
                digest = panel.paired.shared.digest(body)
                saved = expected.pop((case["slot"], case["generator"], role, fmt))
                panel.require(digest == saved["input_sha256"] and saved.get("truth_sha256", truth_sha) == truth_sha
                              and int(truth.sum()) == next(iter(saved["measurements"].values()))["target"],
                              "Delivery or native truth changed")
                rgb = np.asarray(decoded)
                proposals = response(rgb, fmt)
                masks = {arm: panel.selection(value, points[fmt][arm]["point"], truth.shape) for arm, value in proposals.items()}
                ref = previous.format_reference(fmt, body, rgb)
                masks[BASELINE] = (panel.paired.boxed(ref["box"], truth.shape, True)
                                   if ref and ref["selected"] else np.zeros_like(truth))
                yield {"slot": case["slot"], "generator": case["generator"], "role": role, "format": fmt,
                       "input_sha256": digest, "truth_sha256": truth_sha, "proposals": {**proposals, BASELINE: ref},
                       "mask_sha256": {arm: panel.paired.shared.digest(mask.tobytes()) for arm, mask in masks.items()},
                       "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in masks.items()}}
    panel.require(not expected, "Missing frozen views")


def summarize(rows):
    panel.require(len(rows) == 192, "Incomplete confirmation rows")
    reports = {}
    for fmt in panel.FORMATS:
        cohort = [row for row in rows if row["format"] == fmt]
        arms = {}
        for arm in (*ARMS, BASELINE):
            positives = [row for row in cohort if row["role"] in panel.POSITIVE]
            controls = [row for row in cohort if row["role"] in panel.CONTROLS]
            hits = [row for row in positives if panel.useful(row["measurements"][arm])]
            flags = [row for row in controls if row["measurements"][arm]["reported"]]
            extra_hits = [row for row in hits if row["role"] == "feathered_edit"
                          and not panel.useful(row["measurements"][BASELINE])]
            extra_flags = [row for row in flags if not row["measurements"][BASELINE]["reported"]]
            arms[arm] = {"sharp": sum(row["role"] == "edited" for row in hits),
                         "feathered": sum(row["role"] == "feathered_edit" for row in hits),
                         "ordinary": len(flags), "original": sum(row["role"] == "original" for row in flags),
                         "ordinary_groups": len({row["slot"] for row in flags}), "extra_feathered": len(extra_hits),
                         "extra_ordinary": len(extra_flags), "extra_ordinary_groups": len({row["slot"] for row in extra_flags}),
                         "unassessed_positives": sum(row["proposals"][arm] is None for row in positives),
                         "unassessed_controls": sum(row["proposals"][arm] is None for row in controls)}
        reports[fmt] = arms
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("development", "earlier", "later", "boundary_rows", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen = profile(args.development, args.earlier, args.later, args.boundary_rows)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze_path, frozen)
        return
    panel.require(frozen == panel.read(freeze_path), "Frozen profile changed")
    cv2.setNumThreads(1)
    points_path = args.output / "points.json"
    if args.stage == "fit":
        panel.write(points_path, {"freeze_sha256": panel.sha(freeze_path), "points": fit(args.development)})
        return
    points = panel.read(points_path)
    panel.require(points["freeze_sha256"] == panel.sha(freeze_path), "Points not tied to frozen method")
    started = time.monotonic()
    results, journals = {}, {}
    for label, folder, prior in (("earlier", args.earlier, args.boundary_rows),
                                  ("later", args.later, args.later / "outcomes.json")):
        rows = list(confirm(folder, prior, points["points"]))
        journal = args.output / f"{label}-rows.json"
        panel.write(journal, rows)
        results[label], journals[label] = summarize(rows), panel.sha(journal)
        print(label, "verified", len(rows), "encoded views", flush=True)
    eligible = {fmt: all((metrics := result[fmt]["two_scale"])["extra_feathered"] >= MIN_EXTRA_FEATHERED
                         and metrics["extra_ordinary"] <= MAX_EXTRA_CONTROL_VIEWS
                         and metrics["extra_ordinary_groups"] <= MAX_EXTRA_CONTROL_GROUPS
                         for result in results.values()) for fmt in panel.FORMATS}
    panel.write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path),
                "points_sha256": panel.sha(points_path), "journal_sha256": journals,
                "elapsed_s": round(time.monotonic() - started, 3), "views": 384,
                "results": results, "two_scale_advances": eligible})


if __name__ == "__main__":
    main()
