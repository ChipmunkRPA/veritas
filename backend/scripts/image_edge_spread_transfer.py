# SPDX-License-Identifier: Apache-2.0
"""Evaluate native edge-spread differences on rights-reviewed image edits."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL import __version__ as pillow_version
from scipy import __version__ as scipy_version
from scipy.ndimage import map_coordinates

from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend

SCHEMA = "chipmunk.image-edge-spread-transfer/1"
FORMATS = ("png", "jpeg75")
ARMS = ("matched_ring", "opposed_flanks")
PANELS = ("earlier", "later")
POSITIVE = ("edited", "feathered_edit")
CONTROLS = ("original", "local_processing", "global_processing", "blur", "feathered_processing", "local_contrast")
DEV_CONTROLS = ("original", "global_processing", "local_processing", "blurred", "local_contrast", "sharpened")
REFERENCE = {"earlier": "dae5623b6eb947c98b3e97e5ab73a3735ee3cf392d4a902c038869d8259b146e",
             "later": "be4fffd689162bc37c4691fc46128ca495d716e265d282a8ed3f449ac84473e0"}
EARLIER_WAVELET_RECEIPT = "4547d30f9c3f4e999a997f064e4a54e2004c130f487843bdf3490e2d965afcdb"
ROOT = Path(__file__).resolve().parents[2]
CELL = 16
INNER_MARGIN, GAP, RING = 1, 1, 3
BOX_SIZES = ((6, 6), (6, 9), (9, 6), (8, 8), (8, 12), (12, 8), (10, 10), (12, 12))
EDGE_STEP = 4
RADIUS = 6
TANGENT = (-1.5, 0., 1.5)
CONTRAST_BINS = (40., 90.)
ORIENTATION_BINS = 4
STRATA = ORIENTATION_BINS * (len(CONTRAST_BINS) + 1)
MIN_EDGE_CONTRAST = 18.
MIN_WIDTH, MAX_WIDTH = .55, 7.
MIN_PER_STRATUM = 3
MIN_EDGES = 12
MIN_COMMON_STRATA = 2
POINT_FLOOR = 0.08
sha = blend.sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _crossing(profiles, level):
    indexes = (profiles >= level).argmax(axis=1)
    rows = np.arange(len(profiles))
    before = np.maximum(0, indexes - 1)
    lower, upper = profiles[rows, before], profiles[rows, indexes]
    return before + (level - lower) / np.maximum(upper - lower, 1e-6)


def measure_profiles(profiles):
    """Estimate 20–80% edge width only on coherent, single rising transitions."""
    low, high = profiles[:, :3], profiles[:, -3:]
    left, right = low.mean(axis=1), high.mean(axis=1)
    contrast = right - left
    scaled = (profiles - left[:, None]) / np.maximum(contrast[:, None], 1.)
    fall = np.maximum(0, -np.diff(scaled, axis=1)).sum(axis=1)
    endpoints = np.maximum(np.ptp(low, axis=1), np.ptp(high, axis=1))
    width = _crossing(scaled, .8) - _crossing(scaled, .2)
    supported = ((contrast >= MIN_EDGE_CONTRAST) & (fall <= .18) & (endpoints <= .22 * contrast)
                 & (scaled[:, RADIUS] >= .12) & (scaled[:, RADIUS] <= .88)
                 & (width >= MIN_WIDTH) & (width <= MAX_WIDTH))
    return width, contrast, supported


def sample_edges(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (5, 5), .8)
    edges = cv2.Canny(smooth.astype(np.uint8), 15, 40, L2gradient=True)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    y, x = np.nonzero(edges)
    height, width = (size // CELL * CELL for size in gray.shape)
    inside = (y >= 9) & (x >= 9) & (y < height - 9) & (x < width - 9)
    y, x = y[inside], x[inside]
    strength = np.hypot(gx[y, x], gy[y, x])
    nx, ny = gx[y, x] / np.maximum(1e-6, strength), gy[y, x] / np.maximum(1e-6, strength)
    step = np.arange(-RADIUS, RADIUS + 1, dtype=np.float32)
    samples = [map_coordinates(gray, (y[:, None] + ny[:, None] * step + nx[:, None] * tangent,
                                      x[:, None] + nx[:, None] * step - ny[:, None] * tangent), order=1)
               for tangent in TANGENT]
    widths, contrast, supported = measure_profiles(np.median(samples, axis=0))
    y, x, widths, contrast, strength = (item[supported] for item in (y, x, widths, contrast, strength))
    theta = np.arctan2(ny[supported], nx[supported])
    angle = np.round(theta * ORIENTATION_BINS / np.pi).astype(int) % ORIENTATION_BINS
    strata = angle + ORIENTATION_BINS * np.searchsorted(CONTRAST_BINS, contrast)
    seed = y // EDGE_STEP * width + x // EDGE_STEP
    order = np.lexsort((-strength, seed))
    _, index = np.unique(seed[order], return_index=True)
    return y[order[index]], x[order[index]], widths[order[index]], strata[order[index]]


def candidate_boxes(rows, cols):
    boxes = []
    for h, w in BOX_SIZES:
        left, top = np.meshgrid(np.arange(RING, cols - w - RING + 1), np.arange(RING, rows - h - RING + 1))
        boxes.append(np.stack((left, top, left + w, top + h), axis=-1).reshape(-1, 4))
    return np.concatenate(boxes)


def integral_stats(shape, edges):
    y, x, widths, strata = edges
    grid = np.zeros((*shape, STRATA, 2), dtype=np.float64)
    grid_y, grid_x = y // CELL, x // CELL
    np.add.at(grid, (grid_y, grid_x, strata, 0), 1.)
    np.add.at(grid, (grid_y, grid_x, strata, 1), np.log(widths))
    return np.pad(grid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0), (0, 0)))


def sums(integral, boxes):
    x0, y0, x1, y1 = boxes.T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def matched_gap(inside, outside):
    ni, no = inside[:, :, 0], outside[:, :, 0]
    both = (ni >= MIN_PER_STRATUM) & (no >= MIN_PER_STRATUM)
    weights = np.where(both, np.minimum(ni, no), 0.)
    total = weights.sum(axis=1)
    interior = inside[:, :, 1] / np.maximum(1., ni)
    exterior = outside[:, :, 1] / np.maximum(1., no)
    gap = ((interior - exterior) * weights).sum(axis=1) / np.maximum(1., total)
    supported = (total >= MIN_EDGES) & (both.sum(axis=1) >= MIN_COMMON_STRATA)
    return gap, supported, total, both.sum(axis=1)


def matched_proposals(integral, boxes):
    inner = sums(integral, boxes + np.array([INNER_MARGIN, INNER_MARGIN, -INNER_MARGIN, -INNER_MARGIN]))
    outer = sums(integral, boxes + np.array([-RING, -RING, RING, RING]))
    excluded = sums(integral, boxes + np.array([-GAP, -GAP, GAP, GAP]))
    gap, supported, edges, strata = matched_gap(inner, outer - excluded)
    left, top, right, bottom = boxes.T
    sides = ((left - RING, top, left - GAP, bottom), (right + GAP, top, right + RING, bottom),
             (left, top - RING, right, top - GAP), (left, bottom + GAP, right, bottom + RING))
    sign = np.where(gap >= 0, 1., -1.)
    flanks = []
    for flank in sides:
        difference, eligible, _, _ = matched_gap(inner, sums(integral, np.stack(flank, axis=-1)))
        flanks.append(np.where(eligible, difference * sign, -np.inf))
    agreement = np.maximum(np.minimum(flanks[0], flanks[1]), np.minimum(flanks[2], flanks[3]))
    ring_score = np.where(supported, np.abs(gap), -np.inf)
    flank_score = np.where(supported & np.isfinite(agreement), np.minimum(ring_score, np.maximum(0., agreement)), -np.inf)
    results = {}
    for arm, values in zip(ARMS, (ring_score, flank_score), strict=True):
        best = int(values.argmax()) if values.size else 0
        results[arm] = ({"box": (boxes[best] * CELL).tolist(), "score": float(values[best]),
                         "ratio_interior_over_ring": float(np.exp(gap[best])), "matched_edge_support": int(edges[best]),
                         "common_strata": int(strata[best])} if values.size and np.isfinite(values[best]) else None)
    return results


def response(pixels):
    if (pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8
            or min(pixels.shape[:2]) < 256 or np.prod(pixels.shape[:2]) > 1_048_576):
        raise ValueError("Native RGB geometry from 256 pixels up to one megapixel required")
    shape = tuple(size // CELL for size in pixels.shape[:2])
    edges = sample_edges(pixels)
    if len(edges[0]) < MIN_EDGES:
        return {arm: None for arm in ARMS}
    return matched_proposals(integral_stats(shape, edges), candidate_boxes(*shape))


def selection(proposal, point, shape):
    mask = np.zeros(shape, dtype=bool)
    if proposal is not None and proposal["score"] > point:
        x0, y0, x1, y1 = proposal["box"]
        mask[y0:y1, x0:x1] = True
    return mask


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def profile(development, earlier, later, boundary, wavelet):
    directories = dict(zip(("development", *PANELS), (development, earlier, later), strict=True))
    plans = [read(folder / "plan.json")["cases"] for folder in directories.values()]
    for field in ("flickr_photo_id", "flickr_owner"):
        groups = [{item[field] for item in plan} for plan in plans]
        require([len(group) for group in groups] == [8, 12, 12] and len(set.union(*groups)) == 32,
                "Expected disjoint development and confirmation source and creator groups")
    known = {"earlier": sha(boundary), "later": sha(later / "outcomes.json")}
    require(known == REFERENCE, "The exact published confirmation journals are required")
    previous = wavelet.parent / "result.json"
    wavelet_sha = sha(wavelet)
    require(sha(previous) == EARLIER_WAVELET_RECEIPT and read(previous)["journal_sha256"] == wavelet_sha,
            "The earlier wavelet comparison journal differs from the published receipt")
    code = (Path(__file__), Path(blend.__file__), Path(paired.__file__), Path(paired.panel.__file__),
            Path(scoring.__file__), Path(paired.product.previous.__file__))
    return {
        "schema": SCHEMA, "native_panel_sources": {label: {name: sha(folder / name) for name in
                 ("plan.json", "result.json", "attempts.jsonl")} for label, folder in directories.items()},
        "confirmation_journals_sha256": known, "earlier_wavelet_journal_sha256": wavelet_sha,
        "implementation_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "runtime": {"pillow": pillow_version, "numpy": np.__version__, "opencv": cv2.__version__, "scipy": scipy_version},
        "method": {
            "formats": list(FORMATS), "arms": list(ARMS), "native_cell_pixels": CELL, "normal_radius_pixels": RADIUS,
            "tangent_offsets_pixels": list(TANGENT), "candidate_height_width_cells": [list(size) for size in BOX_SIZES],
            "inside_margin_cells": INNER_MARGIN, "flank_gap_cells": GAP, "flank_outer_cells": RING,
            "width_quantiles": [0.20, 0.80], "contrast_bins": list(CONTRAST_BINS), "orientation_bins": ORIENTATION_BINS,
            "minimum_edge_contrast": MIN_EDGE_CONTRAST, "edge_sampling_cell_pixels": EDGE_STEP,
            "minimum_edges_per_stratum": MIN_PER_STRATUM, "minimum_common_strata": MIN_COMMON_STRATA,
            "minimum_matched_edges": MIN_EDGES, "minimum_point": POINT_FLOOR,
            "development_roles": list(DEV_CONTROLS), "confirmation_roles": [*POSITIVE, *CONTROLS],
            "point_rule": "Strictly above the second-highest of the eight development control-source maxima per format and arm.",
            "useful": {"minimum_iou": .30, "minimum_recall": .50},
            "pilot_rule": "Opposed flanks needs at least two additional useful feathered locations beyond wavelet detail in each panel, with at most 10 control flags on at most six source groups per panel. Compare separately by delivery.",
            "confidence_contribution": 0,
        },
    }


def development_controls(original, processed):
    height, width = original.shape[:2]
    box = (width // 3, height // 3, width * 7 // 12, height * 7 // 12)
    photo = Image.fromarray(original)
    changes = (processed, np.asarray(photo.filter(ImageFilter.GaussianBlur(1.25))),
               np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), np.asarray(ImageEnhance.Sharpness(photo).enhance(1.5)))
    return dict(zip(DEV_CONTROLS, (original, processed, *(blend.local(original, item, box, feather=True) for item in changes)),
                    strict=True))


def fit(folder):
    _, plan, attempts = scoring.checked_inputs(folder)
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in FORMATS}
    for _, _, assets, _ in scoring.panels(folder, plan, attempts):
        controls = development_controls(assets["original"], assets["processed"])
        for fmt in FORMATS:
            measured = [response(np.asarray(paired.encoded(pixels, fmt)[1])) for pixels in controls.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max((row[arm]["score"] for row in measured if row[arm] is not None), default=0.))
    require(all(len(values) == 8 for methods in maxima.values() for values in methods.values()), "Incomplete development roster")
    return {fmt: {arm: {"point": max(POINT_FLOOR, sorted(values)[-2]), "source_maxima": values}
                  for arm, values in methods.items()} for fmt, methods in maxima.items()}


def native_variants(base, box):
    photo = base["original"]
    contrast = np.asarray(ImageEnhance.Contrast(Image.fromarray(photo)).enhance(1.2))
    return {**base, "feathered_edit": blend.local(photo, base["edited"], box, feather=True),
            "feathered_processing": blend.local(photo, base["local_processing"], box, feather=True),
            "local_contrast": blend.local(photo, contrast, box)}


def _key(row):
    return row["slot"], row["generator"], row["role"], row["format"]


def confirm(folder, frozen, wavelet, points):
    reference = {_key(row): row for row in read(frozen)}
    previous = {_key(row): row for row in read(wavelet)}
    require(set(reference) == set(previous) and len(reference) == 192, "Incomplete comparison rosters")
    for case, source, known in paired.inputs(folder):
        for role, pixels in native_variants(source, case["box"]).items():
            truth = np.any(pixels != source["original"], axis=2) if role in POSITIVE else np.zeros_like(known)
            require(not np.any(truth & ~known), "The authored edit escapes its qualified patch")
            truth_sha = paired.shared.digest(truth.tobytes())
            for fmt in FORMATS:
                key = (case["slot"], case["generator"], role, fmt)
                expected, reference_row = reference.pop(key), previous.pop(key)
                payload, decoded = paired.encoded(pixels, fmt)
                digest = paired.shared.digest(payload)
                require(digest == expected["input_sha256"] == reference_row["input_sha256"], "Prior encoded JPEG or PNG differs")
                expected_truth = expected.get("truth_sha256")
                require(expected_truth is None or expected_truth == truth_sha, "The changed pixels differ from the frozen native mask")
                baseline = reference_row["measurements"]["luma_detail_deficit"]
                require(baseline["target"] == int(truth.sum()), "The frozen native mask target count differs")
                proposals = response(np.asarray(decoded))
                masks = {arm: selection(proposal, points[fmt][arm]["point"], truth.shape) for arm, proposal in proposals.items()}
                yield {"slot": case["slot"], "generator": case["generator"], "role": role, "format": fmt,
                       "input_sha256": digest, "truth_sha256": truth_sha, "candidate_regions": proposals,
                       "mask_sha256": {arm: paired.shared.digest(mask.tobytes()) for arm, mask in masks.items()},
                       "measurements": {**{arm: scoring.measure(mask, truth) for arm, mask in masks.items()}, "wavelet_detail": baseline}}
    require(not reference and not previous, "The exact confirmation roster was not consumed")


def useful(measurement):
    predicted, truth, overlap = (measurement[k] for k in ("reported", "target", "overlap"))
    return truth > 0 and 10 * overlap >= 3 * (predicted + truth - overlap) and 2 * overlap >= truth


def summarize(rows):
    keys = {(r["slot"], r["role"], r["format"]) for r in rows}
    groups = {i for i, _, _ in keys}
    require(len(groups) == 12 and len(keys) == len(rows)
            and keys == set(product(groups, (*POSITIVE, *CONTROLS), FORMATS)), "Incomplete confirmation roster")
    result = {}
    for fmt in FORMATS:
        views = [row for row in rows if row["format"] == fmt]
        result[fmt] = {}
        for arm in (*ARMS, "wavelet_detail"):
            flagged = [r for r in views if r["role"] in CONTROLS and r["measurements"][arm]["reported"]]
            hits = [r for r in views if r["role"] in POSITIVE and useful(r["measurements"][arm])]
            by_role = Counter(r["role"] for r in flagged + hits)
            result[fmt][arm] = {
                "by_role": {r: by_role[r] for r in (*POSITIVE, *CONTROLS)}, "controls_flagged": len(flagged),
                "control_groups_flagged": len({r["slot"] for r in flagged}), "useful_locations": len(hits),
                "groups_with_useful_locations": len({r["slot"] for r in hits}),
                "new_useful_feathered_vs_wavelet": sum(r["role"] == "feathered_edit" and not useful(r["measurements"]["wavelet_detail"])
                                                      for r in hits),
                "unassessed": sum(r["candidate_regions"][arm] is None for r in views) if arm in ARMS else 0,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("development", "earlier", "later", "boundary_journal", "earlier_wavelet_journal", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    output = args.output
    lock = profile(args.development, args.earlier, args.later, args.boundary_journal, args.earlier_wavelet_journal)
    if args.stage == "freeze":
        output.mkdir(mode=0o700)
        write(output / "freeze.json", lock)
        return
    require(read(output / "freeze.json") == lock, "The protocol, runtime or source intake changed after freeze")
    cv2.setNumThreads(1)
    point_file = output / "points.json"
    fit_receipt = output / "fit-receipt.json"
    if args.stage == "fit":
        require(not point_file.exists() and not fit_receipt.exists(), "Development points were already sealed")
        write(point_file, fit(args.development))
        write(fit_receipt, {"freeze_sha256": sha(output / "freeze.json"), "points_sha256": sha(point_file)})
        print("Sealed edge-spread review points", flush=True)
        return
    require(read(fit_receipt) == {"freeze_sha256": sha(output / "freeze.json"), "points_sha256": sha(point_file)},
            "The development point receipt changed")
    require(not any((output / name).exists() for name in ("earlier-rows.json", "later-rows.json", "aggregate.json")),
            "Use a fresh frozen run directory for confirmation")
    start = time.monotonic()
    points = read(point_file)
    panels, journals = {}, {}
    later = args.later / "outcomes.json"
    for name, folder, reference, wavelet in (("earlier", args.earlier, args.boundary_journal, args.earlier_wavelet_journal),
                                             ("later", args.later, later, later)):
        rows = list(confirm(folder, reference, wavelet, points))
        panels[name] = summarize(rows)
        target = output / f"{name}-rows.json"
        write(target, rows)
        journals[name] = sha(target)
        print(f"Verified and scored {len(rows)} {name} views", flush=True)
    advances = {}
    for fmt in FORMATS:
        candidates = (panel[fmt]["opposed_flanks"] for panel in panels.values())
        advances[fmt] = all(c["new_useful_feathered_vs_wavelet"] >= 2 and c["controls_flagged"] <= 10
                            and c["control_groups_flagged"] <= 6 for c in candidates)
    write(output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": sha(output / "freeze.json"),
          "points_sha256": sha(point_file), "points": points, "journal_sha256": journals,
          "development_control_views": 96, "confirmation_views": 384, "confirmation_source_groups": 24,
          "confirmation": panels, "pilot_advances": advances, "authorship_confidence_contribution": 0,
          "elapsed_s": round(time.monotonic() - start, 3)})


if __name__ == "__main__":
    main()
