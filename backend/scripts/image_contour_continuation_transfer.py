# SPDX-License-Identifier: Apache-2.0
"""Test whether coherent exterior contours disappear inside small native image regions."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import image_boundary_prominence_holdout as paste
from scripts import image_edge_spread_transfer as panel
from scripts import image_residual_sign_transfer as earlier_method

SCHEMA = "chipmunk.image-contour-continuation-transfer/1"
BASELINE = earlier_method.BASELINE
ARMS = ("short_crossing", "two_distances")
CONVENTIONAL = ("hard_same_image_paste", "hard_other_photo_paste",
                "feathered_same_image_paste", "feathered_other_photo_paste")
CELL = panel.CELL
SPECS = ((1.15, 4, 4.), (2.1, 8, 2.5))
SLOPES = (-2, -1, 0, 1, 2)
MIN_SUPPORT = 3
POINT_FLOOR = .10


def plane_integral(gradient, near, min_gradient, inward):
    """Cumulative exterior-contour counts and continuation loss along each native cell plane."""
    height, width = gradient.shape
    planes = np.arange(width // CELL + 1) * CELL
    usable = (planes >= 3 * near) & (planes + 3 * near < width)
    lines = planes[usable]
    coordinates = np.arange(height)

    def extract(normal, tangent=0):
        return gradient[np.clip(coordinates + tangent, 0, height - 1)[:, None], lines + inward * normal]

    exterior = extract(-near)
    strength = np.abs(exterior)
    sign = np.sign(exterior)
    peak = (strength >= min_gradient) & (strength >= np.abs(extract(-near, 1)))
    peak &= strength > np.abs(extract(-near, -1))
    peak[: 6 * near] = peak[-6 * near :] = False
    supported = np.zeros_like(peak)
    continued = np.zeros_like(strength)
    for slope in SLOPES:
        shift = slope * near // 2
        paired = peak & (sign * extract(-3 * near, -shift) >= np.maximum(min_gradient, strength * .5))
        supported |= paired
        tolerance = near // 2
        shallow = np.maximum.reduce([sign * extract(near, shift + offset) for offset in (-tolerance, 0, tolerance)])
        deep = np.maximum.reduce([sign * extract(3 * near, 2 * shift + offset) for offset in (-tolerance, 0, tolerance)])
        continued = np.maximum(continued, np.where(paired, np.minimum(shallow, deep), 0.))
    loss = np.where(supported, np.clip(1 - continued / np.maximum(1., strength * .65), 0., 1.), 0.)
    result = np.zeros((height + 1, len(planes), 2), dtype=np.float64)
    result[1:, usable, 0] = np.cumsum(supported, axis=0)
    result[1:, usable, 1] = np.cumsum(loss, axis=0, dtype=np.float64)
    return result


def side_score(integral, start, end, face, inward):
    """Contrast each crossing with parallel crossings one and two cells farther outside."""
    count, loss = (integral[end, face] - integral[start, face]).T
    first, second = face - inward, face - 2 * inward
    outer_count, outer_loss = (integral[end, first] - integral[start, first]
                              + integral[end, second] - integral[start, second]).T
    excess = loss / np.maximum(1., count) - outer_loss / np.maximum(1., outer_count)
    return np.where((count >= MIN_SUPPORT) & (outer_count >= MIN_SUPPORT), excess, -np.inf), count


def crossings(gray, boxes, spec):
    sigma, near, min_gradient = spec
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    y_grad = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    x_grad = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    left, top, right, bottom = boxes.T
    inset = near
    vertical = top * CELL + inset, bottom * CELL - inset
    horizontal = left * CELL + inset, right * CELL - inset
    sides = [side_score(plane_integral(gradient, near, min_gradient, inward), start, end, face, inward)
             for gradient, (start, end), face, inward in (
                 (y_grad, vertical, left, 1), (y_grad, vertical, right, -1),
                 (x_grad.T, horizontal, top, 1), (x_grad.T, horizontal, bottom, -1),
             )]
    return np.column_stack([side[0] for side in sides]), np.column_stack([side[1] for side in sides])


def response(rgb):
    height, width = rgb.shape[:2]
    panel.require(rgb.dtype == np.uint8 and rgb.shape == (height, width, 3) and
                  min(height, width) >= 256 and height * width <= 1_048_576, "Require native RGB up to one megapixel")
    boxes = panel.candidate_boxes(height // CELL, width // CELL)
    if not len(boxes):
        return dict.fromkeys(ARMS)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    short, short_count = crossings(gray, boxes, SPECS[0])
    long, long_count = crossings(gray, boxes, SPECS[1])
    reports = {}
    for arm, sides in zip(ARMS, (short, np.minimum(short, long)), strict=True):
        scores = np.sort(sides, axis=1)[:, 1]
        index = int(np.argmax(scores))
        reports[arm] = None
        if np.isfinite(scores[index]):
            reports[arm] = {
                "box": (boxes[index] * CELL).tolist(), "score": max(0., float(scores[index])),
                "side_excess": [float(value) if np.isfinite(value) else None for value in sides[index]],
                "short_counts": short_count[index].astype(int).tolist(),
                "long_counts": long_count[index].astype(int).tolist(),
            }
    return reports


def conventional(original, other, box):
    hard = paste.conventional_views(original, other, box)
    soft = {key.replace("hard_", "feathered_"): panel.blend.local(original, image, box, feather=True)
            for key, image in hard.items()}
    return {**hard, **soft}


def profile(development, earlier, later, boundary):
    base = earlier_method.profile(development, earlier, later, boundary)
    return {**base, "schema": SCHEMA, "implementation_sha256": {
                **base["implementation_sha256"],
                **{str(path.relative_to(panel.ROOT)): panel.sha(path) for path in (Path(__file__), Path(paste.__file__))}},
            "method": {
                **{key: base["method"][key] for key in ("formats", "native_cell", "candidate_sizes", "baseline", "useful")},
                "arms": list(ARMS), "smoothing_sigma_near_px_gradient_floor": [list(spec) for spec in SPECS],
                "slopes": list(SLOPES), "outward_control_displacement_px": [CELL, 2 * CELL],
                "minimum_crossings": MIN_SUPPORT, "continuation_ratio": .65,
                "region_score": "Third strongest side above local exterior references. The second arm requires the same side at both distances.",
                "point_rule": "Strictly above the second-highest of eight development control source maxima or 0.10, whichever is larger.",
                "fit_roles": [*panel.DEV_CONTROLS, *CONVENTIONAL], "held_roles": [*panel.POSITIVE, *panel.CONTROLS, *CONVENTIONAL],
                "advance": base["method"]["advance"] | {"arm": "two_distances"},
                "confidence_contribution": 0,
            }}


def fit(folder):
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    sources = [images for _, _, images, _ in panel.scoring.panels(folder, plan, attempts)]
    panel.require(len(sources) == 8, "Incomplete development roster")
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    for index, images in enumerate(sources):
        original = images["original"]
        height, width = original.shape[:2]
        box = width // 3, height // 3, width * 7 // 12, height * 7 // 12
        views = panel.development_controls(original, images["processed"]) | conventional(
            original, sources[(index + 1) % len(sources)]["original"], box)
        for fmt in panel.FORMATS:
            measured = [response(np.asarray(panel.paired.encoded(image, fmt)[1])) for image in views.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max((row[arm]["score"] for row in measured if row[arm] is not None), default=0.))
    return {fmt: {arm: {"source_maxima": values, "point": max(POINT_FLOOR, sorted(values)[-2])}
                  for arm, values in arms.items()} for fmt, arms in maxima.items()}


def confirm(folder, prior, points):
    expected = {panel._key(row): row for row in panel.read(prior)}
    panel.require(len(expected) == 192, "Require exact previous encoded views")
    sources = list(panel.paired.inputs(folder))
    for index, (case, images, _) in enumerate(sources):
        original = images["original"]
        roles = panel.native_variants(images, case["box"]) | conventional(
            original, sources[(index + 1) % len(sources)][1]["original"], case["box"])
        for role, image in roles.items():
            truth = np.any(image != original, axis=2)
            positive = role in panel.POSITIVE
            scored_truth = truth if positive else np.zeros_like(truth)
            truth_sha = panel.paired.shared.digest(scored_truth.tobytes())
            for fmt in panel.FORMATS:
                body, decoded = panel.paired.encoded(image, fmt)
                digest = panel.paired.shared.digest(body)
                if role not in CONVENTIONAL:
                    saved = expected.pop((case["slot"], case["generator"], role, fmt))
                    panel.require(digest == saved["input_sha256"] and saved.get("truth_sha256", truth_sha) == truth_sha
                                  and int(scored_truth.sum()) == next(iter(saved["measurements"].values()))["target"],
                                  "Delivery or native truth changed")
                rgb = np.asarray(decoded)
                proposals = response(rgb)
                masks = {arm: panel.selection(value, points[fmt][arm]["point"], truth.shape) for arm, value in proposals.items()}
                baseline = earlier_method.previous.format_reference(fmt, body, rgb)
                masks[BASELINE] = (panel.paired.boxed(baseline["box"], truth.shape, True)
                                   if baseline and baseline["selected"] else np.zeros_like(truth))
                rank_geometry = {arm: panel.scoring.measure(panel.selection(value, -1., truth.shape), truth)
                                 for arm, value in proposals.items()} if positive else {}
                yield {"slot": case["slot"], "generator": case["generator"], "role": role, "format": fmt,
                       "input_sha256": digest, "truth_sha256": truth_sha, "actual_changes_sha256": panel.paired.shared.digest(truth.tobytes()),
                       "proposals": {**proposals, BASELINE: baseline},
                       "rank_geometry": rank_geometry,
                       "mask_sha256": {arm: panel.paired.shared.digest(mask.tobytes()) for arm, mask in masks.items()},
                       "measurements": {arm: panel.scoring.measure(mask, scored_truth) for arm, mask in masks.items()}}
    panel.require(not expected, "Missing frozen views")


def summary(rows):
    result = {}
    for fmt in panel.FORMATS:
        eligible = [row for row in rows if row["format"] == fmt]
        result[fmt] = {}
        for arm in (*ARMS, BASELINE):
            positives = [row for row in eligible if row["role"] in panel.POSITIVE]
            controls = [row for row in eligible if row["role"] not in panel.POSITIVE]
            hits = [row for row in positives if panel.useful(row["measurements"][arm])]
            flags = [row for row in controls if row["measurements"][arm]["reported"]]
            added = [row for row in flags if not row["measurements"][BASELINE]["reported"]]
            flagged_sources = {row["slot"] for row in flags}
            baseline_sources = {row["slot"] for row in controls if row["measurements"][BASELINE]["reported"]}
            result[fmt][arm] = {
                "sharp": sum(row["role"] == "edited" for row in hits),
                "feathered": sum(row["role"] == "feathered_edit" for row in hits),
                "generator_hits": {generator: sum(row["generator"] == generator for row in hits)
                                   for generator in ("sdxl", "kandinsky")},
                "ordinary": len(flags), "ordinary_groups": len(flagged_sources),
                "per_control": {role: sum(row["role"] == role for row in flags) for role in (*panel.CONTROLS, *CONVENTIONAL)},
                "extra_ordinary": len(added), "extra_ordinary_groups": len({row["slot"] for row in added}),
                "newly_flagged_groups": len(flagged_sources - baseline_sources),
                "extra_feathered": sum(row["role"] == "feathered_edit" and not panel.useful(row["measurements"][BASELINE])
                                       for row in hits),
                "unassessed_positives": sum(row["proposals"][arm] is None for row in positives),
                "unassessed_controls": sum(row["proposals"][arm] is None for row in controls),
                "useful_before_point": sum(panel.useful(row["rank_geometry"][arm]) for row in positives) if arm in ARMS else None,
            }
    return result


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
        panel.require(len(rows) == 288, "Incomplete confirmation rows")
        journal = args.output / f"{label}-rows.json"
        panel.write(journal, rows)
        results[label], journals[label] = summary(rows), panel.sha(journal)
        print(label, "verified", len(rows), "encoded views", flush=True)
    rules = frozen["method"]["advance"]
    advances = {fmt: all((measured := result[fmt]["two_distances"])["extra_feathered"] >= rules["additional_feathered"]
                         and measured["extra_ordinary"] <= rules["additional_ordinary_views"]
                         and measured["extra_ordinary_groups"] <= rules["additional_ordinary_groups"]
                         for result in results.values()) for fmt in panel.FORMATS}
    panel.write(args.output / "aggregate.json", {
        "schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path), "points_sha256": panel.sha(points_path),
        "journal_sha256": journals, "elapsed_s": round(time.monotonic() - started, 3), "views": 576,
        "results": results, "two_distances_advances": advances,
    })


if __name__ == "__main__":
    main()
