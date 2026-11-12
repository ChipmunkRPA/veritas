# SPDX-License-Identifier: Apache-2.0
"""Measure local interpolation periodicity on native image pixels."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from scripts import image_edge_spread_transfer as panel

SCHEMA = "chipmunk.image-local-resampling-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
CELL, RING, SEPARATION = 16, 3, 1
SIZES = ((6, 6), (6, 9), (9, 6), (8, 8), (8, 12), (12, 8), (10, 10), (12, 12))
SCALES = ((6, 5), (7, 5))
FREQUENCIES = np.array(sorted({*range(15, 23), *range(27, 36), *range(40, 47)}), dtype=np.float64) / 100
FREQUENCIES = np.sort(np.r_[FREQUENCIES, 1 / 6, 2 / 7, 1 / 3, 3 / 7])
ARMS = ("both_axes", "quadrant_agreement")
AUTHORED = ("resampled_linear", "resampled_cubic", "resampled_feathered")
EXTRA_CONTROLS = ("whole_linear", "whole_cubic", "native_paste")
CONTROLS = (*panel.CONTROLS, *EXTRA_CONTROLS)
DEVELOPMENT = (*panel.DEV_CONTROLS, *EXTRA_CONTROLS)
ROLE_KINDS = {role: kind for kind, roles in (("control", CONTROLS), ("resampling", AUTHORED), ("generated", panel.POSITIVE))
              for role in roles}
MIN_TEXTURE = .65
POINT_FLOOR = .015


def residual_fields(rgb):
    """Locally scale second differences and remove slow response changes."""
    rows, columns = (n // CELL * CELL for n in rgb.shape[:2])
    gray = cv2.cvtColor(rgb[:rows, :columns], cv2.COLOR_RGB2GRAY).astype(np.float32)
    fields = []
    energies = []
    for kernel in (np.array([[1., -2., 1.]], dtype=np.float32), np.array([[1.], [-2.], [1.]], dtype=np.float32)):
        amplitude = np.abs(cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT_101))
        energy = cv2.blur(amplitude, (9, 9), borderType=cv2.BORDER_REFLECT_101)
        bounded = np.minimum(4., amplitude / np.maximum(2., energy))
        fields.append(bounded - cv2.blur(bounded, (21, 21), borderType=cv2.BORDER_REFLECT_101))
        energies.append(energy)
    return fields, energies


def integral(cells):
    padding = ((1, 0), (1, 0), *((0, 0) for _ in cells.shape[2:]))
    return np.pad(cells.cumsum(axis=0).cumsum(axis=1), padding)


def coefficient_grids(rgb):
    """Project each native 16-pixel cell onto the same image-coordinate phases."""
    fields, energies = residual_fields(rgb)
    rows, cols = (n // CELL for n in fields[0].shape)
    phases = [np.exp(-2j * np.pi * np.arange(n * CELL)[:, None] * FREQUENCIES[None, :])
              .astype(np.complex64).reshape(n, CELL, -1) for n in (cols, rows)]
    grid = (rows, CELL, cols, CELL)
    across_columns = fields[0].reshape(grid).sum(axis=1)
    across_rows = fields[1].reshape(grid).sum(axis=3).transpose(0, 2, 1)
    terms = (np.einsum("rcp,cpf->rcf", across_columns, phases[0]), np.einsum("rcp,rpf->rcf", across_rows, phases[1]))
    texture = np.stack([x.reshape(grid).mean(axis=(1, 3)) for x in energies], axis=-1)
    return tuple(integral(term) for term in terms), integral(texture)


def candidate_boxes(shape):
    rows, columns = shape
    boxes = []
    for h, w in SIZES:
        left, top = np.meshgrid(np.arange(RING, columns - w - RING + 1), np.arange(RING, rows - h - RING + 1))
        boxes.append(np.stack((left, top, left + w, top + h), axis=-1).reshape(-1, 4))
    return np.concatenate(boxes)


def _areas(boxes):
    return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]


def response(rgb):
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 256
            or np.prod(rgb.shape[:2]) > 1_048_576):
        raise ValueError("Native 8-bit RGB pixels from 256 pixels up to one megapixel are required")
    boxes = candidate_boxes(tuple(n // CELL for n in rgb.shape[:2]))
    if not len(boxes):
        return dict.fromkeys(ARMS)
    coefficients, texture = coefficient_grids(rgb)
    inside = boxes + np.array([1, 1, -1, -1])
    excluded = boxes + np.array([-SEPARATION, -SEPARATION, SEPARATION, SEPARATION])
    outer = boxes + np.array([-RING, -RING, RING, RING])
    inside_area = _areas(inside)
    ring_area = _areas(outer) - _areas(excluded)
    inner_complex = tuple(panel.sums(field, inside) / (inside_area * CELL ** 2) for field in coefficients)
    ring_complex = tuple((panel.sums(field, outer) - panel.sums(field, excluded)) / (ring_area * CELL ** 2)
                         for field in coefficients)
    texture_inner = panel.sums(texture, inside) / inside_area
    texture_ring = (panel.sums(texture, outer) - panel.sums(texture, excluded)) / ring_area
    sufficient = np.minimum(texture_inner, texture_ring).min(axis=1) >= MIN_TEXTURE
    core_x, core_y = (np.abs(x) for x in inner_complex)
    # Test strips separately. Pooling them can cancel a whole-image response
    # when its exact frequency lies between the sampled frequencies.
    left, top, right, bottom = boxes.T
    strips = tuple(np.stack(b, axis=1) for b in (
        (left - RING, top, left - SEPARATION, bottom), (right + SEPARATION, top, right + RING, bottom),
        (left, top - RING, right, top - SEPARATION), (left, bottom + SEPARATION, right, bottom + RING)))
    guard = [np.maximum.reduce([np.abs(background), *(np.abs(panel.sums(field, strip)) / (_areas(strip) * CELL ** 2)
                                                      for strip in strips)]) for field, background in
             zip(coefficients, ring_complex, strict=True)]
    contrast = np.minimum(core_x - guard[0], core_y - guard[1])
    strength = np.minimum(1., np.sqrt(inside_area / 64))
    values = np.where(sufficient[:, None], np.maximum(0., contrast) * strength, -np.inf)
    # Project each quadrant onto the full candidate's phase. Two-axis
    # agreement in three quadrants distinguishes distributed periodicity
    # from an isolated line, edge or checkerboard feature.
    left, top, right, bottom = inside.T
    middle_x, middle_y = (left + right) // 2, (top + bottom) // 2
    quadrants = tuple(np.stack(b, axis=1) for b in ((left, top, middle_x, middle_y), (middle_x, top, right, middle_y),
                                                   (left, middle_y, middle_x, bottom), (middle_x, middle_y, right, bottom)))
    direction = tuple(np.conj(x) / np.maximum(1e-6, np.abs(x)) for x in inner_complex)
    contributions = []
    for box in quadrants:
        scale = _areas(box) * CELL ** 2
        signed = [np.real(panel.sums(field, box) / scale * phase) for field, phase in zip(coefficients, direction, strict=True)]
        contributions.append(np.minimum(*signed))
    third_quadrant = np.sort(contributions, axis=0)[1]
    corroborated = np.minimum(values, np.maximum(0., third_quadrant) * strength)
    result = {}
    for arm, grid in zip(ARMS, (values, corroborated), strict=True):
        at, frequency = np.unravel_index(grid.argmax(), grid.shape)
        result[arm] = ({"box": (boxes[at] * CELL).tolist(), "score": float(grid[at, frequency]),
                       "frequency_cycles_per_pixel": float(FREQUENCIES[frequency]),
                       "inside_amplitude_xy": [float(core_x[at, frequency]), float(core_y[at, frequency])],
                       "background_amplitude_xy": [float(x[at, frequency]) for x in guard]}
                       if np.isfinite(grid[at, frequency]) else None)
    return result


def _paste(pixels, box, crop):
    x0, y0, x1, y1 = box
    modified = pixels.copy()
    modified[y0:y1, x0:x1] = np.asarray(crop)
    return modified


def transformed_variants(pixels, box, scale):
    """Construct local exact-factor interventions and matched ordinary controls."""
    numerator, denominator = scale
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = box
    patch_w = (x1 - x0) // numerator * numerator
    patch_h = (y1 - y0) // numerator * numerator
    x0 += (x1 - x0 - patch_w) // 2
    y0 += (y1 - y0 - patch_h) // 2
    destination = (x0, y0, x0 + patch_w, y0 + patch_h)
    photo = Image.fromarray(pixels)
    small_w, small_h = patch_w // numerator * denominator, patch_h // numerator * denominator
    left, top = x0 + (patch_w - small_w) // 2, y0 + (patch_h - small_h) // 2
    donor = photo.crop((left, top, left + small_w, top + small_h))
    linear = _paste(pixels, destination, donor.resize((patch_w, patch_h), Image.Resampling.BILINEAR))
    cubic = _paste(pixels, destination, donor.resize((patch_w, patch_h), Image.Resampling.BICUBIC))
    offset = denominator / numerator
    mapping = (offset, 0, (1 - offset) * width / 2, 0, offset, (1 - offset) * height / 2)
    whole = [np.asarray(photo.transform(photo.size, Image.Transform.AFFINE, mapping, method))
             for method in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)]
    donor_x = width - patch_w if x0 < (width - patch_w) // 2 else 0
    donor_y = height - patch_h if y0 < (height - patch_h) // 2 else 0
    paste = _paste(pixels, destination, photo.crop((donor_x, donor_y, donor_x + patch_w, donor_y + patch_h)))
    feathered = panel.blend.local(pixels, linear, destination, feather=True)
    return dict(zip((*AUTHORED, *EXTRA_CONTROLS), (linear, cubic, feathered, *whole, paste), strict=True))


def profile(development, earlier, later, boundary):
    source_roles = dict(zip(("development", *panel.PANELS), (development, earlier, later), strict=True))
    plans = [panel.read(folder / "plan.json")["cases"] for folder in source_roles.values()]
    for field in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[field] for row in cases} for cases in plans]
        panel.require([len(g) for g in groups] == [8, 12, 12] and len(set.union(*groups)) == 32,
                      "Expected disjoint development, earlier and later source and creator groups")
    previous = {"earlier": panel.sha(boundary), "later": panel.sha(later / "outcomes.json")}
    panel.require(previous == panel.REFERENCE, "Require the exact two archived regional comparisons")
    files = (Path(__file__), Path(panel.__file__), Path(panel.paired.__file__), Path(panel.paired.panel.__file__),
             Path(panel.blend.__file__), Path(panel.scoring.__file__), Path(panel.paired.product.previous.__file__))
    return {
        "schema": SCHEMA, "source_group_counts": [8, 12, 12], "previous_confirmation_sha256": previous,
        "intake_sha256": {role: {name: panel.sha(folder / name) for name in ("plan.json", "attempts.jsonl", "result.json")}
                          for role, folder in source_roles.items()},
        "implementation_sha256": {str(path.relative_to(ROOT)): panel.sha(path) for path in files},
        "runtime": {"pillow": pillow_version, "numpy": np.__version__, "opencv": cv2.__version__},
        "method": {
            "arms": list(ARMS), "formats": list(panel.FORMATS), "cell_pixels": CELL, "flank_gap_cells": SEPARATION,
            "flank_outer_cells": RING, "sizes_hw_cells": [list(size) for size in SIZES], "min_texture": MIN_TEXTURE,
            "frequencies_cycles_per_pixel": FREQUENCIES.tolist(), "scales": [list(scale) for scale in SCALES],
            "scale_assignment": "Source slot modulo two, applied to both dimensions without anisotropic resize.",
            "controls": list(CONTROLS), "authored_resampling": list(AUTHORED), "unmodified_generated_edits": list(panel.POSITIVE),
            "development_roles": list(DEVELOPMENT), "minimum_point": POINT_FLOOR,
            "point_rule": "Strictly above the second-highest of eight source-level control maxima per delivery and arm.",
            "mask_rule": "The selected native rectangle. Useful at 0.30 IoU and 0.50 edited-pixel recall.",
            "geometry_pilot_rule": "In both panels, quadrant agreement must localize at least 18 of 36 authored edits including at least 4 of 12 feathered; at most 14 of 108 ordinary flags from at most 6 of 12 source groups.",
            "generated_edit_pilot_rule": "In both panels, quadrant agreement must localize at least two unmodified generated feathered edits at the same control bounds. This is separate from geometric evidence qualification.",
            "confidence_contribution": 0,
        },
    }


def _encoded(pixels, fmt):
    body, image = panel.paired.encoded(pixels, fmt)
    return panel.paired.shared.digest(body), np.asarray(image)


def fit(folder):
    _, plan, receipts = panel.scoring.checked_inputs(folder)
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    for slot, _, native, _ in panel.scoring.panels(folder, plan, receipts):
        pixels = native["original"]
        h, w = pixels.shape[:2]
        controls = panel.development_controls(pixels, native["processed"])
        extra = transformed_variants(pixels, (w // 3, h // 3, w * 7 // 12, h * 7 // 12), SCALES[slot % 2])
        controls.update({role: extra[role] for role in EXTRA_CONTROLS})
        for fmt in panel.FORMATS:
            observed = [response(_encoded(pixels, fmt)[1]) for pixels in controls.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max((row[arm]["score"] for row in observed if row[arm] is not None), default=0.))
    panel.require(all(len(values) == 8 for arms in maxima.values() for values in arms.values()), "Incomplete development groups")
    return {fmt: {arm: {"source_maxima": values, "point": max(POINT_FLOOR, sorted(values)[-2])}
                  for arm, values in arms.items()} for fmt, arms in maxima.items()}


def confirm(folder, sealed, points):
    prior = {panel._key(row): row for row in panel.read(sealed)}
    panel.require(len(prior) == 192, "Exact historic image roster required")
    for case, native, prior_truth in panel.paired.inputs(folder):
        original = native["original"]
        slot = case["slot"]
        scale = SCALES[slot % 2]
        views = panel.native_variants(native, case["box"]) | transformed_variants(original, case["box"], scale)
        panel.require(set(views) == set(ROLE_KINDS), "Incomplete matched image variants")
        for role, pixels in views.items():
            kind = ROLE_KINDS[role]
            truth = np.any(pixels != original, axis=2) if kind != "control" else np.zeros_like(prior_truth)
            truth_sha = panel.paired.shared.digest(truth.tobytes())
            for fmt in panel.FORMATS:
                digest, decoded = _encoded(pixels, fmt)
                if role in panel.POSITIVE or role in panel.CONTROLS:
                    saved = prior.pop((slot, case["generator"], role, fmt))
                    panel.require(digest == saved["input_sha256"] and int(truth.sum()) == next(iter(saved["measurements"].values()))["target"]
                                  and saved.get("truth_sha256", truth_sha) == truth_sha, "Exact archived image or mask differs")
                measured = response(decoded)
                masks = {arm: panel.selection(value, points[fmt][arm]["point"], truth.shape) for arm, value in measured.items()}
                yield {"slot": slot, "kind": kind, "role": role, "format": fmt, "scale": scale,
                       "input_sha256": digest, "truth_sha256": truth_sha, "proposals": measured,
                       "mask_sha256": {arm: panel.paired.shared.digest(mask.tobytes()) for arm, mask in masks.items()},
                       "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in masks.items()}}
    panel.require(not prior, "Not every archived comparison view was replayed")


def summarize(rows):
    panel.require(len(rows) == 336 and len({(r["slot"], r["role"], r["format"]) for r in rows}) == len(rows),
                  "Expected 336 exact distinct image views per panel")
    result = {}
    for fmt in panel.FORMATS:
        cohort = [row for row in rows if row["format"] == fmt]
        result[fmt] = {}
        for arm in ARMS:
            flags = [r for r in cohort if r["kind"] == "control" and r["measurements"][arm]["reported"]]
            hits = [r for r in cohort if r["kind"] != "control" and panel.useful(r["measurements"][arm])]
            by_role = Counter(r["role"] for r in flags + hits)
            resampled = [r for r in hits if r["kind"] == "resampling"]
            result[fmt][arm] = {
                "by_role": {role: by_role[role] for role in ROLE_KINDS}, "controls_flagged": len(flags),
                "control_groups_flagged": len({row["slot"] for row in flags}), "authored_resampling_useful": len(resampled),
                "useful_resampling_by_scale": {f"{n}/{d}": sum(tuple(row["scale"]) == (n, d) for row in resampled)
                                               for n, d in SCALES},
                "unmodified_generated_useful": sum(r["kind"] == "generated" for r in hits),
                "unassessed": sum(r["proposals"][arm] is None for r in cohort),
            }
    return result


def decision(results):
    advances = {}
    for fmt in panel.FORMATS:
        candidates = [panels[fmt]["quadrant_agreement"] for panels in results.values()]
        controls = all(row["controls_flagged"] <= 14 and row["control_groups_flagged"] <= 6 for row in candidates)
        advances[fmt] = {"authored_geometry": controls and all(row["authored_resampling_useful"] >= 18
                               and row["by_role"]["resampled_feathered"] >= 4 for row in candidates),
                         "unmodified_generated_edits": controls and all(row["by_role"]["feathered_edit"] >= 2 for row in candidates)}
    return advances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("development", "earlier", "later", "boundary_rows", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    output = args.output
    frozen = profile(args.development, args.earlier, args.later, args.boundary_rows)
    freeze_path = output / "freeze.json"
    if args.stage == "freeze":
        output.mkdir(mode=0o700)
        panel.write(freeze_path, frozen)
        return
    panel.require(panel.read(freeze_path) == frozen, "Methods, runtime or source panel changed after freeze")
    cv2.setNumThreads(1)
    points, seal = output / "points.json", output / "fit-receipt.json"
    if args.stage == "fit":
        panel.require(not points.exists() and not seal.exists(), "The development points were already sealed")
        panel.write(points, fit(args.development))
        panel.write(seal, {"freeze_sha256": panel.sha(freeze_path), "points_sha256": panel.sha(points)})
        print("Sealed local resampling points", flush=True)
        return
    panel.require(panel.read(seal) == {"freeze_sha256": panel.sha(freeze_path), "points_sha256": panel.sha(points)},
                  "Development receipt changed")
    names = ("earlier-rows.json", "later-rows.json", "aggregate.json")
    panel.require(not any((output / name).exists() for name in names), "The frozen panel has already been opened")
    result, journals = {}, {}
    started = time.monotonic()
    for label, folder, archived in (("earlier", args.earlier, args.boundary_rows),
                                    ("later", args.later, args.later / "outcomes.json")):
        rows = list(confirm(folder, archived, panel.read(points)))
        result[label] = summarize(rows)
        journal = output / f"{label}-rows.json"
        panel.write(journal, rows)
        journals[label] = panel.sha(journal)
        print(f"Scored {len(rows)} {label} resampling views", flush=True)
    panel.write(output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path),
                "points_sha256": panel.sha(points), "points": panel.read(points), "confirmation_journal_sha256": journals,
                "development_views": 144, "confirmation_views": 672, "prior_native_views_replayed": 384,
                "confirmation_source_groups": 24, "pilot_advances": decision(result), "confirmation": result,
                "elapsed_s": round(time.monotonic() - started, 3), "authorship_confidence_contribution": 0})


if __name__ == "__main__":
    main()
