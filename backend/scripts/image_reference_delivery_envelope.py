# SPDX-License-Identifier: Apache-2.0
"""Test local differences outside a modeled delivery envelope for earlier JPEGs."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import (
    reference_compare,
    reference_crop_resize,
    reference_png_exact_resize,
    reference_resize,
)
from app.evidence import image_reference
from scripts import image_reference_png_jpeg_stability as prior

SCHEMA = "chipmunk.image-reference-delivery-envelope/1"
FLOORS = (2.5, 3.5, 4.5, 6.)
SCALES = (1.6, 2.8)
ALTERNATE_JPEG = (83,)
DESIGN = {"minimum_excess_rgb": FLOORS, "gaussian_sigma": SCALES, "additional_jpeg_quality": ALTERNATE_JPEG,
          "channel_margin_multiplier": 1.25, "background_noise_percentile": 70,
          "background_noise_multiplier": 2.75, "support_neighborhood": 7,
          "minimum_region_fraction": .0015, "minimum_region_pixels": 48, "authorship_weight": 0}
GATE = {"minimum_useful_feathered": 3, "minimum_useful_sharp": 3, "minimum_useful_public_tone": 4,
        "maximum_nuisance_highlights": 1, "maximum_incompatible_or_wrong_crops": 0,
        "maximum_mislocalized_generator_views": 2}
CONTROLS = prior.CONTROLS | {"matched_earlier"}
COMPONENTS = ("current", "raw", "fixed_ten", "envelope")
read, write, digest, pin, need = prior.read, prior.write, prior.digest, prior.pin, prior.need


def projected_views(reference, rectangle, size):
    x, y, width, height = (rectangle[key] for key in ("x", "y", "w", "h"))
    rectangle = x, y, x + width, y + height
    variants = (reference, *(reference_compare._prior_jpeg(reference, quality) for quality in ALTERNATE_JPEG))
    return [reference_png_exact_resize._project(variant, rectangle, size, kernel).astype(np.float32)
            for variant in variants for kernel in reference_png_exact_resize.KERNELS.values()]


def delivery_excess(views, submitted, sigma):
    """Measure color changes beyond the local range of plausible ordinary renderings."""
    actual = cv2.GaussianBlur(submitted.astype(np.float32), (0, 0), sigma)
    lo, hi = np.full_like(actual, np.inf), np.full_like(actual, -np.inf)
    for view in views:
        smoothed = cv2.GaussianBlur(view, (0, 0), sigma)
        for channel in range(3):
            plane = smoothed[:, :, channel]
            gain, offset = reference_compare._tone(plane, actual[:, :, channel])
            corrected = np.clip(plane * gain + offset, 0, 255)
            np.minimum(lo[:, :, channel], corrected, out=lo[:, :, channel])
            np.maximum(hi[:, :, channel], corrected, out=hi[:, :, channel])
    margin = DESIGN["channel_margin_multiplier"] * cv2.blur(hi - lo, (7, 7))
    residual = np.maximum(actual - hi, lo - actual) - margin
    return np.maximum(residual.max(axis=2), 0)


def envelope_mask(excess, floor):
    point = max(floor, DESIGN["background_noise_multiplier"] * float(np.percentile(excess, 70)))
    seeds = np.uint8(excess >= point)
    core = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contiguous = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return reference_compare._regions(contiguous, max(48, round(excess.size * .0015)))


def point_id(sigma, floor):
    return f"sigma_{sigma:g}_floor_{floor:g}"


def measured_metrics(mask, truth):
    return {**prior.prior.stats(mask, truth), "mask_sha256": digest(mask.tobytes())}


def cases(args, phase):
    if phase == "development":
        for row, before, after, reference, submitted, truth in prior.prior.observations(args.earlier, args.development):
            yield {**row, "cohort": row["generator"]}, before, reference, after, submitted, truth
    else:
        yield from prior.inputs(args.public_photos, args.development / "plan.json")


def freeze(args):
    existing_dev = prior.read(args.development / "plan.json")["cases"]
    held_flux, occupied = prior._licensed_flux({"cases": existing_dev})
    held_controls = prior._public_controls(args.public_photos, occupied)
    paths = [Path(__file__), *(Path(module.__file__) for module in (prior, prior.prior, reference_compare,
             reference_crop_resize, reference_png_exact_resize, reference_resize, image_reference))]
    return {"schema": SCHEMA, "design": DESIGN, "gate": GATE, "registration": "existing correlation crop comparison",
            "useful": {"minimum_iou": .30, "minimum_recall": .50, "requires_exact_crop": True},
            "selection": "Among points with no development wrong crop and at most one nuisance view, prefer most useful "
                         "feathered and then sharp edits, least nuisance, highest floor and least smoothing. "
                         "If no point is eligible, report development only.",
            "source_counts": {"development_edit_groups": len(existing_dev), "held_flux_edit_groups": len(held_flux),
                              "held_public_photo_groups": len(held_controls)},
            "code": {path.name: pin(path) for path in paths},
            "rows": {phase: [row for row, *_ in cases(args, phase)] for phase in ("development", "held")}}


def readback(scratch, record, reference, submitted, mask, name):
    prior.prior._native_mapping(record, submitted)
    with TemporaryDirectory(dir=scratch) as folder:
        out = Path(folder) / name
        image_reference._write_evidence(out, record, reference, submitted, mask)
        saved = read(out / "comparison.json")
        decoded, count = prior.prior.review.verify_assets(out, saved, reference, submitted)
        need(np.array_equal(decoded, mask), "Readback mask differs from the measured submitted pixels")
    return count


def compare_case(scratch, item, expected, selected):
    row, before, reference, after, submitted, truth = item
    need(row == expected, "Input or native pixel truth differs from the freeze")
    original, target = scratch / "original.jpg", scratch / "submitted.png"
    original.write_bytes(before)
    target.write_bytes(after)
    zero = np.zeros_like(truth, dtype=bool)
    empty = measured_metrics(zero, truth)
    with TemporaryDirectory(dir=scratch) as folder:
        out = Path(folder) / "baseline"
        baseline = image_reference.inspect(original, target, evidence_dir=out, allow_proportional_resize=True,
                                           allow_oriented_crop=True, allow_oriented_resize=True, allow_embedded_photo=True)
        current_mask, _ = prior.prior.review.verify_assets(out, baseline, reference, submitted)
    bounds = baseline.get("reference_crop_bounds")
    current_crop = {key: bounds[key] for key in ("x", "y", "w", "h")} if bounds else None
    configs = [(sigma, floor) for sigma in SCALES for floor in FLOORS] if selected is None else [tuple(selected)]
    responses = {"current": measured_metrics(current_mask, truth), "raw": empty, "fixed_ten": empty,
                 "envelope": {point_id(*cfg): empty for cfg in configs}}
    checked = {**row, "current_status": baseline["status"], "current_crop": current_crop,
               "measured_crop": None, "responses": responses, "verified_overlays": 0}
    if baseline["status"] == "comparable":
        return checked
    with Image.open(io.BytesIO(after), formats=["PNG"]) as carrier:
        proposal, raw_mask = reference_crop_resize.compare(reference, submitted, carrier)
    if not image_reference.accepted_crop_resize(proposal, submitted.shape[1::-1]):
        return checked
    crop = {key: proposal["reference_crop_bounds"][key] for key in ("x", "y", "w", "h")}
    responses["raw"] = measured_metrics(raw_mask, truth)
    _, fixed, _ = prior.refine(reference, submitted, raw_mask, crop)
    responses["fixed_ten"] = measured_metrics(fixed, truth)
    checked["measured_crop"] = crop
    views = projected_views(reference, crop, submitted.shape[1::-1])
    for sigma in sorted({cfg[0] for cfg in configs}):
        excess = delivery_excess(views, submitted, sigma)
        for floor in (cfg[1] for cfg in configs if cfg[0] == sigma):
            regions, mask, omitted = envelope_mask(excess, floor)
            responses["envelope"][point_id(sigma, floor)] = measured_metrics(mask, truth)
            if selected is not None:
                for region in regions:
                    mapped = reference_resize.native_crop(region, (crop["w"], crop["h"]), submitted.shape[1::-1])
                    region["reference_region"] = {**mapped, "x": mapped["x"] + crop["x"], "y": mapped["y"] + crop["y"]}
                measured = {**proposal, "regions": regions, "additional_regions": omitted}
                checked["verified_overlays"] = readback(scratch, measured, reference, submitted, mask, "candidate")
    return checked


def observations(args, phase, lock, selected=None):
    rows = []
    with TemporaryDirectory(dir=args.run) as scratch:
        for item, expected in zip(cases(args, phase), lock["rows"][phase], strict=True):
            rows.append(compare_case(Path(scratch), item, expected, selected))
            if len(rows) % 12 == 0:
                print(f"{phase}: {len(rows)}/{len(lock['rows'][phase])} pairs checked", flush=True)
    return rows


def summary(rows, point):
    strata, counts = defaultdict(Counter), Counter()
    for row in rows:
        role, cohort = row["role"], row["cohort"]
        item = strata[f"{cohort}/{role}"]
        exact = row["measured_crop"] is not None and row["measured_crop"] == row["crop"]
        item.update(views=1, exact=exact, current_comparable=row["current_status"] == "comparable")
        counts["incompatible_or_wrong_crops"] += row["measured_crop"] is not None and not exact
        for name in COMPONENTS:
            source = row["responses"][name]
            response = source[point] if name == "envelope" else source
            correct = (row["current_crop"] == row["crop"] and row["crop"] is not None) if name == "current" else exact
            useful = correct and response["useful"]
            highlighted = bool(response["selected"])
            item.update({f"{name}_useful": useful, f"{name}_highlighted": highlighted,
                         f"{name}_selected_pixels": response["selected"], f"{name}_truth_overlap_pixels": response["overlap"],
                         f"{name}_truth_pixels": response["target"]})
            if name != "envelope":
                continue
            counts["nuisance_highlights"] += role in CONTROLS and highlighted
            counts["mislocalized_generator_views"] += role in prior.AI_EDITS and highlighted and not useful
            counts[f"useful_{role}"] += useful
            counts["useful_public_tone"] += role == "smooth_tone" and cohort == "ordinary" and useful
    checks = {name: (counts[name.split("_", 1)[1]] >= bound if name.startswith("minimum")
                     else counts[name.split("_", 1)[1]] <= bound) for name, bound in GATE.items()}
    return {"views": len(rows), "native_source_crops_verified": 2 * sum(row["verified_overlays"] for row in rows),
            "native_overlays_verified": sum(row["verified_overlays"] for row in rows),
            "strata": {name: dict(value) for name, value in strata.items()}, "counts": dict(counts),
            "conditions": checks, "passed": all(checks.values())}


def select(rows):
    points = {(sigma, floor): summary(rows, point_id(sigma, floor))["counts"] for sigma in SCALES for floor in FLOORS}
    valid = {key: value for key, value in points.items() if not value["incompatible_or_wrong_crops"]
             and value["nuisance_highlights"] <= GATE["maximum_nuisance_highlights"]}
    selected = max(valid, key=lambda p: (valid[p]["useful_feathered"], valid[p]["useful_sharp"],
                   -valid[p]["nuisance_highlights"], p[1], -p[0]), default=None)
    return {"selected": selected, "point_counts": {point_id(*key): value for key, value in points.items()}}


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "develop", "held", "review"))
    for name in ("earlier", "development", "public_photos", "run"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(lock, freeze(args))
        print(pin(lock), flush=True)
        return
    frozen = read(lock)
    need(normalized(freeze(args)) == frozen, "Frozen code or source inputs differ")
    selection = args.run / "development.json"
    if args.stage in {"develop", "review"}:
        rows = observations(args, "development", frozen)
        development = {"freeze_sha256": pin(lock), **select(rows), "rows": rows}
        if args.stage == "develop":
            write(selection, development)
            print(json.dumps(select(rows)), flush=True)
            return
        need(normalized(development) == read(selection), "Development replay differs")
    dev = read(selection)
    need(dev["freeze_sha256"] == pin(lock) and normalized(select(dev["rows"])) == {
        key: dev[key] for key in ("selected", "point_counts")}, "Development receipt differs")
    selected = dev["selected"]
    need(selected is not None, "No development point qualified for held scoring")
    rows = observations(args, "held", frozen, selected)
    outcome = {"freeze_sha256": pin(lock), "development_sha256": pin(selection), "selected": selected,
               "summary": summary(rows, point_id(*selected)), "rows": rows}
    result = args.run / "held.json"
    if args.stage == "review":
        need(outcome == read(result), "Held native replay differs")
        write(args.run / "review.json", {"verified": True, "held_sha256": pin(result), "summary": outcome["summary"]})
    else:
        write(result, outcome)
    print(json.dumps({key: value for key, value in outcome["summary"].items() if key != "strata"}), flush=True)


if __name__ == "__main__":
    main()
