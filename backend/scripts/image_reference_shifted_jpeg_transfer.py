# SPDX-License-Identifier: Apache-2.0
"""Replay translated JPEG overlap while excluding pixels without an earlier counterpart."""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_alignment, reference_compare, reference_exact_jpeg
from scripts import image_reference_exact_jpeg_transfer as prior

SCHEMA = "chipmunk.image-reference-shifted-jpeg-transfer/1"
TILE = 16
MAX_OFFSET = 24
SHIFTS = ((7, -11), (-9, 5), (14, 13), (-16, -7))
DESIGN = {
    "carrier": "Identical dimensions, earlier upright RGB8 PNG and submitted baseline standard JPEG75 with 4:2:0 sampling.",
    "alignment": "Start from the existing corroborated integer offset. On unresolved alignment, the existing phase correlation nominates the only tested offset.",
    "phase_nomination": {"minimum_response": .30, "maximum_fractional_distance": .40},
    "proof": "Recreate the JPEG on its submitted grid and require the existing exact unique RGB block support on a certified interior. Independent black and white simulations must agree throughout that interior.",
    "max_offset": MAX_OFFSET, "guard_native_pixels": TILE,
    "difference": "Existing paired local difference and prior JPEG envelope run only on the certified submitted-grid interior.",
    "qualification": {"development": 20, "confirmation": 12, "prior_project_exposure": True},
    "gate": {"extra_feathered": 6, "assessed_ordinary_fills": 18, "max_added_ordinary_highlights": 2,
             "max_wrong_edit_locations": 1, "max_bad_transforms": 0, "max_unassessed_pixels_reported": 0},
    "confidence_weight": 0,
}
SHARP, SOFT = "sharp", "feathered"
FILLS = ("black_fill", "white_fill", "mirror_fill")
OTHER = ("fractional", "rotate", "unrelated", "nonstandard")
NUISANCE = (*FILLS, "recompress")
ROLES = (SHARP, SOFT, *NUISANCE, *OTHER)


def slice_region(bounds):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    return np.s_[y:y+h, x:x+w]


def offset_bounds(reference, submitted, incumbent):
    measured = incumbent.get("alignment")
    if measured:
        offset = measured["reference_minus_submitted"]
        return (offset["x"], offset["y"]), "corroborated"
    shift, response = reference_compare._alignment(reference, submitted)
    estimate = np.asarray(shift)
    rounded = np.rint(estimate)
    if response < .30 or np.max(np.abs(estimate-rounded)) > .40:
        return None, None
    return tuple(int(value) for value in -rounded), "phase_nominated"


def certified_interior(bounds):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    x0, y0 = (((start + 2*TILE - 1) // TILE) * TILE for start in (x, y))
    x1, y1 = (((stop - TILE) // TILE) * TILE for stop in (x+w, y+h))
    return {"x": x0, "y": y0, "w": x1-x0, "h": y1-y0}


def expected_delivery(reference, offset, overlap, carrier, fill):
    dx, dy = offset
    canvas = np.full_like(reference, fill)
    before = overlap | {"x": overlap["x"]+dx, "y": overlap["y"]+dy}
    canvas[slice_region(overlap)] = reference[slice_region(before)]
    return reference_exact_jpeg.encoded_reference(canvas, carrier)


def compare_pair(reference, carrier, earlier, submitted, incumbent):
    if incumbent["status"] == "comparable" or not reference_exact_jpeg.qualified_carriers(reference, carrier, earlier, submitted):
        return None, None
    native = np.array(reference)
    rgb = np.array(carrier)
    offset, method = offset_bounds(native, rgb, incumbent)
    if offset is None or not 1 <= max(map(abs, offset)) <= MAX_OFFSET:
        return None, None
    dx, dy = offset
    overlap = reference_alignment._overlap(native.shape[:2], rgb.shape[:2], dx, dy)
    if overlap is None:
        return None, None
    bounds = certified_interior(overlap)
    if min(bounds["w"], bounds["h"]) < 160:
        return None, None
    inner = slice_region(bounds)
    reconstructed = expected_delivery(native, offset, overlap, carrier, 0)[inner]
    white = expected_delivery(native, offset, overlap, carrier, 255)[inner]
    if not np.array_equal(reconstructed, white):
        return None, None
    proof = reference_exact_jpeg.alignment_proof(reconstructed, rgb[inner])
    if proof is None:
        return None, None

    def clean_residual(candidate):
        report, mask = reference_compare.compare(candidate, rgb[inner])
        return mask if report["status"] == "comparable" else None

    primary = clean_residual(reconstructed)
    if primary is None:
        return None, None
    supported = np.zeros(primary.shape, bool)
    if primary.any():
        alternates = [clean_residual(expected_delivery(reference_compare._prior_jpeg(native, quality), offset, overlap, carrier, 0)[inner])
                      for quality in reference_compare.PRIOR_JPEG_QUALITIES]
        masks = [mask for mask in alternates if mask is not None]
        if len(masks) < 2:
            return None, None
        _, supported, _ = reference_compare._regions(primary & (np.sum(masks, axis=0) >= 2), max(48, round(primary.size*.0015)))
    full_mask = np.zeros(rgb.shape[:2], bool)
    full_mask[inner] = supported
    return {"dx": dx, "dy": dy, "nomination": method, "submitted_interior": bounds,
            "earlier_interior": bounds | {"x": bounds["x"]+dx, "y": bounds["y"]+dy},
            "anchor_sha256": proof["anchor_sha256"], "exact_tiles": proof["exact_tiles"],
            "expected_interior_sha256": proof["expected_decoded_rgb_sha256"],
            "full_tiles": proof["full_tiles"], "anchor_count": proof["anchor_count"],
            "supported_zones": proof["supported_zones"]}, full_mask


def translate(rgb, dx, dy, *, border=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_NEAREST):
    matrix = np.array([[1., 0., dx], [0., 1., dy]], np.float32)
    return cv2.warpAffine(rgb, matrix, rgb.shape[1::-1], flags=interpolation,
                          borderMode=border, borderValue=(value, value, value))


def photo_groups(args, lock):
    args.later = args.later_f5
    heads = prior.panel.heads
    development = list(heads.bound_cases(args, lock["profile"], "held"))
    plan, known = prior.panel.fresh.planned(args)
    prior.require(plan == prior.read(args.output_fresh / "plan.json"), "Confirmation photo plan differs")
    confirmation = list(prior.panel.fresh.groups(args.output_fresh, plan, known))
    prior.require((len(development), len(confirmation)) == (20, 12), "Unexpected photo groups")
    yield "development", [(item["recipient"], item["family"], views, mask) for item, views, mask, _ in development]
    yield "confirmation", [(item["recipient"], item["generator"], views, mask) for item, views, mask in confirmation]


def deliveries(group, unrelated, slot):
    identity, _, native, truth = group
    base = prior.panel.heads.base
    original = native["original"]
    dx, dy = SHIFTS[slot % len(SHIFTS)]
    options = {"border": cv2.BORDER_REFLECT_101}
    target = (-dx, -dy)
    empty = np.zeros(original.shape[:2], bool)
    _, edits = base.authored(native, truth, identity)
    for role in (SHARP, SOFT):
        edited = edits[role]
        exact = np.any(edited != original, axis=-1)
        yield role, translate(edited, dx, dy, **options), translate(exact.astype(np.uint8), dx, dy).astype(bool), target, 75
    yield "black_fill", translate(original, dx, dy), empty, target, 75
    yield "white_fill", translate(original, dx, dy, value=255), empty, target, 75
    yield "mirror_fill", translate(original, dx, dy, **options), empty, target, 75
    yield "recompress", translate(reference_compare._prior_jpeg(original, 87), dx, dy, **options), empty, target, 75
    yield "fractional", translate(original, dx+.5, dy+.5, interpolation=cv2.INTER_LINEAR, **options), empty, None, 75
    height, width = original.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2., height/2.), 1.2, 1.)
    rotated = cv2.warpAffine(original, rotation, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    yield "rotate", translate(rotated, dx, dy, **options), empty, None, 75
    other = cv2.resize(unrelated, (width, height), interpolation=cv2.INTER_AREA)
    yield "unrelated", other, empty, None, 75
    yield "nonstandard", translate(original, dx, dy, **options), empty, target, 87


def measurement(mask, exact):
    if mask is None:
        mask = np.zeros(exact.shape, bool)
    target, selected, intersection = (int(view.sum()) for view in (exact, mask, exact & mask))
    return {"target": target, "selected": selected, "intersection": intersection, "mask_sha256": prior.sha(mask.tobytes()),
            "useful": target > 0 and 2*intersection >= target and 10*intersection >= 3*(target + selected - intersection)}


def evaluate(group, unrelated, slot):
    identity, family, views, _ = group
    earlier = prior.encode_png(views["original"])
    source = prior.panel.heads.base
    for role, rgb, exact, true_offset, quality in deliveries(group, unrelated, slot):
        body = source.encode(rgb, quality)
        with Image.open(io.BytesIO(earlier)) as native, Image.open(io.BytesIO(body)) as carrier:
            incumbent, old_mask = reference_alignment.compare(np.array(native), np.array(carrier), carrier)
            proof, new_mask = compare_pair(native, carrier, earlier, body, incumbent)
        if proof:
            inner = slice_region(proof["submitted_interior"])
            prior.require(np.count_nonzero(new_mask) == np.count_nonzero(new_mask[inner]), "Mask used an unassessed source location")
        offset = [proof["dx"], proof["dy"]] if proof else None
        yield {"source": identity, "family": family, "role": role, "earlier_sha256": prior.sha(earlier),
               "jpeg_sha256": prior.sha(body), "exact_sha256": prior.sha(exact.tobytes()),
               "true_offset": list(true_offset) if true_offset else None, "new_offset": offset, "proof": proof,
               "incumbent": measurement(old_mask, exact) | {"status": incumbent["status"]},
               "candidate": measurement(new_mask, exact)}


def summarize(rows):
    results = Counter()
    by_role = {role: Counter() for role in ROLES}
    for row in rows:
        role, proof, old, new = (row[key] for key in ("role", "proof", "incumbent", "candidate"))
        exact_offset = proof is not None and row["new_offset"] == row["true_offset"]
        positive = role in {SHARP, SOFT}
        by_role[role].update(files=1, incumbent_comparable=old["status"] == "comparable", incumbent_useful=old["useful"],
                             candidate_comparable=proof is not None, candidate_useful=new["useful"] and exact_offset,
                             correct_offsets=exact_offset, candidate_flags=new["selected"] > 0)
        results["extra_feathered"] += role == SOFT and new["useful"] and exact_offset and not old["useful"]
        results["assessed_ordinary_fills"] += role in FILLS and exact_offset
        results["max_added_ordinary_highlights"] += role in NUISANCE and new["selected"] > 0
        results["max_wrong_edit_locations"] += positive and new["selected"] > 0 and not (new["useful"] and exact_offset)
        results["max_bad_transforms"] += proof is not None and (role in OTHER or not exact_offset)
    results["max_unassessed_pixels_reported"] = 0
    passed = {key: (results[key] <= limit if key.startswith("max_") else results[key] >= limit)
              for key, limit in DESIGN["gate"].items()}
    return {"roles": by_role, "gate": {key: {"observed": results[key], "passed": ok} for key, ok in passed.items()},
            "pass": all(passed.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    lock, _, _ = prior.panel.fresh.previous(args)
    files = (Path(__file__), Path(reference_alignment.__file__), Path(reference_compare.__file__), Path(reference_exact_jpeg.__file__),
             args.prior_probe / "freeze.json", args.output_fresh / "plan.json", args.output_fresh / "intake.json")
    seal = {"schema": SCHEMA, "design": DESIGN, "sha256": {f"{file.parent.name}/{file.name}": prior.sha(file.read_bytes()) for file in files}}
    freeze = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(freeze, seal)
        return
    prior.require(prior.read(freeze) == seal, "Frozen comparison differs")
    outcome, start = {}, time.monotonic()
    for phase, groups in photo_groups(args, lock):
        rows = []
        for slot, group in enumerate(groups):
            other = groups[(slot+1) % len(groups)][2]["original"]
            rows.extend(evaluate(group, other, slot))
        private = args.output / f"{phase}.json"
        if args.stage == "run":
            prior.write(private, rows)
        else:
            prior.require(rows == prior.read(private), "Native comparison replay differs")
        outcome[phase] = summarize(rows)
        print(phase, outcome[phase]["gate"], flush=True)
    result = seal | {"outcome": outcome}
    if args.stage == "run":
        prior.write(args.output / "result.json", result)
    else:
        prior.require(result == prior.read(args.output / "result.json"), "Aggregate replay differs")
        prior.write(args.publication, result | {"verified_replays": 320, "replay_seconds": round(time.monotonic()-start, 2)})


if __name__ == "__main__":
    main()
