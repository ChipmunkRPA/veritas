# SPDX-License-Identifier: Apache-2.0
"""Test an earlier JPEG against cropped and resized PNG evidence."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, features
from PIL import __version__ as pillow_version

from app.detectors.image import (
    reference_alignment,
    reference_compare,
    reference_crop_resize,
    reference_resize,
)
from app.evidence import image_reference
from scripts import image_background_patch_score as panel
from scripts import image_reference_crop_resize_transfer as prior
from scripts import image_reference_jpeg90_transfer as jpeg
from scripts import review_image_reference_overlays as review

SCHEMA = "chipmunk.image-reference-png-jpeg-transfer/1"
POSITIVE = {"sharp", "feathered", "smooth_tone", "local_blur"}
NUISANCE = {"untouched_crop", "prior_jpeg", "contrast", "matched_earlier"}
INCOMPATIBLE = {"rotation", "fractional", "nonproportional", "different_photo"}
GATE = {"minimum_added_feathered": 6, "minimum_added_smooth_tone": 4, "minimum_exact_matched_earlier": 8,
        "maximum_wrong_or_incompatible_crops": 0, "maximum_extra_nuisance_views": 4,
        "maximum_extra_nuisance_groups": 3}
QUAL = 95
read, write, digest = jpeg.read, jpeg.write, jpeg.digest


def pin(path):
    return digest(path.read_bytes())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def selected(older, held):
    plans = [read(folder / "plan.json") for folder in (older, held)]
    for name in ("flickr_owner", "flickr_photo_id"):
        groups = [{row[name] for row in plan["cases"]} for plan in plans]
        need([len(group) for group in groups] == [12, 12] and not groups[0] & groups[1],
             "Require development and confirmation photo/creator separation")
    source = list(panel.inputs(held))
    need(len(source) == 12 and Counter(row["generator"] for row, _, _ in source) == {"sdxl": 6, "kandinsky": 6},
         "Frozen held source coverage differs")
    return source


def image_bytes(pixels):
    with io.BytesIO() as output:
        Image.fromarray(pixels).save(output, "PNG")
        return output.getvalue()


def observations(older, held):
    source = selected(older, held)
    for index, (row, native, _) in enumerate(source):
        slot, box = row["slot"], row["box"]
        master = native["original"]
        encoded = jpeg.jpeg(master, QUAL)
        earlier = jpeg.pixels(encoded)
        donor = source[(index + 1) % len(source)][1]["original"]
        tested = list(prior.cases(slot, native, box, donor))
        baseline = next(pixels for role, pixels, _, _ in tested if role == "untouched_crop")
        matched = next(pixels for role, pixels, _, _ in prior.cases(slot, {**native, "original": earlier}, box, donor)
                       if role == "untouched_crop")
        # The ordinary retouch makes a smooth local color adjustment over the known native footprint.
        toned = np.clip(master.astype(np.int16) + [30, -24, 18], 0, 255).astype(np.uint8)
        tone = prior.panel.blend.local(master, toned, box, feather=True)
        tone_view = next(pixels for role, pixels, _, _ in prior.cases(slot, {**native, "edited": tone}, box, donor)
                         if role == "sharp")
        geometry = next(crop for role, _, _, crop in tested if role == "untouched_crop")
        tested += [("matched_earlier", matched, None, geometry), ("smooth_tone", tone_view, None, geometry)]
        for role, pixels, _, crop in tested:
            truth = np.any(pixels != baseline, axis=2) if role in POSITIVE else np.zeros(pixels.shape[:2], dtype=bool)
            submitted = image_bytes(pixels)
            row_pin = {"slot": slot, "generator": row["generator"], "role": role, "crop": crop,
                       "earlier_sha256": digest(encoded), "submitted_sha256": digest(submitted),
                       "truth_sha256": digest(truth.tobytes()), "truth_pixels": int(truth.sum())}
            yield row_pin, encoded, submitted, earlier, pixels, truth


def freeze(older, held):
    modules = (Path(__file__), *(Path(m.__file__) for m in (prior, jpeg, panel, review, reference_alignment,
               reference_compare, reference_crop_resize, reference_resize, image_reference)))
    return {"schema": SCHEMA, "earlier_jpeg_quality": QUAL, "submitted": "single-frame upright RGB PNG",
            "gate": GATE, "code": {path.name: pin(path) for path in modules},
            "source_receipts": {role: {name: pin(folder / name) for name in ("plan.json", "attempts.jsonl", "result.json")}
                                for role, folder in (("development", older), ("held", held))},
            "runtime": {"opencv": cv2.__version__, "pillow": pillow_version, "numpy": np.__version__,
                        "jpeg": features.version_codec("jpg")},
            "useful": {"minimum_iou": .30, "minimum_recall": .50, "requires_exact_crop": True},
            "rows": [row for row, *_ in observations(older, held)]}


def stats(mask, truth):
    area, target, overlap = int(mask.sum()), int(truth.sum()), int((mask & truth).sum())
    return {"selected": area, "target": target, "overlap": overlap,
            "useful": target > 0 and 10 * overlap >= 3 * (area + target - overlap) and 2 * overlap >= target}


def _native_mapping(record, pixels):
    reference = record["reference_crop_bounds"]
    width, height = pixels.shape[1], pixels.shape[0]
    for region in record["regions"]:
        x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
        left = reference["x"] + floor(Fraction(x * reference["w"], width))
        top = reference["y"] + floor(Fraction(y * reference["h"], height))
        right = reference["x"] + ceil(Fraction((x + w) * reference["w"], width))
        bottom = reference["y"] + ceil(Fraction((y + h) * reference["h"], height))
        need(region["reference_region"] == {"x": left, "y": top, "w": right - left, "h": bottom - top},
             "Native projection differs from its independent integer bounds")


def measure(older, held, planned, work):
    for index, (case, expected) in enumerate(zip(observations(older, held), planned["rows"], strict=True), 1):
        key, before, after, ref, sub, truth = case
        need(key == expected, "Submitted or authored source commitments differ")
        with TemporaryDirectory(dir=work, prefix="paired-native-") as scratch:
            scratch = Path(scratch)
            previous, submitted = scratch / "earlier.jpg", scratch / "submitted.png"
            previous.write_bytes(before)
            submitted.write_bytes(after)
            base_dir, extra_dir = scratch / "base", scratch / "extra"
            current = image_reference.inspect(previous, submitted, evidence_dir=base_dir, allow_proportional_resize=True,
                                              allow_oriented_crop=True, allow_oriented_resize=True, allow_embedded_photo=True)
            prior_mask, _ = review.verify_assets(base_dir, current, ref, sub)
            previous_crop = current.get("reference_crop_bounds")
            previous_crop = {k: previous_crop[k] for k in ("x", "y", "w", "h")} if previous_crop else None
            if current["status"] != "comparable":
                with Image.open(io.BytesIO(after), formats=["PNG"]) as carrier:
                    candidate, mask = reference_crop_resize.compare(ref, sub, carrier)
                    if not image_reference.accepted_crop_resize(candidate, carrier.size):
                        candidate, mask = None, None
            else:
                candidate, mask = None, None
            if candidate is None:
                record = {"status": "not_selected", "regions": []}
                actual_mask, overlays, crop = np.zeros_like(truth), 0, None
            else:
                _native_mapping(candidate, sub)
                image_reference._write_evidence(extra_dir, candidate, ref, sub, mask)
                record = read(extra_dir / "comparison.json")
                actual_mask, overlays = review.verify_assets(extra_dir, record, ref, sub)
                need(np.array_equal(actual_mask, mask), "Readback does not reproduce the recorded mask")
                crop = {k: record["reference_crop_bounds"][k] for k in ("x", "y", "w", "h")}
            baseline_stats, candidate_stats = stats(prior_mask, truth), stats(actual_mask, truth)
            if current["status"] == "comparable":
                proposed_stats = baseline_stats
            else:
                proposed_stats = candidate_stats | {"useful": candidate_stats["useful"] and crop == key["crop"]}
            yield {**key, "current_status": current["status"], "candidate_status": record["status"],
                   "current_crop": previous_crop, "candidate_crop": crop, "candidate_mask_sha256": digest(actual_mask.tobytes()),
                   "current": baseline_stats, "proposed": proposed_stats,
                   "candidate_crop_assets_verified": 2 * overlays, "candidate_overlays_verified": overlays}
        if index % 12 == 0:
            print(f"{index}/{len(planned['rows'])} pairs checked", flush=True)


def summarize(rows):
    summary, nuisance_sources = {}, set()
    wrong = 0
    for row in rows:
        role, slot = row["role"], row["slot"]
        counts = summary.setdefault(role, Counter())
        useful = row["proposed"]["useful"]
        introduced = row["candidate_status"] == "comparable"
        selected = introduced and row["proposed"]["selected"] > 0
        counts.update(views=1, current_compared=row["current_status"] == "comparable", candidate_compared=introduced,
                      candidate_exact_crop=introduced and row["candidate_crop"] == row["crop"],
                      current_useful=row["current"]["useful"], proposed_useful=useful,
                      added_useful=useful and not row["current"]["useful"],
                      candidate_highlighted=selected, current_highlighted=row["current"]["selected"] > 0,
                      candidate_exact_no_difference=introduced and row["candidate_crop"] == row["crop"] and not selected)
        if introduced and (row["candidate_crop"] != row["crop"] or role in INCOMPATIBLE):
            wrong += 1
        if role in NUISANCE and selected:
            nuisance_sources.add(slot)
    outcome = {"added_feathered": summary["feathered"]["added_useful"],
               "added_smooth_tone": summary["smooth_tone"]["added_useful"],
               "exact_matched_earlier": summary["matched_earlier"]["candidate_exact_no_difference"]
               + sum(r["role"] == "matched_earlier" and r["current_status"] == "comparable"
                     and r["current_crop"] == r["crop"] and r["current"]["selected"] == 0 for r in rows),
               "wrong_or_incompatible_crops": wrong,
               "extra_nuisance_views": sum(summary[role]["candidate_highlighted"] for role in NUISANCE),
               "extra_nuisance_groups": len(nuisance_sources)}
    passed = all(outcome[key.split("_", 1)[1]] >= value if key.startswith("minimum_")
                 else outcome[key.split("_", 1)[1]] <= value for key, value in GATE.items())
    return {"scope_gate_passed": passed, "counts": outcome, "strata": summary, "views": len(rows),
            "verified_source_crops": sum(r["candidate_crop_assets_verified"] for r in rows),
            "verified_native_overlays": sum(r["candidate_overlays_verified"] for r in rows)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "run", "review"))
    cli.add_argument("development", type=Path)
    cli.add_argument("held", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    cv2.setNumThreads(1)
    committed = freeze(args.development, args.held)
    plan = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(plan, committed)
        print(pin(plan), flush=True)
        return
    need(read(plan) == committed, "Code, rights-bound inputs, or exact truth changed after the freeze")
    rows = list(measure(args.development, args.held, committed, args.output))
    result = {"schema": SCHEMA, "freeze_sha256": pin(plan), "rows": rows, "summary": summarize(rows)}
    destination = args.output / "result.json"
    if args.stage == "review":
        need(read(destination) == json.loads(json.dumps(result)), "Native evidence or outcomes changed on replay")
    else:
        write(destination, result)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
