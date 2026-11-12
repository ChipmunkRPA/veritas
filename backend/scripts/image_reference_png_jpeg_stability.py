# SPDX-License-Identifier: Apache-2.0
"""Confirm strong local color differences after an earlier-JPEG and later-PNG crop resize."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from app.detectors.image import reference_compare, reference_crop_resize, reference_resize
from app.evidence import image_reference
from scripts import image_flux_photo_expansion_v285 as flux
from scripts import image_jpeg_copy_public_replay as public
from scripts import image_reference_png_jpeg_transfer as prior

SCHEMA = "chipmunk.image-reference-png-jpeg-stability/1"
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
PRIVATE = ROOT / "backend/data/independent-evaluation"
MIN_RESIDUAL = 10
GATE = {"min_ai_feathered_added": 3, "min_ai_sharp_added": 3, "min_ordinary_smooth_added": 4,
        "max_nuisance_highlights": 1, "max_wrong_crops": 0}
AI_EDITS = ("sharp", "feathered")
CONTROLS = prior.NUISANCE | prior.INCOMPATIBLE
REFERENCE = prior.QUAL
read, write, digest, pin, need = prior.read, prior.write, prior.digest, prior.pin, prior.need


def refine(reference, submitted, initial, rectangle):
    x, y, w, h = (rectangle[k] for k in ("x", "y", "w", "h"))
    image = Image.fromarray(reference[y:y + h, x:x + w])
    actual = cv2.GaussianBlur(submitted.astype(np.float32), (0, 0), 1.25)
    stable = initial.copy()
    for kernel in reference_resize.KERNELS:
        candidate = np.asarray(image.resize(submitted.shape[1::-1], kernel)).astype(np.float32)
        expected = cv2.GaussianBlur(candidate, (0, 0), 1.25)
        residual = np.zeros(initial.shape, dtype=np.float32)
        for channel in range(3):
            before, after = expected[:, :, channel], actual[:, :, channel]
            gain, offset = reference_compare._tone(before, after)
            np.maximum(residual, np.abs(after - np.clip(before * gain + offset, 0, 255)), out=residual)
        stable &= residual >= MIN_RESIDUAL
    regions, mask, omitted = reference_compare._regions(stable, max(48, round(stable.size * .0015)))
    for region in regions:
        crop = reference_resize.native_crop(region, (w, h), submitted.shape[1::-1])
        region["reference_region"] = {**crop, "x": crop["x"] + x, "y": crop["y"] + y}
    return regions, mask, omitted


def _licensed_flux(earlier_manifest):
    held = PRIVATE / "image-flux-photo-expansion-v285"
    source_plan = held / "private-inputs.json"
    frozen = read(PUBLIC / "image-flux-photo-expansion-v285-freeze.json")
    result = read(PUBLIC / "image-flux-photo-expansion-complete-v285.json")
    need(pin(source_plan) == frozen["private_inputs_sha256"], "Earlier source assignments differ")
    cache = read(PRIVATE / "image-fresh-source-cache-v39/private-groups.json")
    by_pin = {r.get("source_jpeg_sha256"): r for r in cache["sources"]}
    occupied = {r["flickr_owner"].lower() for r in earlier_manifest["cases"]}
    output = []
    for row in read(source_plan)["rows"]:
        slot = row["slot"]
        source = by_pin[row["source_jpeg_sha256"]]
        need(source["author"].lower() not in occupied, "A source contributor overlaps the scored panel")
        occupied.add(source["author"].lower())
        native = Path(source["source_jpeg_file"])
        need(pin(native) == row["source_jpeg_sha256"] and source["license_url"] == "https://creativecommons.org/licenses/by/2.0/",
             "Licensed original differs")
        rgb = prior.jpeg.pixels(native.read_bytes())
        need(digest(rgb.tobytes()) == row["source_rgb_sha256"], "Licensed native RGB differs")
        n = int(slot[-1])
        batch = held / f"batch-{n:02d}/authored/output"
        need(pin(batch / "result.json") == result["batch_receipts"][str(n)]["result_sha256"], "Generator receipt differs")
        entry = next(r for r in read(batch / "result.json")["cases"] if r["slot"] == slot)
        need(flux.verify_case(row, entry, batch / slot)["status"] == "qualified_geometry", "Generator edit is not exact")
        edited = flux.read_pixels(batch / slot / "composite.png", entry["composite_png_sha256"], "RGB")
        a, b, c, d = row["crop_ltrb_exclusive"]
        l, t, r, bot = row["mask_ltrb_exclusive"]
        left = max(0, min(rgb.shape[1] - 600, a - 172))
        top = max(0, min(rgb.shape[0] - 550, b - 145))
        native_crop = rgb[top:top + 550, left:left + 600].copy()
        prepared = native_crop.copy()
        prepared[b - top:d - top, a - left:c - left] = edited
        target = (a + l - left, b + t - top, a + r - left, b + bot - top)
        output.append((row["source_jpeg_sha256"], native_crop, prepared, target))
    need(len(output) == 6, "Need the six exact independent local generative edits")
    return output, occupied


def _public_controls(public_dir, occupied):
    result = []
    for index, row in enumerate(public.roster()[:6]):
        need(row["author"].casefold() not in occupied, "Public photo contributor overlaps the generator group")
        _, source = public._source(public_dir / row["file"], row)
        original = np.asarray(ImageOps.contain(source, (700, 700), Image.Resampling.LANCZOS))
        h, w = original.shape[:2]
        box = (w // 4 + index, h // 4, w // 4 + w // 5 + index, h // 4 + h // 6)
        result.append((row["source_sha256"], original, original, box))
    return result


def inputs(public_dir, earlier_manifest):
    ai, used = _licensed_flux(read(earlier_manifest))
    ordinary = _public_controls(public_dir, used)
    for cohort, group in (("ai", ai), ("ordinary", ordinary)):
        for index, (source, original, edited, box) in enumerate(group):
            donor = group[(index + 1) % len(group)][1]
            base = list(prior.prior.cases(index, {"original": original, "edited": edited}, box, donor))
            baseline = next(view for role, view, _, _ in base if role == "untouched_crop")
            roles = CONTROLS | {"local_blur"} | (set(AI_EDITS) if cohort == "ai" else set())
            toned = np.asarray(ImageEnhance.Color(Image.fromarray(original)).enhance(.42))
            toned = np.clip(toned.astype(np.int16) + [17, -15, 11], 0, 255).astype(np.uint8)
            ordinary_edit = prior.prior.panel.blend.local(original, toned, box, feather=True)
            tone = next(row for row in prior.prior.cases(index, {"original": original, "edited": ordinary_edit}, box, donor)
                        if row[0] == "sharp")
            case_items = [row for row in base if row[0] in roles]
            case_items.append(("smooth_tone", *tone[1:]))
            before = prior.jpeg.jpeg(original, REFERENCE)
            ref = prior.jpeg.pixels(before)
            for role, pixels, _, crop in case_items:
                after = prior.image_bytes(pixels)
                truth = (np.any(pixels != baseline, axis=2) if role in prior.POSITIVE
                         else np.zeros(pixels.shape[:2], dtype=bool))
                yield {"source": source, "cohort": cohort, "slot": index, "role": role, "crop": crop,
                       "earlier_sha256": digest(before), "submitted_sha256": digest(after),
                       "truth_sha256": digest(truth.tobytes())}, before, ref, after, pixels, truth


def freeze(public_dir, earlier_manifest):
    code = (Path(__file__), Path(prior.__file__), Path(reference_crop_resize.__file__), Path(reference_resize.__file__),
            Path(reference_compare.__file__), Path(image_reference.__file__))
    return {"schema": SCHEMA, "gate": GATE, "minimum_residual_rgb_each_kernel": MIN_RESIDUAL,
            "source_selection": "Six retained separately attributed FLUX groups and first six pinned public JPEG replay sources",
            "code": {path.name: pin(path) for path in code}, "rows": [row for row, *_ in inputs(public_dir, earlier_manifest)]}


def measure(public_dir, earlier, frozen, work):
    rows = []
    with TemporaryDirectory(dir=work) as scratch:
        scratch = Path(scratch)
        original, submitted = scratch / "reference.jpg", scratch / "submitted.png"
        for index, (item, expected) in enumerate(zip(inputs(public_dir, earlier), frozen["rows"], strict=True), 1):
            row, before, reference, after, pixels, truth = item
            need(row == expected, "Native input differs from its freeze")
            original.write_bytes(before)
            submitted.write_bytes(after)
            current = image_reference.inspect(original, submitted, allow_proportional_resize=True, allow_oriented_crop=True,
                                              allow_oriented_resize=True, allow_embedded_photo=True)
            zero = np.zeros(pixels.shape[:2], dtype=bool)
            status, crop, raw, stable, assets = "not_selected", None, zero, zero, 0
            if current["status"] != "comparable":
                with Image.open(io.BytesIO(after), formats=["PNG"]) as carrier:
                    candidate, mask = reference_crop_resize.compare(reference, pixels, carrier)
                if image_reference.accepted_crop_resize(candidate, pixels.shape[1::-1]):
                    status, raw = candidate["status"], mask
                    crop = {key: candidate["reference_crop_bounds"][key] for key in ("x", "y", "w", "h")}
                    regions, stable, omitted = refine(reference, pixels, mask, crop)
                    candidate.update(regions=regions, additional_regions=omitted)
                    prior._native_mapping(candidate, pixels)
                    with TemporaryDirectory(dir=work) as output:
                        evidence = Path(output) / "evidence"
                        image_reference._write_evidence(evidence, candidate, reference, pixels, stable)
                        readback, assets = prior.review.verify_assets(evidence, candidate, reference, pixels)
                    need(np.array_equal(stable, readback), "Native artifact pixels differ")
            rows.append({**row, "current_status": current["status"], "status": status, "measured_crop": crop,
                         "raw": prior.stats(raw, truth), "stable": prior.stats(stable, truth),
                         "stable_mask_sha256": digest(stable.tobytes()), "overlays_verified": assets})
            if index % 12 == 0:
                print(f"{index}/{len(frozen['rows'])} pairs checked", flush=True)
    return rows


def result(rows):
    stats, counts = {}, Counter()
    for row in rows:
        role, cohort = row["role"], row["cohort"]
        accepted = row["status"] == "comparable"
        exact = accepted and row["crop"] == row["measured_crop"]
        bucket = stats.setdefault(f"{cohort}/{role}", Counter())
        bucket.update(views=1, exact=bool(exact), raw_useful=exact and row["raw"]["useful"],
                      stable_useful=exact and row["stable"]["useful"], raw_highlighted=bool(row["raw"]["selected"]),
                      stable_highlighted=bool(row["stable"]["selected"]))
        if accepted and not exact:
            counts["wrong_crops"] += 1
        if role in CONTROLS and row["stable"]["selected"]:
            counts["nuisance_highlights"] += 1
        if exact and row["stable"]["useful"]:
            counts[f"{cohort}_{role}_added"] += 1
    # The original freeze uses the short name for this prespecified retouch role.
    counts["ordinary_smooth_added"] = counts["ordinary_smooth_tone_added"]
    conditions = {k: counts[k[4:]] >= bound if k.startswith("min_") else counts[k[4:]] <= bound for k, bound in GATE.items()}
    return {"counts": dict(counts), "strata": stats, "conditions": conditions, "passed": all(conditions.values()),
            "views": len(rows), "verified_source_crops": 2 * sum(row["overlays_verified"] for row in rows),
            "verified_overlays": sum(row["overlays_verified"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("public_dir", type=Path)
    parser.add_argument("earlier_manifest", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    path = args.work / "freeze.json"
    frozen = freeze(args.public_dir, args.earlier_manifest)
    if args.stage == "freeze":
        args.work.mkdir(mode=0o700)
        write(path, frozen)
        print(pin(path), flush=True)
        return
    need(read(path) == frozen, "Preparation changed after freeze")
    rows = measure(args.public_dir, args.earlier_manifest, frozen, args.work)
    measured = {"freeze_sha256": pin(path), "result": result(rows), "rows": rows}
    output = args.work / "result.json"
    if args.stage == "review":
        need(json.loads(json.dumps(measured)) == read(output), "Native readback differs")
    else:
        write(output, measured)
    print(json.dumps(measured["result"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
