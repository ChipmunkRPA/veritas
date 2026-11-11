"""Compare single-image local JPEG grid measurements on matched native pixels."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import jpeg_dq, local_jpeg_grid
from app.detectors.image.local_jpeg_grid import AC_THRESHOLD, ARMS, HOP, TILE, response
from scripts import image_background_patch_score as panel
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-local-grid-transfer/1"
ROLES = ("original", "global_shift", "contrast", "blur", "aligned_patch", "sharp_shift", "feathered_shift")
EDITS = ("sharp_shift", "feathered_shift")
FORMATS = {"png": 0, "jpeg90": 90, "jpeg75": 75}
sha, write = shared.file_digest, shared.write


def selection(result, arm, point, shape):
    mask, _ = local_jpeg_grid.pixel_map(result, arm, point, shape)
    regions = [{"phase_xy": [phase % 8, phase // 8], "cells": area}
               for phase, _, (_, _, _, _, area) in local_jpeg_grid.selected_regions(result, arm, point)]
    return mask > 0, regions


def jpeg_copy(rgb, quality, shift=(0, 0)):
    x, y = shift
    padded = np.pad(rgb, ((y, 0), (x, 0), (0, 0)), mode="reflect")
    with io.BytesIO() as buffer:
        Image.fromarray(padded).save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as im:
            return np.array(im.convert("RGB"))[y:, x:].copy()


def materials(original, donor, box):
    base = jpeg_copy(original, 67)
    shifted = jpeg_copy(donor, 59, (3, 5))
    photo = Image.fromarray(base)
    return dict(zip(ROLES, (base, jpeg_copy(original, 67, (3, 5)),
                            blend.local(base, np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), box),
                            blend.local(base, np.asarray(photo.filter(ImageFilter.GaussianBlur(1.25))), box),
                            blend.local(base, jpeg_copy(donor, 59), box), blend.local(base, shifted, box),
                            blend.local(base, shifted, box, feather=True)), strict=True))


def delivery(pixels, quality):
    return shared.flux.previous.encode(Image.fromarray(pixels), quality)


def fit(folder):
    _, plan, receipts = scoring.checked_inputs(folder)
    maximum = {fmt: {arm: [] for arm in ARMS} for fmt in FORMATS}
    for _, _, native, _ in scoring.panels(folder, plan, receipts):
        source = native["original"]
        h, w = source.shape[:2]
        views = materials(source, source, (w//3, h//3, w*7//12, h*7//12))
        for fmt, quality in FORMATS.items():
            controls = [response(np.asarray(delivery(views[role], quality)[1]))["scores"] for role in ROLES if role not in EDITS]
            for arm in ARMS:
                maximum[fmt][arm].append(max(row[arm] for row in controls))
    return {fmt: {arm: {"values": values, "point": max(.025, sorted(values)[-2])} for arm, values in arms.items()}
            for fmt, arms in maximum.items()}


def compare(intake, points):
    for case, native, _ in panel.inputs(intake):
        views = materials(native["original"], native["edited"], case["box"])
        for role, pixels in views.items():
            target = np.any(pixels != views["original"], axis=2) if role in EDITS else np.zeros(pixels.shape[:2], bool)
            for fmt, quality in FORMATS.items():
                body, im = delivery(pixels, quality)
                result = response(np.asarray(im))
                measures = {}
                for arm in ARMS:
                    mask, regions = selection(result, arm, points[fmt][arm]["point"], target.shape)
                    measures[arm] = {**scoring.measure(mask, target), "regions": regions}
                baseline = np.zeros_like(target)
                if quality and (dq := jpeg_dq.analyse(body)) is not None and (region := dq.localised) is not None:
                    baseline[8*region.y:8*(region.y+region.h), 8*region.x:8*(region.x+region.w)] = True
                measures["existing_aligned_grid"] = scoring.measure(baseline, target)
                yield {"slot": case["slot"], "generator": case["generator"], "role": role, "format": fmt,
                       "input_sha256": shared.digest(body), "truth_sha256": shared.digest(target.tobytes()),
                       "global_phase": result["global_phase"], "measurements": measures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("development", "earlier", "later", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    sources = (args.development, args.earlier, args.later)
    groups = [{case["flickr_photo_id"] for case in json.loads((path/"plan.json").read_bytes())["cases"]} for path in sources]
    assert [len(g) for g in groups] == [8, 12, 12] and len(set.union(*groups)) == 32
    freeze = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)), "sources": [sha(p/"result.json") for p in sources],
              "baseline_sha256": sha(Path(jpeg_dq.__file__)), "signal_sha256": sha(Path(local_jpeg_grid.__file__)), "tile_px": TILE * 8, "hop_px": HOP * 8, "formats": FORMATS,
              "roles": list(ROLES), "ac_threshold": AC_THRESHOLD, "prior_background_quality": 67,
              "prior_donor_quality": 59, "donor_padding_xy": [3, 5], "minimum_review_point": .025}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output/"freeze.json", freeze)
        return
    assert freeze == json.loads((args.output/"freeze.json").read_bytes())
    cv2.setNumThreads(1)
    start = time.monotonic()
    points = fit(args.development)
    assert all(len(v["values"]) == 8 for arms in points.values() for v in arms.values())
    write(args.output/"points.json", points)
    print("Sealed phase review points", flush=True)
    results = {}
    for name, path in (("earlier", args.earlier), ("later", args.later)):
        rows = list(compare(path, points))
        assert len(rows) == 252
        write(args.output/f"{name}-rows.json", rows)
        results[name] = sha(args.output/f"{name}-rows.json")
        print(f"Completed {name} panel", flush=True)
    write(args.output/"result.json", {"schema": SCHEMA, "paired_views": 504, "development_views": 120,
          "freeze_sha256": sha(args.output/"freeze.json"), "points_sha256": sha(args.output/"points.json"),
          "journal_sha256": results, "elapsed_s": round(time.monotonic()-start, 3)})


if __name__ == "__main__":
    main()
