# SPDX-License-Identifier: Apache-2.0
"""Evaluate a second separated native boundary, using a source-disjoint fit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.detectors.image import sharp_boundary as fixed
from scripts import image_reference_orientation_transfer as source
from scripts import image_small_boundary_control_stress as common
from scripts import review_image_boundary_multiple_transfer as review

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path(__file__).with_name("capture_image_boundary_multiple.cjs")
SIZES = ((704, 544), (640, 608), (768, 512), (512, 704))
FIT_PHOTOS, HELD_PHOTOS = 8, 12
ARMS = ("separated", "frame_excluded")
POSITIVE = ("double_sharp", "sharp_feathered")
SINGLE = ("single_left", "single_right")
ORDINARY = ("original", "global_tone", "roundtrip_resize", "local_blur")
SCREEN = ("overlay", "tiles", "photos")
SPACING = 20
RIM_SHARE_NUMERATOR, RIM_SHARE_DENOMINATOR = 4, 5
RIM_MIN_SIDES = 3
RIM_OFFSETS = (-1, 0, 1)
RULES = {"minimum_sharp_pairs": 6, "maximum_added_screen_views": 2, "maximum_added_screen_groups": 1,
         "maximum_added_photo_views": 2, "maximum_added_photo_groups": 2, "maximum_wrong_single_groups": 1}
read = common.read
sha = common.digest
require = common.require


def plan(archive):
    inventory = source.binding(archive)["sources"]
    # Two recorded aliases describe one contributor. Slot 27 depicts children.
    eligible = (row for row in inventory if row["slot"] not in (19, 27))
    ordered = sorted(eligible, key=lambda row: sha(f"image-boundary-multiple/1/{row['sha256']}".encode()))
    return {"fit": ordered[:FIT_PHOTOS], "held": ordered[FIT_PHOTOS:FIT_PHOTOS+HELD_PHOTOS],
            "screen": ordered[FIT_PHOTOS+HELD_PHOTOS:]}


def photo(archive, record, size):
    with Image.open(archive / record["file"]) as raw:
        return ImageOps.fit(ImageOps.exif_transpose(raw).convert("RGB"), size, Image.Resampling.LANCZOS)


def prepare(archive, run):
    run.mkdir(mode=0o700)
    path = run / "screen-photos"
    path.mkdir()
    for slot, record in enumerate(plan(archive)["screen"]):
        photo(archive, record, (768, 544)).save(path / f"{slot}.jpg", quality=91, subsampling=0)


def lock(archive, run):
    roster = plan(archive)
    capture_plan, captures = (read(run / name) for name in ("capture-plan.json", "captures.json"))
    require(len(capture_plan["groups"]) == len(roster["screen"]) == 16, "Incomplete authored UI roster")
    expected = {(record["slot"], variant) for record in capture_plan["groups"] for variant in SCREEN}
    rgb_hashes = {}
    for record in captures["rows"]:
        slot, variant = record["slot"], record["variant"]
        expected.remove((slot, variant))
        name = f"{slot}-{variant}.png"
        require(record["filename"] == name and sha((run / "captures" / name).read_bytes()) == record["png_sha256"]
                and sha((run / "screen-photos" / f"{slot}.jpg").read_bytes()) == record["source_sha256"],
                "An authored UI input changed")
        with Image.open(run / "captures" / name) as rgb:
            group = capture_plan["groups"][slot]
            require(rgb.size == (group["width"], group["height"]), "A native Chromium viewport changed")
            rgb_hashes[name] = sha(rgb.convert("RGB").tobytes())
    require(not expected and len(set(rgb_hashes.values())) == 48, "Expected 48 unique Chromium views")
    code = (Path(__file__), Path(fixed.__file__), Path(source.__file__), Path(common.__file__), Path(review.__file__), CAPTURE)
    return {"schema": review.SCHEMA, "source_receipt_sha256": sha(source.SOURCE.read_bytes()), "roster": roster,
            "capture_plan": capture_plan, "captures": captures, "capture_rgb_sha256": rgb_hashes,
            "code_sha256": {path.name: sha(path.read_bytes()) for path in code},
            "fixed_point": fixed.REVIEW_POINT, "rules": RULES, "arms": list(ARMS), "separation_pixels": SPACING,
            "roles": {"positive": list(POSITIVE), "single": list(SINGLE), "photo_control": list(ORDINARY),
                      "screen_control": list(SCREEN)},
            "rim": {"support": [RIM_SHARE_NUMERATOR, RIM_SHARE_DENOMINATOR], "side_minimum": RIM_MIN_SIDES,
                    "offsets": list(RIM_OFFSETS)},
            "points": "Strictly above the second-largest fit source maximum, with the shipped first point as floor.",
            "holdout": "12 photo groups and 12 Chromium groups. Fit has 8 photo groups and 4 Chromium groups."}


def secondary(pixels, first):
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = first
    offset = fixed.CONTEXT_OFFSET
    rgb = pixels.astype(np.float32)
    v, h = fixed._step_response(rgb), fixed._step_response(rgb.transpose(1, 0, 2)).T
    v = np.pad(v.cumsum(0), ((1, 0), (0, 0)))
    h = np.pad(h.cumsum(1), ((0, 0), (1, 0)))
    best = None
    for w in (round(width * frac) for frac in fixed.OFFSETS):
        left = np.arange(offset, width - offset - w)
        right = left + w
        apart_x = (right + SPACING <= x0) | (left - SPACING >= x1)
        for d in (round(height * frac) for frac in fixed.OFFSETS):
            top = np.arange(offset, height - offset - d)
            bottom = top + d
            apart_y = (bottom + SPACING <= y0) | (top - SPACING >= y1)
            scores = (v[np.ix_(bottom, left)] - v[np.ix_(top, left)]) / d
            scores *= (v[np.ix_(bottom, right)] - v[np.ix_(top, right)]) / d
            scores *= (h[np.ix_(top, right)] - h[np.ix_(top, left)]) / w
            scores *= (h[np.ix_(bottom, right)] - h[np.ix_(bottom, left)]) / w
            scores[~(apart_x[None, :] | apart_y[:, None])] = -1
            row, col = np.unravel_index(int(scores.argmax()), scores.shape)
            score = float(max(0, scores[row, col]) ** .25)
            if scores[row, col] >= 0 and (best is None or score > best["score"]):
                best = {"box": [int(left[col]), int(top[row]), int(right[col]), int(bottom[row])], "score": score}
    return best


def rim_support(rgb, box):
    x0, y0, x1, y1 = box
    packed = rgb.astype(np.uint32)
    pixels = packed[:, :, 0] << 16 | packed[:, :, 1] << 8 | packed[:, :, 2]
    supported = {}
    for side in range(4):
        for step in RIM_OFFSETS:
            line = (pixels[y0+3:y1-3, x0+step] if side == 0 else pixels[y0+3:y1-3, x1+step] if side == 1 else
                    pixels[y0+step, x0+3:x1-3] if side == 2 else pixels[y1+step, x0+3:x1-3])
            colors, counts = np.unique(line, return_counts=True)
            for color in colors[counts * RIM_SHARE_DENOMINATOR >= len(line) * RIM_SHARE_NUMERATOR]:
                supported.setdefault(int(color), set()).add(side)
    return max(map(len, supported.values()), default=0)


def targets(size, slot):
    width, height = size
    dimensions = [(round(width * fixed.OFFSETS[(slot + index) % 3]),
                   round(height * fixed.OFFSETS[(slot // 3 + 2 * index) % 3])) for index in range(2)]
    left, top = 53 + slot * 4, 78 + slot % 5 * 8
    w, h = dimensions[0]
    first = left, top, left + w, top + h
    w, h = dimensions[1]
    right, bottom = width - 49 - slot * 2, height - 63 - slot % 4 * 9
    return first, (right-w, bottom-h, right, bottom)


def views(archive, roster, slot):
    size = SIZES[slot % len(SIZES)]
    count = len(roster)
    host = photo(archive, roster[slot], size)
    original = np.asarray(host)
    a, b = targets(size, slot)
    require(review.apart(a, b, SPACING), "A truth pair violates declared separation")
    donors = [photo(archive, roster[(slot + delta) % count], size) for delta in (1, 5)]
    patches = [donor.crop((101, 113, 101+right-left, 113+bottom-top))
               for donor, (left, top, right, bottom) in zip(donors, (a, b), strict=True)]
    yield "original", original, []
    yield "global_tone", np.asarray(ImageEnhance.Contrast(host).enhance(1.13)), []
    yield "roundtrip_resize", np.asarray(common.resize(host)), []
    mask = Image.new("L", size)
    mask.paste(255, a)
    local = Image.composite(host.filter(ImageFilter.GaussianBlur(1.25)), host, mask.filter(ImageFilter.GaussianBlur(8)))
    yield "local_blur", np.asarray(local), []
    insertions = (*zip(SINGLE, ((0,), (1,)), strict=True), *((role, (0, 1)) for role in POSITIVE))
    for role, selected in insertions:
        edited, truths = host.copy(), []
        for index in selected:
            box, patch = (a, b)[index], patches[index]
            alpha = None
            if role == POSITIVE[1] and index == 1:
                alpha = Image.new("L", patch.size)
                alpha.paste(255, (12, 12, patch.width-12, patch.height-12))
                alpha = alpha.filter(ImageFilter.GaussianBlur(5))
            before = np.asarray(edited).copy()
            edited.paste(patch, box[:2], alpha)
            truth = np.any(np.asarray(edited) != before, axis=2)
            x0, y0, x1, y1 = box
            require(0 < int(truth.sum()) == int(truth[y0:y1, x0:x1].sum()), "The exact source edit escaped its target box")
            truths.append((box, truth))
        yield role, np.asarray(edited), truths


def audit_source_crop(rgb, proposal):
    x, y, right, bottom = proposal["box"]
    with Image.fromarray(rgb).crop((x, y, right, bottom)) as crop:
        require(sha(crop.tobytes()) == proposal["crop_rgb_sha256"], "Independent crop bytes differ")
    pixels = rgb.astype(np.float32)
    v, h = fixed._step_response(pixels), fixed._step_response(pixels.transpose(1, 0, 2)).T
    side_means = [float(line.mean()) for line in (v[y:bottom, x], v[y:bottom, right], h[y, x:right], h[bottom, x:right])]
    expected = np.prod(side_means) ** .25
    require(abs(expected - proposal["score"]) <= 0.00001, "Direct edge sums differ from the prefix search")


def observe(domain, slot, role, rgb, truths, encoded_sha=None, audit=False):
    before = fixed.measure(rgb)
    first = {"box": list(before.box), "score": before.score}
    second = secondary(rgb, before.box) if before.score > fixed.REVIEW_POINT else None
    if second:
        second["rim_sides"] = rim_support(rgb, second["box"])
    for proposal in filter(None, (first, second)):
        x, y, right, bottom = proposal["box"]
        proposal["crop_rgb_sha256"] = sha(rgb[y:bottom, x:right].tobytes())
        if audit:
            audit_source_crop(rgb, proposal)
    records = []
    for box, truth in truths:
        overlaps = []
        for proposal in (first, second):
            if proposal is None:
                overlaps.append(0)
                continue
            x, y, r, b = proposal["box"]
            overlaps.append(int(truth[y:b, x:r].sum()))
        records.append({"box": list(box), "sha256": sha(truth.tobytes()), "pixels": int(truth.sum()), "overlaps": overlaps})
    data = source.encode(Image.fromarray(rgb), "PNG") if encoded_sha is None else None
    return {"domain": domain, "slot": slot, "role": role, "size": [rgb.shape[1], rgb.shape[0]],
            "input_sha256": encoded_sha or sha(data), "rgb_sha256": sha(rgb.tobytes()), "first": first, "second": second,
            "truths": records}


def measurements(archive, run, frozen, cohort, audit=False):
    rows = []
    for slot in range(len(frozen["roster"][cohort])):
        for role, rgb, truths in views(archive, frozen["roster"][cohort], slot):
            if cohort == "held" or role not in POSITIVE:
                rows.append(observe("photo", slot, role, rgb, truths, audit=audit))
    for capture in frozen["captures"]["rows"]:
        if (capture["slot"] < 4) != (cohort == "fit"):
            continue
        with Image.open(run / "captures" / capture["filename"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        rows.append(observe("screen", capture["slot"], capture["variant"], pixels, [], capture["png_sha256"], audit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "fit", "held", "verify"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    archive, run = args.archive, args.run
    if args.stage == "prepare":
        prepare(archive, run)
        return
    frozen = lock(archive, run)
    frozen_path, points_path, fit_path, public_path = (run / name for name in ("freeze.json", "points.json", "fit.json", "result.json"))
    if args.stage == "freeze":
        source.write(frozen_path, frozen)
        print(sha(frozen_path.read_bytes()), flush=True)
        return
    require(read(frozen_path) == frozen, "The source or candidate changed after freeze")
    if args.stage == "fit":
        fit = measurements(archive, run, frozen, "fit")
        source.write(fit_path, fit)
        source.write(points_path, review.points(fit, frozen))
        return
    points, fit = read(points_path), read(fit_path)
    require(points == review.points(fit, frozen), "The frozen fit decisions changed")
    held = measurements(archive, run, frozen, "held", audit=args.stage == "verify")
    result = {"schema": review.SCHEMA, "freeze": frozen, "freeze_sha256": sha(frozen_path.read_bytes()), "points": points,
              "fit": fit, "held": held, "summary": review.summarize(held, frozen, points)}
    if args.stage == "held":
        source.write(public_path, result)
    else:
        require(fit == measurements(archive, run, frozen, "fit", audit=True) and result == read(public_path),
                "Native reconstruction differs")
        print(f"Verified {len(fit)+len(held)} exact views and source-bound evidence proposals.")
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
