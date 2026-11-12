"""Test PNG crop-and-resize source binding with exact textured RGB agreement."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.detectors.image import reference_compare, reference_crop_resize, reference_resize
from scripts import image_edge_spread_transfer as panel
from scripts import image_reference_crop_resize_transfer as prior

SCHEMA = "chipmunk.image-png-exact-crop-resize/1"
KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
           "lanczos": Image.Resampling.LANCZOS, "opencv_area": None}
digest = panel.paired.shared.digest
GRID = 5
TILE_SIDE = 20
TILE_MIN_TEXTURE = .12
TILE_GRADIENT_MIN = 6
MIN_TILES = 12
MIN_AXIS = 4
MIN_GLOBAL_EQUAL = .65
MAX_MISMATCH_BOX = .40
TOP_PROPOSALS = 8
MIN_PROPOSAL_CORRELATION = .75
MAX_REFERENCE_SIDE = 768
MAX_SUBMITTED_SIDE = 845
MIN_SCALE, MAX_SCALE = .72, 1.17
FRESH_CASES = 12
FRESH_ROLES = ("untouched_crop", "local_blur", "local_contrast", "contrast", "prior_jpeg", "rotation",
               "fractional", "quarter_pixel", "nonproportional", "different_photo")
FRESH_CONTROL = ("local_blur", "local_contrast")
UNSUPPORTED = (*prior.INCOMPATIBLE, "quarter_pixel")


def _project(reference, rectangle, size, kernel):
    x0, y0, x1, y1 = rectangle
    crop = reference[y0:y1, x0:x1]
    if kernel is None:
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(crop).resize(size, kernel))


def _textured_tiles(submitted):
    height, width = submitted.shape[:2]
    half = TILE_SIDE // 2
    tiles = []
    for row, column in product(range(GRID), repeat=2):
        x = round((column + .5) / GRID * width)
        y = round((row + .5) / GRID * height)
        bounds = np.s_[y-half:y+half, x-half:x+half]
        patch = submitted[bounds].astype(np.int16)
        dx = np.any(np.abs(np.diff(patch, axis=1)) >= TILE_GRADIENT_MIN, axis=2)
        dy = np.any(np.abs(np.diff(patch, axis=0)) >= TILE_GRADIENT_MIN, axis=2)
        if (dx.mean() + dy.mean()) / 2 >= TILE_MIN_TEXTURE:
            tiles.append((row, column, bounds))
    return tiles


def _spread(cells):
    return (len(cells) >= MIN_TILES and len({r for r, _ in cells}) >= MIN_AXIS
            and len({c for _, c in cells}) >= MIN_AXIS
            and all(any((r < GRID // 2 if top else r > GRID // 2)
                        and (c < GRID // 2 if left else c > GRID // 2) for r, c in cells)
                    for top, left in product((False, True), repeat=2)))


def _supported(projected, submitted, tiles):
    equal = np.all(projected == submitted, axis=2)
    if equal.mean() < MIN_GLOBAL_EQUAL:
        return None
    supports = [(r, c) for r, c, bounds in tiles if equal[bounds].all()]
    if not _spread(supports):
        return None
    ys, xs = np.where(~equal)
    if len(xs) and (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min()) > MAX_MISMATCH_BOX * equal.size:
        return None
    return equal, supports


def _scope(rectangle, submitted_shape):
    x0, y0, x1, y1 = rectangle
    widths = (x1 - x0, y1 - y0)
    sizes = submitted_shape[1], submitted_shape[0]
    return widths != sizes and all(round(MIN_SCALE * base) <= measured <= round(MAX_SCALE * base)
                                   for base, measured in zip(widths, sizes, strict=True))


def candidate(reference, submitted):
    """Return a unique exact projection and a native difference mask if corroborated."""
    if (min(*reference.shape[:2], *submitted.shape[:2]) < 160 or max(reference.shape[:2]) > MAX_REFERENCE_SIDE
            or max(submitted.shape[:2]) > MAX_SUBMITTED_SIDE):
        return None, None
    textured = _textured_tiles(submitted)
    if not _spread([(row, column) for row, column, _ in textured]):
        return None, None
    gray_reference, gray_submitted = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (reference, submitted))
    rough = reference_crop_resize._rough_bounds(gray_reference, gray_submitted)
    if rough is None:
        return None, None
    proposals = sorted(reference_crop_resize._scores(gray_reference, gray_submitted, rough[0]), reverse=True)
    good = None
    for rank, (correlation, rectangle, *_features) in enumerate(proposals[:TOP_PROPOSALS], 1):
        if correlation < MIN_PROPOSAL_CORRELATION or not _scope(rectangle, submitted.shape[:2]):
            continue
        for label, kernel in KERNELS.items():
            projected = _project(reference, rectangle, submitted.shape[1::-1], kernel)
            support = _supported(projected, submitted, textured)
            if support is None:
                continue
            equal, cells = support
            if good is not None:
                if rectangle != good["rectangle"] or not np.array_equal(projected, good["projected"]):
                    return None, None
                good["kernels"].append(label)
            else:
                good = {"rectangle": rectangle, "projected": projected, "equal": equal,
                        "support": cells, "proposal_rank": rank, "kernels": [label]}
    if good is None:
        return None, None
    x0, y0, x1, y1 = good["rectangle"]
    difference = ~good["equal"]
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    for region in regions:
        crop = reference_resize.native_crop(region, (x1 - x0, y1 - y0), submitted.shape[1::-1])
        region["reference_region"] = {**crop, "x": crop["x"] + x0, "y": crop["y"] + y0}
    return {"crop": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "kernels": good["kernels"], "exact_tiles": good["support"], "exact_pixels": int(good["equal"].sum()),
            "proposed_rank": good["proposal_rank"], "projected_sha256": digest(good["projected"].tobytes()),
            "regions": regions, "omitted": omitted}, mask


def fresh_sources(commons, excluded):
    roster = panel.read(commons / "plan.json")
    receipts = panel.read(commons / "acquisition.json")
    panel.require(panel.sha(commons / "plan.json") == receipts["roster_sha256"], "Commons roster mismatch")
    excluded_slots = {row["slot"] for row in panel.read(excluded)["calibration_sources"]}
    candidates = []
    for source, receipt in zip(roster["sources"], receipts["attempts"], strict=True):
        panel.require(source["slot"] == receipt["slot"], "Commons receipt has a different slot")
        if receipt["status"] == "qualified" and source["slot"] not in excluded_slots:
            panel.require(source["licence"] in ("CC BY 3.0", "CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0"),
                          "Unsupported source license")
            w, h = source["source_dimensions"]
            if max(w, h) / min(w, h) <= 2.2:
                candidates.append({"slot": source["slot"], "sha256": receipt["sha256"], "file": receipt["name"],
                                   "pageid": source["pageid"], "author": source["author"], "license": source["licence"]})
    candidates.sort(key=lambda row: digest(f'{SCHEMA}/{row["sha256"]}'.encode()))
    selection = candidates[:FRESH_CASES]
    panel.require(len(selection) == FRESH_CASES and all(len({r[key] for r in selection}) == FRESH_CASES
                  for key in ("pageid", "sha256", "author")), "Expected twelve distinct Commons source records")
    for row in selection:
        panel.require(panel.sha(commons / row["file"]) == row["sha256"], "Pinned Commons JPEG differs")
    return selection


def fresh_cases(slot, source, donor):
    h, w = source.shape[:2]
    rect = (round(.09*w), round(.07*h), round(.93*w), round(.90*h))
    x0, y0, x1, y1 = rect
    known = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    scale = (.78, .91, 1.10)[slot % 3]
    size = tuple(round(scale * known[k]) for k in ("w", "h"))
    label = list(KERNELS)[slot % 4]
    kernel = KERNELS[label]

    def resize(image):
        return _project(np.asarray(image), rect, size, kernel)

    base = resize(source)
    photo = Image.fromarray(source)
    contrast = ImageEnhance.Contrast(photo).enhance(1.12)
    blur = photo.filter(ImageFilter.GaussianBlur(2.0))
    patch = (w // 3, h // 3, w // 2, h // 2)
    local_blur = resize(panel.blend.local(source, np.asarray(blur), patch, feather=True))
    local_contrast = resize(panel.blend.local(source, np.asarray(contrast), patch, feather=True))
    prior_jpeg = resize(reference_compare._prior_jpeg(source, 83))
    rotated = resize(photo.rotate(2., resample=Image.Resampling.BICUBIC))
    stretched = _project(source, rect, (round(size[0]*1.06), size[1]), kernel)
    unrelated = np.asarray(Image.fromarray(donor).resize(size, Image.Resampling.BICUBIC))

    def shifted(amount):
        offset = ((1, 0, amount*size[0]/known["w"]), (0, 1, amount*size[1]/known["h"]))
        return cv2.warpAffine(base, np.asarray(offset, float), size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    values = (base, local_blur, local_contrast, resize(contrast), prior_jpeg, rotated, shifted(.5), shifted(.25),
              stretched, unrelated)
    for role, rgb in zip(FRESH_ROLES, values, strict=True):
        truth = np.any(rgb != base, axis=2) if role in FRESH_CONTROL else np.zeros(rgb.shape[:2], dtype=bool)
        yield role, rgb, truth, known if role not in UNSUPPORTED else None, label


def profile(args):
    folders = {"development": args.development, "earlier": args.earlier, "later": args.later}
    original = panel.read(args.prior / "freeze.json")
    for role, folder in folders.items():
        for name, digest in original["intakes"][role].items():
            panel.require(panel.sha(folder / name) == digest, "Exposed source panel changed")
    journals = {role: panel.sha(args.prior / f"{role}.json") for role in folders}
    published = panel.read(panel.ROOT / "docs/evaluation-panels/image-reference-crop-resize-transfer-2026-11-11.json")
    panel.require(panel.sha(args.prior / "freeze.json") == published["freeze_sha256"]
                  and all(journals[name] == value for name, value in published["journal_sha256"].items()),
                  "The exposed panel does not match its public commitments")
    for part in ("development", "confirm"):
        earlier = panel.read(args.prior / f"{part}-summary.json")
        panel.require(earlier["freeze_sha256"] == panel.sha(args.prior / "freeze.json")
                      and all(journals[role] == digest for role, digest in earlier["journal_sha256"].items()),
                      "Earlier measurements differ from their recorded receipts")
    return {"schema": SCHEMA, "exposed_freeze_sha256": panel.sha(args.prior / "freeze.json"),
            "exposed_journals": journals, "source_sha256": panel.sha(Path(__file__)),
            "commons": fresh_sources(args.commons, args.excluded), "excluded_sha256": panel.sha(args.excluded),
            "kernels": list(KERNELS), "proposals": TOP_PROPOSALS, "minimum_global_pixel_agreement": MIN_GLOBAL_EQUAL,
            "texture": [TILE_SIDE, TILE_GRADIENT_MIN, TILE_MIN_TEXTURE], "grid": GRID, "tiles": MIN_TILES,
            "axis": MIN_AXIS, "maximum_mismatch_box": MAX_MISMATCH_BOX,
            "runtime": [cv2.__version__, Image.__version__, np.__version__],
            "advance": "Both held panels have >=4 new feathered locations, >=6 untouched exact crops, no wrong crops, "
                       "and at most 4 nuisance flags across 3 sources. Fresh panel has at least 8 exact untouched crops, "
                       "at least 8 useful local-blur and 8 useful local-contrast locations, and no wrong or incompatible crops."}


def exposed(folder, original, role):
    sources = prior.sources(folder, role)
    recorded = {(r["source"], r["role"]): r for r in panel.read(original / f"{role}.json") if r["format"] == "png"}
    for index, (slot, generator, images, target) in enumerate(sources):
        other = sources[(index + 1) % len(sources)][2]["original"]
        for case, rgb, truth, expected in prior.cases(slot, images, target, other):
            old = recorded.pop((slot, case))
            yield slot, generator, images["original"], case, rgb, truth, expected, old, None
    panel.require(not recorded, "Incomplete exposed input roster")


def fresh(commons, roster):
    photos = []
    for item in roster:
        with Image.open(commons / item["file"]) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            oriented.thumbnail((640, 640), Image.Resampling.LANCZOS)
            photos.append(np.asarray(oriented))
    for ordinal, photo in enumerate(photos):
        for role, pixels, truth, known, kernel in fresh_cases(ordinal, photo, photos[(ordinal + 1) % len(photos)]):
            yield ordinal, "ordinary", photo, role, pixels, truth, known, None, kernel


def measure(views):
    for slot, generator, reference, case, rgb, truth, expected, old, kernel in views:
        body, carrier = panel.paired.encoded(rgb, "png")
        input_sha256 = digest(body)
        target = digest(truth.tobytes())
        if old:
            panel.require((input_sha256, target, expected) == (old["input_sha256"], old["truth_sha256"], old["truth"]),
                          "A frozen PNG delivery or its exact region changed")
        proof, mask = candidate(reference, np.asarray(carrier))
        if mask is None:
            mask = np.zeros_like(truth)
        yield {"slot": slot, "family": generator, "role": case, "exact_input": input_sha256, "truth_sha256": target,
               "authored_crop": expected, "authored_kernel": kernel, "proof": proof, "mask_sha256": digest(mask.tobytes()),
               "measurement": panel.scoring.measure(mask, truth),
               "prior": {key: old[key] for key in ("crop", "candidate", "reference")} if old else None}


def summarize(rows):
    counts = {}
    nuisance = set()
    for row in rows:
        cell = counts.setdefault(row["role"], Counter())
        proof = row["proof"]
        measurement = row["measurement"]
        prior_hit = row["prior"] is not None and panel.useful(row["prior"]["reference"])
        crop = proof["crop"] if proof else None
        if row["role"] in prior.ORDINARY and measurement["reported"]:
            nuisance.add(row["slot"])
        cell.update(planned=1, exact_crop=int(proof is not None and crop == row["authored_crop"]),
                    wrong_crop=int(proof is not None and crop != row["authored_crop"]),
                    localized=int(panel.useful(measurement)), added=int(panel.useful(measurement) and not prior_hit),
                    highlighted=int(bool(measurement["reported"])),
                    exact_kernel=int(proof is not None and row["authored_kernel"] in proof["kernels"]))
    return {"roles": counts, "wrong_or_incompatible": sum(cell["wrong_crop"] for cell in counts.values()),
            "nuisance": sum(cell["highlighted"] for role, cell in counts.items() if role in prior.ORDINARY),
            "nuisance_groups": len(nuisance)}


def qualified(outcomes):
    exposed_pass = all(item["wrong_or_incompatible"] == 0 and item["nuisance"] <= 4 and item["nuisance_groups"] <= 3
                       and item["roles"]["untouched_crop"]["exact_crop"] >= 6
                       and item["roles"]["feathered"]["added"] >= 4 for key, item in outcomes.items()
                       if key in ("earlier", "later"))
    fresh = outcomes["commons"]
    fresh_pass = (fresh["wrong_or_incompatible"] == fresh["nuisance"] == 0
                  and fresh["roles"]["untouched_crop"]["exact_crop"] >= 8
                  and all(fresh["roles"][role]["localized"] >= 8 for role in FRESH_CONTROL))
    return {"exposed_regression_passed": exposed_pass, "fresh_ordinary_passed": fresh_pass, "scope_qualifies": exposed_pass and fresh_pass}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for field in ("development", "earlier", "later", "prior", "commons", "excluded", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    frozen = profile(args)
    lock = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(lock, frozen)
        return
    panel.require(frozen == panel.read(lock), "Frozen source or protocol changed")
    cv2.setNumThreads(1)
    start = time.monotonic()
    inputs = {name: exposed(getattr(args, name), args.prior, name) for name in ("development", "earlier", "later")}
    inputs["commons"] = fresh(args.commons, frozen["commons"])
    outcomes, journals = {}, {}
    for (name, views), expected in zip(inputs.items(), (80, 120, 120, 120), strict=True):
        rows = list(measure(views))
        panel.require(len(rows) == expected, "Incomplete source panel")
        receipt = args.output / f"{name}.json"
        panel.write(receipt, rows)
        outcomes[name], journals[name] = summarize(rows), panel.sha(receipt)
        print(f"{name}: {len(rows)} views", flush=True)
    result = {"schema": SCHEMA, "outcomes": outcomes, "journals": journals, "freeze_sha256": panel.sha(lock),
              "elapsed_s": round(time.monotonic() - start, 3), **qualified(outcomes)}
    panel.write(args.output / "result.json", result)
    print({key: result[key] for key in qualified(outcomes)}, flush=True)


if __name__ == "__main__":
    main()
