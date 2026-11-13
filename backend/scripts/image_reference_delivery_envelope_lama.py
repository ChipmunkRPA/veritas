# SPDX-License-Identifier: Apache-2.0
"""Challenge the locked JPEG delivery envelope on another inpainting family."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from scripts import image_reference_delivery_envelope as base
from scripts import review_image_reference_delivery_envelope as audit

SCHEMA = "chipmunk.image-reference-delivery-envelope-lama/1"
ROLES = {"feathered", "sharp", "untouched_crop", "prior_jpeg", "contrast", "local_blur", "fractional", "different_photo"}
GATE = {"minimum_useful_feathered": 4, "maximum_nuisance_highlights": 1, "maximum_incompatible_or_wrong_crops": 0}
COUNT = 8
read, write, digest, pin, need = base.read, base.write, base.digest, base.pin, base.need


def qualified(args):
    prior = args.primary_run
    plan = read(prior / "freeze.json")
    need(plan["schema"] == base.SCHEMA and pin(Path(base.__file__)) == plan["code"][Path(base.__file__).name],
         "Primary method freeze changed")
    earlier = read(args.development / "plan.json")["cases"]
    flux, occupied = base.prior._licensed_flux({"cases": earlier})
    base.prior._public_controls(args.public_photos, occupied)
    occupied = {name.casefold() for name in occupied}
    occupied.update(item["author"].casefold() for item in base.prior.public.roster()[:6])
    ids = {str(row["flickr_photo_id"]) for row in earlier}
    qualified_path = args.lama / "qualified-private.json"
    need(pin(qualified_path) == read(args.lama / "summary-v225.json")["qualified_private_sha256"], "LaMa publisher freeze changed")
    roster = read(qualified_path)
    choices, used_ids = [], set()
    for row in roster["qualified"]:
        source = row["source"]
        photo, owner = str(source["flickr_photo_id"]), source["flickr_owner"].casefold()
        author = source["photo_attribution"]["Author"].casefold()
        need(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/", "Unexpected photo licence")
        if photo not in ids and photo not in used_ids and owner not in occupied and author not in occupied:
            choices.append(row)
            used_ids.add(photo)
            occupied.update((owner, author))
    choices.sort(key=lambda row: digest(f"delivery-envelope-lama/1/{row['source']['flickr_photo_id']}".encode()))
    need(len(choices) >= COUNT and len(flux) == 6, "Insufficient independent recorded groups")
    return Path(roster["cache"]), choices[:COUNT]


def source_pixels(cache, row):
    arrays = []
    for role in ("original", "edited", "semantic_mask"):
        asset = row["measurement"]["files"][role]
        data = (cache / asset["file"]).read_bytes()
        need(digest(data) == asset["sha256"], "Publisher original, edit or mask differs")
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            arrays.append(np.array(image.convert("L" if role == "semantic_mask" else "RGB")))
    original, edited, semantic = arrays
    exact = np.any(original != edited, axis=2)
    truth_path = cache / row["truth_file"]
    need(pin(truth_path) == row["truth_png_sha256"], "Publisher-derived mask bytes differ")
    with Image.open(truth_path, formats=["PNG"]) as mask:
        need(np.array_equal(exact, np.asarray(mask.convert("L")) > 0) and not (exact & (semantic == 0)).any(),
             "Exact changed pixels disagree with the publisher mask")
    yy, xx = np.where(exact)
    return original, edited, (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))


def inputs(args):
    cache, sources = qualified(args)
    native = [source_pixels(cache, row) for row in sources]
    for index, (original, edited, box) in enumerate(native):
        before = base.prior.prior.jpeg.jpeg(original, base.prior.REFERENCE)
        reference = base.prior.prior.jpeg.pixels(before)
        donor = native[(index + 1) % len(native)][0]
        rows = list(base.prior.prior.prior.cases(index, {"original": original, "edited": edited}, box, donor))
        unchanged = next(view for role, view, _, _ in rows if role == "untouched_crop")
        source = sources[index]["measurement"]["files"]["original"]["sha256"]
        for role, submitted, _, crop in rows:
            if role not in ROLES:
                continue
            after = base.prior.prior.image_bytes(submitted)
            truth = (np.any(submitted != unchanged, axis=2) if role in {"feathered", "sharp", "local_blur"}
                     else np.zeros(submitted.shape[:2], dtype=bool))
            row = {"cohort": "lama", "slot": index, "role": role, "crop": crop, "earlier_sha256": digest(before),
                   "source_sha256": source, "submitted_sha256": digest(after), "truth_sha256": digest(truth.tobytes()),
                   "truth_pixels": int(truth.sum())}
            yield row, before, reference, after, submitted, truth


def freeze(args):
    primary = args.primary_run
    selection = read(primary / "development.json")
    primary_freeze = pin(primary / "freeze.json")
    need(selection["freeze_sha256"] == primary_freeze and selection["selected"] == [1.6, 6.0], "Development point changed")
    return {"schema": SCHEMA, "point": selection["selected"], "primary_freeze_sha256": primary_freeze,
            "primary_development_sha256": pin(primary / "development.json"), "method_sha256": pin(Path(base.__file__)),
            "challenge_sha256": pin(Path(__file__)), "source_groups": COUNT, "gate": GATE,
            "rows": [row for row, *_ in inputs(args)]}


def measure(args, frozen):
    with TemporaryDirectory(dir=args.run) as folder:
        rows = []
        for item, expected in zip(inputs(args), frozen["rows"], strict=True):
            rows.append(base.compare_case(Path(folder), item, expected, frozen["point"]))
            if len(rows) % len(ROLES) == 0:
                print(f"{len(rows)}/{len(frozen['rows'])} pairs checked", flush=True)
    point = audit.point_id(frozen["point"])
    need(len(rows) == COUNT * len(ROLES) and all(count == COUNT for count in Counter(row["role"] for row in rows).values()),
         "Source role denominator changed")
    for row in rows:
        for method in ("current", "raw", "fixed_ten"):
            audit.metrics(row, row["responses"][method])
        audit.metrics(row, row["responses"]["envelope"][point])
    independent = audit.evaluate(rows, point, GATE)
    measured = base.summary(rows, point)
    need(all(independent[k] == measured[k] for k in ("counts", "strata", "native_overlays_verified", "native_source_crops_verified")),
         "Independent challenge accounting differs")
    return {"schema": SCHEMA, "freeze_sha256": pin(args.run / "freeze.json"), "point": frozen["point"],
            "summary": independent, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("primary_run", "development", "public_photos", "lama", "run"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    path = args.run / "freeze.json"
    frozen = freeze(args)
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(path, frozen)
        print(pin(path), flush=True)
        return
    need(read(path) == frozen, "Challenge sources or measurement changed after freeze")
    result = measure(args, frozen)
    outcome = args.run / "held.json"
    if args.stage == "review":
        need(read(outcome) == result, "Challenge replay differs")
        write(args.run / "review.json", {"verified": True, "held_sha256": pin(outcome), "summary": result["summary"]})
    else:
        write(outcome, result)
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "strata"}), flush=True)


if __name__ == "__main__":
    main()
