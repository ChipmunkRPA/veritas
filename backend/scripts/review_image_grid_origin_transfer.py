"""Independently verify the shifted image inputs and native mask arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from scipy import ndimage

from scripts import image_edge_spread_transfer as panel
from scripts import image_flux_product_comparison_v288 as flux
from scripts.review_image_regional_flip_transfer import measured, prepared_digest, restore_masks, useful

BASE = ("native", "flip", "recompression")
OFFSETS = {"horizontal_offset": (0, 32), "vertical_offset": (32, 0)}
RAW = (*BASE, *OFFSETS)
ALL = (*RAW, "offset_intersection", "original_supported_by_both", "original_supported_by_either")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def native_support(original, corroboration):
    labels, number = ndimage.label(original, structure=np.ones((3, 3), dtype=bool))
    retained = np.zeros_like(original)
    for value, bounds in enumerate(ndimage.find_objects(labels, max_label=number), 1):
        part = labels[bounds] == value
        if 2 * np.count_nonzero(part & corroboration[bounds]) >= np.count_nonzero(part):
            retained[bounds] |= part
    return retained


def padded(photo, offset):
    """Recreate the reflected prefix with exact source pixels and explicit indices."""
    dy, dx = offset
    values = np.asarray(photo)
    yy = np.r_[np.arange(dy, 0, -1), np.arange(photo.height)]
    xx = np.r_[np.arange(dx, 0, -1), np.arange(photo.width)]
    return Image.fromarray(values[yy[:, None], xx[None, :]])


def unpack(body, row):
    pixels = math.prod(row["shape"])
    size = (pixels + 7) // 8
    require(digest(body) == row["mask_sha256"] and len(body) == size * len(RAW), "Stored mask bytes differ")
    values = np.unpackbits(np.frombuffer(body, np.uint8).reshape(len(RAW), size), axis=1)
    require(not values[:, pixels:].any(), "Mask packing has invalid trailing bits")
    saved = dict(zip(RAW, values[:, :pixels].reshape(len(RAW), *row["shape"]).astype(bool), strict=True))
    x, y = (saved[name] for name in OFFSETS)
    intersection = x & y
    return {**saved, "offset_intersection": intersection,
            "original_supported_by_both": native_support(saved["native"], intersection),
            "original_supported_by_either": native_support(saved["native"], x | y)}


def deliveries(earlier, later):
    for name, folder in (("earlier", earlier), ("later", later)):
        for item, views, _ in panel.paired.inputs(folder):
            original = Image.fromarray(views["original"])
            for role, pixels in panel.native_variants(views, item["box"]).items():
                body, photo = panel.paired.encoded(pixels, "png")
                target = None
                if role in {"edited", "feathered_edit"}:
                    r, g, b = ImageChops.difference(original, photo).split()
                    target = ImageChops.lighter(ImageChops.lighter(r, g), b)
                yield name, item["slot"], role, body, photo, target
    cases, items = flux.source_cases()
    for slot in range(len(cases)):
        images, _ = flux.material(cases, items, slot)
        for role, image in images.items():
            body, photo = flux.previous.encode(image, 0)
            target = None
            if role == "ai":
                r, g, b = ImageChops.difference(images["original"], photo).split()
                target = ImageChops.lighter(ImageChops.lighter(r, g), b)
            yield "flux", slot, role, body, photo, target


def count(rows):
    output = {}
    for cohort, group in rows.items():
        positives = [row for row in group if row["role"] in {"edited", "feathered_edit", "ai"}]
        negatives = [row for row in group if row not in positives]
        roles = sorted({row["role"] for row in negatives})
        arms = {}
        for arm in ALL:
            hits = [row for row in positives if useful(row["measurements"][arm])]
            false = [row for row in negatives if row["measurements"][arm]["reported"]]
            arms[arm] = {"useful": len(hits), "missed": len(positives) - len(hits), "controls_flagged": len(false),
                         "controls_flagged_by_type": {role: sum(row["role"] == role for row in false) for role in roles},
                         "useful_lost_vs_flip": sum(row not in hits and useful(row["measurements"]["flip"]) for row in positives),
                         "additional_useful_groups_vs_flip": len({row["slot"] for row in hits if not useful(row["measurements"]["flip"])})}
        output[cohort] = {"positive": len(positives), "controls": len(negatives), "arms": arms}
    return output


def audit(args):
    run = args.run
    freeze = panel.read(run / "freeze.json")
    result = panel.read(run / "result.json")
    journal = (run / "rows.jsonl").read_bytes()
    require(digest(journal) == result["journal_sha256"] and panel.sha(run / "freeze.json") == result["freeze_sha256"]
            and panel.sha(args.prior / "rows.json") == freeze["previous_flip_journal_sha256"]
            and panel.sha(flux.V285_RESULT) == freeze["flux_source_receipt_sha256"]
            and freeze["method"]["offsets_yx"] == [list(offset) for offset in OFFSETS.values()]
            and freeze["method"]["arms"] == list(ALL), "Published study bindings differ")
    for name, folder in (("earlier", args.earlier), ("later", args.later)):
        require(all(panel.sha(folder / file) == sha for file, sha in freeze["intakes"][name].items()),
                "A frozen source intake changed")
    rows = [json.loads(line) for line in journal.splitlines()]
    pending = {(row["cohort"], row["slot"], row["role"]): row for row in rows}
    require(len(pending) == len(rows) == sum(freeze["source_views"]) == 222, "Duplicate or missing views")
    prior = {(row["panel"], row["slot"], row["role"]): row for row in panel.read(args.prior / "rows.json")}
    checked = {name: [] for name in freeze["cohorts"]}
    matches = 0
    for cohort, slot, role, body, photo, target in deliveries(args.earlier, args.later):
        row = pending.pop((cohort, slot, role))
        truth = np.asarray(target) != 0 if target is not None else np.zeros((photo.height, photo.width), bool)
        require(row["shape"] == list(truth.shape) and digest(body) == row["input_sha256"]
                and digest(truth.tobytes()) == row["truth_sha256"], "Native source or target pixels differ")
        for offset, recorded in zip(OFFSETS.values(), row["offset_commitments"], strict=True):
            shifted = padded(photo, offset)
            require(shifted.crop((offset[1], offset[0], shifted.width, shifted.height)).tobytes() == photo.tobytes()
                    and list(offset) == recorded["offset_yx"]
                    and digest(shifted.tobytes()) == recorded["padded_rgb_sha256"]
                    and prepared_digest(shifted) == recorded["prepared_float32_sha256"], "Shifted inference input differs")
        masks = unpack((run / "masks" / f"{cohort}-{slot:02d}-{role}.bin").read_bytes(), row)
        require({arm: measured(mask, truth) for arm, mask in masks.items()} == row["measurements"], "Native overlap mismatch")
        if cohort != "flux":
            old = restore_masks(args.prior / "masks" / f"{cohort}-{slot:02d}-{role}.bin", prior.pop((cohort, slot, role)))
            require(all(np.array_equal(masks[name], old["supported_components" if name == "flip" else name]) for name in BASE),
                    "Original mask differs from the archived comparison")
            matches += 3
        checked[cohort].append(row)
    require(not pending and not prior and count(checked) == result["counts"] and result["inference_calls"] == 534,
            "Study readout or coverage did not reproduce")
    comparable = [result["counts"][cohort]["arms"] for cohort in ("earlier", "later")]
    filter_pass = all(a["original_supported_by_both"]["useful_lost_vs_flip"] == 0
                      and 5 * a["original_supported_by_both"]["controls_flagged"] <= 4 * a["flip"]["controls_flagged"]
                      for a in comparable)
    discovery_pass = all(a["offset_intersection"]["additional_useful_groups_vs_flip"] >= 2
                         and a["offset_intersection"]["controls_flagged"] <= a["flip"]["controls_flagged"] for a in comparable)
    return {"source_views": len(rows), "shifted_inputs_verified": 2 * len(rows), "native_measurements_verified": len(ALL) * len(rows),
            "earlier_mask_matches": matches, "advancement": {"filter": filter_pass, "discovery": discovery_pass}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "earlier", "later", "prior"):
        parser.add_argument(key, type=Path)
    print(json.dumps(audit(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
