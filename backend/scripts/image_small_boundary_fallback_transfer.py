# SPDX-License-Identifier: Apache-2.0
"""Freeze and evaluate the complementary small-boundary decision on different sources."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from scripts import image_boundary_extent_transfer as method

panel = method.panel
MIN_SHORT_EDGE = 56
MAX_AXIS_FRACTION = .18
ORDINARY = ("original", "contrast", "blur", "global_processing")
POSITIVES = ("sharp", "feathered")
ROOT = Path(__file__).resolve().parents[2]
SOURCE_RECEIPT = ROOT / "docs/evaluation-panels/image-flux-photo-expansion-complete-v285.json"


def flux_receipts(folder):
    public = panel.read(SOURCE_RECEIPT)
    summary = panel.read(folder / "summary.json")
    panel.require(summary["batch_receipts"] == public["batch_receipts"] and summary["source_groups"] == 6,
                  "The generated outputs must match the prior published release")
    records = {}
    for path in sorted(folder.glob("batch-*/authored/output/result.json")):
        for item in panel.read(path)["cases"]:
            panel.require(item["slot"] not in records, "Duplicate generated result")
            records[item["slot"]] = (path.parent, item)
    panel.require(len(records) == 6, "Need all six generated local patches")
    return records


def protocol(args):
    inputs = panel.read(args.flux / "private-inputs.json")["rows"]
    receipts = flux_receipts(args.flux)
    panel.require(len(inputs) == len({row["source_jpeg_sha256"] for row in inputs}) == 6, "Generated groups overlap")
    bindings = {}
    for row in inputs:
        slot, sha = row["slot"], row["source_jpeg_sha256"]
        source = args.source_cache / f"{sha}.bin"
        folder, result = receipts[slot]
        panel.require(panel.sha(source) == sha and panel.sha(folder / slot / "composite.png") == result["composite_png_sha256"],
                      "Generated patch or original source changed")
        bindings[slot] = {"earlier_sha256": sha, "generated_sha256": result["composite_png_sha256"],
                          "raw_changed_pixels": result["changed_pixels"]}
    original = panel.read(args.selection / "result.json")
    panel.require(original["points_sha256"] == panel.sha(args.selection / "points.json")
                  and original["freeze_sha256"] == panel.sha(args.selection / "freeze.json"), "Selection results differ")
    seen = {row["slot"] for row in panel.read(args.selection / "freeze.json")["commons"]["controls"]}
    commons = [row for row in method.commons.binding(args.commons)["sources"] if row["slot"] not in seen]
    panel.require(len(commons) == 25 and not ({row["sha256"] for row in commons}
                  & {row["earlier_sha256"] for row in bindings.values()}), "Expected 25 disjoint Commons slots")
    return {"schema": "chipmunk.image-small-boundary-fallback-transfer/1", "code_sha256": panel.sha(Path(__file__)),
            "selection_sha256": panel.sha(args.selection / "result.json"), "prior_method_sha256": panel.sha(Path(method.__file__)),
            "points_sha256": original["points_sha256"], "commons": commons, "flux": bindings,
            "generated_source_plan_sha256": panel.sha(args.flux / "private-inputs.json"),
            "generated_public_receipt_sha256": panel.sha(SOURCE_RECEIPT), "min_selected_side": MIN_SHORT_EDGE,
            "max_axis_fraction": MAX_AXIS_FRACTION,
            "gate": "At least 12 new Commons sharp and two new native generated sharp locations. At most two new untouched source flags and four total new ordinary flags, with every fixed-size incumbent location preserved."}


def fallback(record, shape):
    if record["counts"]["fixed"][0]:
        return record["counts"]["fixed"]
    x, y, right, bottom = record["proposals"]["distributed"]["box"]
    w, h = right - x, bottom - y
    reported, target, overlap = record["counts"]["distributed"]
    if MIN_SHORT_EDGE <= w <= shape[1] * MAX_AXIS_FRACTION and MIN_SHORT_EDGE <= h <= shape[0] * MAX_AXIS_FRACTION:
        return [reported, target, overlap]
    return [0, target, 0]


def views(original, replacement, box):
    blurred = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(1.1)))
    toned = np.asarray(ImageEnhance.Contrast(Image.fromarray(original)).enhance(1.12))
    ordered = (original, panel.blend.local(original, toned, box), panel.blend.local(original, blurred, box), toned,
               replacement, panel.blend.local(original, replacement, box, feather=True))
    for role, pixels in zip((*ORDINARY, *POSITIVES), ordered, strict=True):
        target = np.any(pixels != original, axis=2) if role in POSITIVES else np.zeros(original.shape[:2], dtype=bool)
        panel.require(role not in POSITIVES or target.any(), "Missing known edit pixels")
        yield role, pixels, target


def commons_inputs(args, lock):
    def rgb(path):
        with Image.open(path) as image, ImageOps.exif_transpose(image) as native:
            return np.array(ImageOps.fit(native.convert("RGB"), (704, 544), Image.Resampling.LANCZOS))

    roster = lock["commons"]
    for index, item in enumerate(roster):
        slot = item["slot"]
        original = rgb(args.commons / item["file"])
        donor = rgb(args.commons / roster[(index + 11) % len(roster)]["file"])
        left, top = 121 + (slot * 43) % 325, 101 + (slot * 29) % 229
        width, height = (65, 94) if slot % 2 else (94, 63)
        right, bottom = left + width, top + height
        replacement = original.copy()
        replacement[top:bottom, left:right] = donor[165:165+height, 280:280+width]
        yield "commons", slot, None, views(original, replacement, (left, top, right, bottom))


def flux_inputs(args):
    outputs = flux_receipts(args.flux)
    for index, row in enumerate(panel.read(args.flux / "private-inputs.json")["rows"]):
        slot, sha = row["slot"], row["source_jpeg_sha256"]
        with Image.open(args.source_cache / f"{sha}.bin") as source:
            panel.require(source.getexif().get(274, 1) == 1 and min(source.size) >= 512, "Original is outside native scope")
            w, h = min(768, source.width), min(768, source.height)
            left, top = (source.width - w) // 2, (source.height - h) // 2
            native = np.array(source.convert("RGB").crop((left, top, left + w, top + h)))
        x, y, right, bottom = row["crop_ltrb_exclusive"]
        x, y, right, bottom = x - left, y - top, right - left, bottom - top
        with Image.open(args.flux / slot / "source.png") as source:
            panel.require(np.array_equal(native[y:bottom, x:right], np.asarray(source)), "Exact generated input crop differs")
        folder, expected = outputs[slot]
        with Image.open(folder / slot / "composite.png") as composite:
            edited = native.copy()
            edited[y:bottom, x:right] = np.asarray(composite)
        a, b, c, d = row["mask_ltrb_exclusive"]
        box = x + a, y + b, x + c, y + d
        panel.require(int(np.any(edited != native, axis=2).sum()) == expected["changed_pixels"], "Exact generated truth differs")
        yield "flux", index, "flux", views(native, edited, box)


def summary(rows):
    counts = defaultdict(Counter)
    extra_untouched = set()
    for row in rows:
        target = counts[(row["cohort"], row["role"])]
        fixed, combined = row["fixed"], row["combined"]
        target.update({"views": 1, "old_flags": bool(fixed[0]), "new_flags": bool(combined[0]),
                       "old_useful": method.useful(fixed), "new_useful": method.useful(combined),
                       "new_locations": method.useful(combined) and not method.useful(fixed),
                       "extra_flags": bool(combined[0]) and not fixed[0]})
        if row["role"] == "original" and combined[0] and not fixed[0]:
            extra_untouched.add((row["cohort"], row["slot"]))
    extra = sum(cell["extra_flags"] for (_, role), cell in counts.items() if role in ORDINARY)
    return {"counts": {f"{cohort}/{role}": value for (cohort, role), value in counts.items()}, "extra_ordinary": extra,
            "extra_untouched_groups": len(extra_untouched),
            "advances": counts["commons", "sharp"]["new_locations"] >= 12
            and counts["flux", "sharp"]["new_locations"] >= 2 and extra <= 4 and len(extra_untouched) <= 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "verify"))
    for name in ("selection", "commons", "flux", "source_cache", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = protocol(args)
    path = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(path, lock)
        return
    panel.require(panel.read(path) == lock, "Prospective source or method freeze changed")
    points = panel.read(args.selection / "points.json")
    start = time.monotonic()
    rows = []
    for cohort, slot, editor, inputs in chain(commons_inputs(args, lock), flux_inputs(args)):
        for role, pixels, truth in inputs:
            record = method.observe(cohort, slot, editor, role, pixels, truth, points)
            result = fallback(record, truth.shape)
            fixed = record["counts"]["fixed"]
            panel.require(not fixed[0] or fixed == result, "An incumbent finding was changed")
            rows.append({"cohort": cohort, "slot": slot, "role": role, "input_sha256": record["input_sha256"],
                         "truth_sha256": record["truth_sha256"], "fixed": fixed, "combined": result,
                         "box": record["proposals"]["distributed"]["box"], "score": record["proposals"]["distributed"]["score"]})
    panel.require(len(rows) == 186, "Missing held views")
    result = summary(rows) | {"freeze_sha256": panel.sha(path), "views": len(rows)}
    journal, aggregate = (args.output / name for name in ("rows.json", "result.json"))
    if args.phase == "run":
        panel.write(journal, rows)
        panel.write(aggregate, result)
    else:
        panel.require(panel.read(journal) == rows and panel.read(aggregate) == result, "Exact prospective replay differs")
        panel.write(args.output / "audit.json", {"verified_views": len(rows), "result_sha256": panel.sha(aggregate)})
    print(result, round(time.monotonic() - start, 2), flush=True)


if __name__ == "__main__":
    main()
