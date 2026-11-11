"""Replay integer reference alignment on source-bound crops, shifts and controls."""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_alignment as alignment
from app.detectors.image import reference_compare as compare
from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as metrics
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-reference-alignment-transfer/1"
FORMATS = ("png", "jpeg75")


def _translation(pixels, dx, dy, mode):
    h, w = pixels.shape[:2]
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(pixels, matrix, (w, h), flags=mode)


def _views(slot, native, box, other):
    original, edited = native["original"], native["edited"]
    h, w = original.shape[:2]
    x, y = w // 25 + slot % 5, h // 28 + slot % 7
    crop = np.s_[y:h - h // 25, x:w - w // 28]
    feathered = blend.local(original, edited, box, feather=True)
    for role, image in (("crop_original", original), ("crop_sharp", edited), ("crop_feathered", feathered),
                        ("crop_prior_jpeg", compare._prior_jpeg(original, 83))):
        truth = np.any(image[crop] != original[crop], axis=2) if role.endswith(("sharp", "feathered")) else None
        for fmt in FORMATS:
            yield role, fmt, image[crop], truth, (x, y)

    dx, dy = (7, -11) if slot % 2 else (-9, 5)
    for role, image in (("shift_original", original), ("shift_sharp", edited)):
        truth = None if role == "shift_original" else _translation(
            np.any(edited != original, axis=2).astype(np.uint8), dx, dy, cv2.INTER_NEAREST).astype(bool)
        yield role, "png", _translation(image, dx, dy, cv2.INTER_NEAREST), truth, (-dx, -dy)
    yield "partial_shift_jpeg", "jpeg75", _translation(original, dx, dy, cv2.INTER_NEAREST), None, (-dx, -dy)
    yield "fractional_shift", "png", _translation(original, dx + .5, dy + .5, cv2.INTER_LINEAR), None, None
    rotated = cv2.warpAffine(original, cv2.getRotationMatrix2D((w / 2, h / 2), .7, 1.), (w, h))
    yield "rotation", "png", rotated, None, None
    resized = cv2.resize(original, (round(w * .94), round(h * .94)), interpolation=cv2.INTER_AREA)
    yield "resize", "png", resized, None, None
    yield "different_photo", "png", other, None, None


def _counts(rows):
    counts = Counter()
    for row in rows:
        measured = row["new"]
        key = f"{row['format']}/{row['role']}"
        counts[key + "/views"] += 1
        counts[key + "/old_comparable"] += row["old_status"] == "comparable"
        counts[key + "/new_comparable"] += row["new_status"] == "comparable"
        counts[key + "/exact_offsets"] += row["known_offset"] is not None and row["new_offset"] == row["known_offset"]
        counts[key + "/wrong_offsets"] += row["new_offset"] is not None and row["new_offset"] != row["known_offset"]
        counts[key + "/displayed"] += measured["reported"] > 0
        if measured["target"]:
            overlap, target = measured["overlap"], measured["target"]
            union = measured["reported"] + target - overlap
            counts[key + "/useful"] += Fraction(overlap, union) >= Fraction(3, 10) and Fraction(overlap, target) >= Fraction(1, 2)
    return dict(sorted(counts.items()))


def _measurements(intake):
    source = list(paired.inputs(intake))
    for i, (row, native, _) in enumerate(source):
        original = native["original"]
        digest = shared.digest(original.tobytes())
        other = source[(i + 1) % len(source)][1]["original"]
        for role, fmt, pixels, truth, expected in _views(row["slot"], native, row["box"], other):
            body, image = paired.encoded(pixels, fmt)
            submitted = np.asarray(image)
            if truth is None:
                truth = np.zeros(submitted.shape[:2], bool)
            with Image.open(io.BytesIO(body)) as carrier:
                old, _ = compare.compare_across_deliveries(original, submitted, carrier)
                new, mask = alignment.compare(original, submitted, carrier)
            offset = new.get("alignment", {}).get("reference_minus_submitted")
            assert sum(region["selected_pixels"] for region in new["regions"]) == (int(mask.sum()) if mask is not None else 0)
            if mask is not None and offset is not None:
                a = new["alignment"]["submitted_overlap"]
                box = np.s_[a["y"]:a["y"] + a["h"], a["x"]:a["x"] + a["w"]]
                assert mask[box].sum() == mask.sum()
                for region in new["regions"]:
                    mapped = region["reference_region"]
                    assert mapped == {"x": region["x"] + offset["x"], "y": region["y"] + offset["y"],
                                      "w": region["w"], "h": region["h"]}
            measure = metrics.measure(mask if mask is not None else np.zeros_like(truth), truth)
            yield {"slot": row["slot"], "format": fmt, "role": role, "original_rgb_sha256": digest,
                   "submitted_sha256": shared.digest(body), "truth_sha256": shared.digest(truth.tobytes()),
                   "old_status": old["status"], "new_status": new["status"], "known_offset": list(expected) if expected else None,
                   "new_offset": [offset["x"], offset["y"]] if offset is not None else None, "new": measure,
                   "mask_sha256": shared.digest(mask.tobytes()) if mask is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("intake", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.intake / "plan.json").read_bytes())
    assert len({row["flickr_photo_id"] for row in plan["cases"]}) == len(plan["cases"]) == 12
    files = (Path(__file__), Path(alignment.__file__), Path(compare.__file__), Path(paired.__file__), Path(blend.__file__))
    freeze = {"schema": SCHEMA, "code_sha256": {p.name: shared.file_digest(p) for p in files},
              "source_plan_sha256": shared.file_digest(args.intake / "plan.json"),
              "source_intake_sha256": shared.file_digest(args.intake / "result.json"),
              "image_groups": 12, "rows_expected": 180, "opencv": cv2.__version__, "minimum_iou": .30, "minimum_recall": .50}
    path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        shared.write(path, freeze)
        return
    assert freeze == json.loads(path.read_bytes())
    cv2.setNumThreads(1)
    started = time.monotonic()
    rows = list(_measurements(args.intake))
    assert len(rows) == freeze["rows_expected"] and len({r["slot"] for r in rows}) == 12
    journal = args.output / "rows.json"
    shared.write(journal, rows)
    shared.write(args.output / "result.json", {"schema": SCHEMA, "freeze_sha256": shared.file_digest(path),
                 "rows_sha256": shared.file_digest(journal), "counts": _counts(rows),
                 "views": len(rows), "elapsed_s": round(time.monotonic() - started, 3)})


if __name__ == "__main__":
    main()
