# SPDX-License-Identifier: MIT
"""Measure local differences after both crop and scale in WebP deliveries."""

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_reference_resized_webp_transfer as parent
from scripts import image_reference_webp_residual_holdout as photos

SCHEMA = "chipmunk.image-reference-crop-resize-webp-transfer/1"
PROFILES = {
    "development": {"bicubic76": (7, 10, .35, .27, 89, "bicubic", 76, 3)},
    "held": {"lanczos70": (9, 11, .83, .14, 91, "lanczos", 70, 5), "area89": (12, 5, .25, .81, 88, "area", 89, 0)},
}
EDITS = {"sharp", "feathered", "small_sharp", "small_feathered"}
UNCHANGED = {"crop_resize_only", "prior_webp", "prior_jpeg"}
SCOPE = {"mirrored", "stretched", "unrelated"}
ORDINARY = {"local_blur", "local_tone"}
ROLES = EDITS | UNCHANGED | SCOPE | ORDINARY


def deliveries(args, split, *, identity=None):
    import numpy as np
    from scripts.image_residual_encoder_transfer import authored

    bank = list(parent.source_bank(args, split))
    for index, (family, group, source, original, edited) in enumerate(bank):
        if identity is not None and source != identity:
            continue
        before = photos.encode(original, "PNG")
        rw, rh = original.shape[1::-1]
        foreign = bank[(index + len(bank)//2) % len(bank)][3]
        for profile, (width_loss, height_loss, xshare, yshare, scale, filter_name, quality, encoder) in PROFILES[split].items():
            dx, dy = round(width_loss * rw / 100), round(height_loss * rh / 100)
            w, h = rw - dx, rh - dy
            x, y = round(dx * xshare), round(dy * yshare)
            crop, edit = (array[y:y+h, x:x+w] for array in (original, edited))
            crop_truth = np.any(crop != edit, axis=2)
            if crop_truth.any():
                ordinary, changes = authored({"original": crop, "edited": edit}, crop_truth, source,
                                             allow_invisible=True)
            else:
                ordinary, _ = authored({"original": crop}, None, source)
                changes = {"sharp": crop, "feathered": crop}
            dims = (round(scale * w / 100), round(scale * h / 100))
            projected = parent.resize(crop, dims, filter_name)
            variants = {**{key: changes[key] for key in ("sharp", "feathered")},
                        **{key: ordinary[key] for key in ORDINARY}, "crop_resize_only": crop,
                        "small_sharp": parent.small_edit(crop, changes["sharp"], feathered=False),
                        "small_feathered": parent.small_edit(crop, changes["sharp"], feathered=True),
                        "prior_webp": photos.image_array(photos.encode(crop, "WEBP", quality=62, method=4), "WEBP"),
                        "prior_jpeg": photos.image_array(photos.encode(crop, "JPEG", quality=79, subsampling=2), "JPEG"),
                        "mirrored": crop[:, ::-1], "unrelated": foreign}
            for role in sorted(ROLES):
                if role == "stretched":
                    pixels = parent.resize(crop, (dims[0] + 13, dims[1]), filter_name)
                else:
                    pixels = parent.resize(variants[role], dims, filter_name)
                truth = np.any(projected != pixels, axis=2) if role in EDITS else np.zeros(pixels.shape[:2], bool)
                carrier = photos.encode(pixels, "WEBP", quality=quality, method=encoder)
                yield ({"source": source, "group": group, "family": family, "profile": profile, "role": role,
                        "crop": None if role in SCOPE else [x, y, w, h], "dimensions": list(pixels.shape[1::-1]),
                        "reference_sha256": photos.sha(before), "submitted_sha256": photos.sha(carrier),
                        "changed": int(truth.sum()), "truth_sha256": photos.sha(truth.tobytes())}, before, carrier, truth)


def review(receipt):
    split = receipt["split"]
    roster = defaultdict(set)
    counts = defaultdict(Counter)
    for row in receipt["rows"]:
        case = row["case"]
        key = case["profile"], case["role"]
        seen = roster[case["group"]]
        registered, selected, overlap = (row[key] for key in ("registered", "selected", "overlap"))
        changed = case["changed"]
        photos.require(key not in seen and 0 <= overlap <= min(selected, changed)
                       and (case["role"] in EDITS or changed == 0)
                       and (registered is not None) == bool(row["selection"])
                       and (registered is not None or selected == 0), "Inconsistent measured crop evidence")
        seen.add(key)
        correct = registered is not None and registered == case["crop"]
        counts[key].update(files=1, baseline_comparable=row["baseline"], exact_crop=correct,
                           wrong_crop=registered is not None and not correct, flags=selected > 0,
                           useful=correct and changed > 0 and photos.useful((selected, overlap), changed))
        if case["role"] in EDITS and changed == 0:
            counts[key].update(invisible_edits=1, invisible_edit_flags=selected > 0)
    expected = {(profile, role) for profile in PROFILES[split] for role in ROLES}
    size = 26 if split == "development" else sum(photos.COUNTS.values())
    photos.require(receipt["schema"] == SCHEMA and len(roster) == size and all(mates == expected for mates in roster.values()),
                   "One or more source groups lack their planned deliveries")
    results = {f"{profile}/{role}": dict(count) for (profile, role), count in sorted(counts.items())}
    if "outcomes" in receipt:
        photos.require(results == receipt["outcomes"], "Recorded totals differ")
    return results


def run(args):
    import cv2
    from PIL import Image, features
    from app.detectors.image import reference_lossy_webp_crop, reference_lossy_webp_crop_resize, reference_lossy_webp_resize

    cv2.setNumThreads(1)
    rows = []
    for case, ref, carrier, truth in deliveries(args, args.phase):
        with Image.open(io.BytesIO(ref)) as earlier, Image.open(io.BytesIO(carrier)) as submitted:
            baseline, _ = reference_lossy_webp_resize.compare(earlier, submitted, ref, carrier)
            if baseline["status"] != "comparable":
                baseline, _ = reference_lossy_webp_crop.compare(earlier, submitted, ref, carrier)
            result, mask = reference_lossy_webp_crop_resize.compare(earlier, submitted, ref, carrier)
        proof = result.get("lossy_webp_crop_resize")
        bounds = [proof["reference_crop"][key] for key in ("x", "y", "w", "h")] if proof else None
        rows.append({"case": case, "baseline": baseline["status"] == "comparable", "registered": bounds,
                     "selected": int(mask.sum()) if mask is not None else 0, "overlap": int((mask & truth).sum()) if mask is not None else 0,
                     "selection": photos.sha(mask.tobytes()) if mask is not None else None})
    receipt = {"schema": SCHEMA, "split": args.phase, "libwebp": features.version("webp"), "rows": rows}
    receipt["outcomes"] = review(receipt)
    photos.save(args.output, receipt)
    return receipt["outcomes"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("development", "held", "review"))
    cli.add_argument("output", type=Path)
    parent.arguments(cli)
    args = cli.parse_args()
    if args.phase != "review" and args.development_sources is None:
        cli.error("--development-sources is required to verify the source split")
    results = review(photos.read(args.output)) if args.phase == "review" else run(args)
    print(json.dumps(results, sort_keys=True, indent=2))
