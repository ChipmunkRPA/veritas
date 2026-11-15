# SPDX-License-Identifier: MIT
"""Measure native WebP crop registration and localized differences."""

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_reference_resized_webp_transfer as parent
from scripts import image_reference_webp_residual_holdout as photos

SCHEMA = "chipmunk.image-reference-cropped-webp-transfer/1"
PROFILES = {"development": {"asymmetric": (9, 11, .28, .73, 77, 2)},
            "held": {"tight_corner": (13, 12, .09, .88, 67, 4), "one_axis": (8, 0, .82, 0, 89, 6)}}
EDITS = {"sharp", "feathered", "small_sharp", "small_feathered"}
ORDINARY = {"crop_only", "prior_webp", "prior_jpeg"}
INCOMPATIBLE = {"stretched", "mirrored", "unrelated"}


def cases(args, split, *, identity=None):
    import numpy as np

    from scripts.image_residual_encoder_transfer import authored

    bank = list(parent.source_bank(args, split))
    for index, (family, group, source, original, edited) in enumerate(bank):
        if identity is not None and source != identity:
            continue
        ref = photos.encode(original, "PNG")
        w, h = original.shape[1::-1]
        for profile, (width_loss, height_loss, left_share, top_share, quality, effort) in PROFILES[split].items():
            dx, dy = round(w * width_loss / 100), round(h * height_loss / 100)
            x, y = round(dx * left_share), round(dy * top_share)
            sw, sh = w - dx, h - dy
            cropped = original[y:y+sh, x:x+sw]
            changed = edited[y:y+sh, x:x+sw]
            _, variants = authored({"original": cropped, "edited": changed}, np.any(cropped != changed, axis=2), source)
            foreign = bank[(index + len(bank) // 2) % len(bank)][3]
            variants.update(
                crop_only=cropped,
                prior_webp=photos.image_array(photos.encode(cropped, "WEBP", quality=60, method=4), "WEBP"),
                prior_jpeg=photos.image_array(photos.encode(cropped, "JPEG", quality=78, subsampling=2), "JPEG"),
                stretched=parent.resize(original, (sw, sh), "bilinear"),
                mirrored=cropped[:, ::-1], unrelated=parent.resize(foreign, (sw, sh), "area"),
                small_sharp=parent.small_edit(cropped, variants["sharp"], feathered=False),
                small_feathered=parent.small_edit(cropped, variants["sharp"], feathered=True),
            )
            for role in sorted(EDITS | ORDINARY | INCOMPATIBLE):
                pixels = variants[role]
                mask = np.any(cropped != pixels, axis=2) if role in EDITS else np.zeros((sh, sw), dtype=bool)
                body = photos.encode(pixels, "WEBP", quality=quality, method=effort)
                yield {"source": source, "group": group, "family": family, "profile": profile, "role": role,
                       "reference": photos.sha(ref), "submitted": photos.sha(body),
                       "crop": None if role in INCOMPATIBLE else [x, y, sw, sh], "changed": int(mask.sum()),
                       "truth": photos.sha(mask.tobytes())}, ref, body, mask


def review(receipt):
    split = receipt["split"]
    expected = {(profile, role) for profile in PROFILES[split] for role in EDITS | ORDINARY | INCOMPATIBLE}
    coverage = defaultdict(set)
    counts = defaultdict(Counter)
    for row in receipt["rows"]:
        case = row["case"]
        role, profile = case["role"], case["profile"]
        seen = coverage[case["group"]]
        changed, selected, overlap = case["changed"], row["selected"], row["overlap"]
        photos.require((profile, role) in expected and (profile, role) not in seen
                              and bool(changed) == (role in EDITS) and 0 <= overlap <= min(changed, selected)
                              and (row["registered"] is not None) == bool(row["selection"])
                              and (row["registered"] is not None or selected == overlap == 0), "Invalid crop trial")
        seen.add((profile, role))
        correct = row["registered"] is not None and row["registered"] == case["crop"]
        counts[role].update(files=1, baseline_comparable=int(row["baseline"]), exact_crop=int(correct),
                            wrong_crop=int(row["registered"] is not None and not correct), flags=int(bool(selected)),
                            useful=int(correct and photos.useful((selected, overlap), changed)))
    size = 26 if split == "development" else sum(photos.COUNTS.values())
    photos.require(receipt["schema"] == SCHEMA and len(coverage) == size
                          and all(seen == expected for seen in coverage.values()), "Missing source deliveries")
    outcomes = {role: dict(count) for role, count in sorted(counts.items())}
    if "outcomes" in receipt:
        photos.require(receipt["outcomes"] == outcomes, "Reported counts disagree with the recorded rows")
    return outcomes


def run(args):
    import cv2
    from PIL import Image, features

    from app.detectors.image import reference_lossy_webp_crop, reference_lossy_webp_resize

    cv2.setNumThreads(1)
    rows = []
    for case, ref, body, truth in cases(args, args.phase):
        with Image.open(io.BytesIO(ref)) as earlier, Image.open(io.BytesIO(body)) as submitted:
            baseline, _ = reference_lossy_webp_resize.compare(earlier, submitted, ref, body)
            result, mask = reference_lossy_webp_crop.compare(earlier, submitted, ref, body)
        if proof := result.get("lossy_webp_crop"):
            bounds = [proof["reference_crop"][key] for key in ("x", "y", "w", "h")]
        else:
            bounds = None
        rows.append({"case": case, "baseline": baseline["status"] == "comparable", "registered": bounds,
                     "selected": int(mask.sum()) if mask is not None else 0,
                     "overlap": int((mask & truth).sum()) if mask is not None else 0,
                     "selection": photos.sha(mask.tobytes()) if mask is not None else None})
    receipt = {"schema": SCHEMA, "split": args.phase, "libwebp": features.version("webp"), "rows": rows}
    receipt["outcomes"] = review(receipt)
    photos.save(args.path, receipt)
    return receipt["outcomes"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "review"))
    parser.add_argument("path", type=Path)
    parent.arguments(parser)
    args = parser.parse_args()
    results = review(photos.read(args.path)) if args.phase == "review" else run(args)
    print(json.dumps(results, indent=2, sort_keys=True))
