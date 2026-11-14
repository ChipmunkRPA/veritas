# SPDX-License-Identifier: Apache-2.0
"""Compare exact-grid JPEG detail loss at standard qualities 75 and 85."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

QUALITIES = (75, 85)
POSITIVE = ("weak_blur", "moderate_blur", "strong_blur")
ORDINARY = ("unchanged", "tandem", "global_blur", "global_tone", "local_sharpen", "different_photo",
            "shifted", "wrong_quality", "generated_feather")
SCHEMA = "chipmunk.image-jpeg-detail-quality-transfer/1"
DESIGN = {
    "scope": "Same framing and standard quality in both upright baseline 4:2:0 RGB JPEGs. Reuse the current JPEG90 exact-grid anchors and texture loss rule. Do not tune points per delivery.",
    "inputs": "Every eligible photo from the previous 34-source confirmation bank. Prior project exposure; new Q75 and Q85 delivery recipes. Freeze all source, delivery and exact pre-encode truth commitments before measuring.",
    "comparison": "Full earlier-copy routing with all supported comparison options. Count an added result only when the incumbent has no region.",
    "useful": "At least 0.30 native-pixel intersection over union and 0.50 edited-pixel recall.",
    "advance_each": {"additional_useful_moderate_or_strong_sources": 8, "max_added_nuisance_deliveries": 2,
                     "max_added_unchanged_global_or_incompatible": 0},
    "confidence_weight": 0,
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, record):
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, allow_nan=False, separators=(",", ":"))
        file.write("\n")


def inputs(sources):
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    from scripts import image_residual_encoder_transfer as delivery

    for index, (_, row, views, support, _) in enumerate(sources):
        source = row["original"]["sha256"]
        original = views["original"]
        height, width = support.shape
        photo = Image.fromarray(original)
        sigma = (.7, 1.35, 2.1)
        alpha = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = ((1 - np.cos(np.pi * np.minimum(alpha / 8, 1))) / 2)[..., None]

        def local(changed, original=original, alpha=alpha):
            return np.rint(original * (1 - alpha) + np.asarray(changed) * alpha).astype(np.uint8)

        blur = [photo.filter(ImageFilter.GaussianBlur(radius)) for radius in sigma]
        wrong = cv2.resize(sources[(index + 1) % len(sources)][2]["original"], (width, height), interpolation=cv2.INTER_AREA)
        ordinary = {
            "unchanged": original, "global_blur": np.asarray(blur[-1]),
            "global_tone": np.asarray(ImageEnhance.Contrast(photo).enhance(1.11)),
            "local_sharpen": local(photo.filter(ImageFilter.UnsharpMask(1.5, 120, 0))),
            "different_photo": wrong, "shifted": np.roll(original, (3, 6), axis=(0, 1)),
            "wrong_quality": local(blur[1]), "generated_feather": local(views["edited"]),
        }
        positives = {role: local(rgb) for role, rgb in zip(POSITIVE, blur, strict=True)}
        for quality in QUALITIES:
            before = delivery.encode(original, quality)
            variants = positives | ordinary | {"tandem": delivery.decode(before)}
            for role in (*POSITIVE, *ORDINARY):
                pixels = variants[role]
                after = delivery.encode(pixels, 90 if role == "wrong_quality" else quality)
                truth = np.any(pixels != original, axis=-1) if role in POSITIVE else np.zeros_like(support)
                yield {"quality": quality, "source": source, "family": row["generator"], "role": role,
                       "shape": [height, width], "reference_sha256": sha(before), "submitted_sha256": sha(after),
                       "truth_sha256": sha(truth.tobytes()), "changed": int(truth.sum())}, before, after, truth


def measured(case, scratch):
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_jpeg_detail_loss
    from app.evidence import image_reference
    from scripts.image_reference_same_jpeg_transfer import OPTIONS

    key, before, after, truth = case
    old, changed = scratch / "earlier.jpg", scratch / "submitted.jpg"
    old.write_bytes(before)
    changed.write_bytes(after)
    incumbent = image_reference.inspect(old, changed, allow_same_jpeg=True, allow_jpeg_detail_loss=True, **OPTIONS)
    evidence, mask = None, None
    if not incumbent["regions"]:
        with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b:
            evidence, mask = reference_jpeg_detail_loss.compare(a, b, before, after, alternate_quality=key["quality"])
    pixels, overlap = (int(mask.sum()), int((mask & truth).sum())) if mask is not None else (0, 0)
    return key | {"prior_reported": bool(incumbent["regions"]), "added_pixels": pixels, "overlap": overlap,
                  "mask_sha256": sha(np.packbits(mask).tobytes()) if mask is not None else None,
                  "measured_tiles": sum(region["directly_measured_tiles"] for region in evidence["regions"]) if evidence else 0}


def recount(freeze, rows):
    cases = freeze["cases"]
    require(freeze["schema"] == SCHEMA and freeze["design"] == DESIGN and len(cases) == len(rows), "Unexpected freeze")
    roster = {}
    for key in cases:
        for field in ("source", "reference_sha256", "submitted_sha256", "truth_sha256"):
            require(len(key[field]) == 64 and all(ch in "0123456789abcdef" for ch in key[field]), "Invalid input hash")
        by_source = roster.setdefault(key["source"], {})
        binding = key["quality"], key["role"]
        require(binding not in by_source, "Duplicate source variant")
        by_source[binding] = key
    roles = {(quality, role) for quality in QUALITIES for role in (*POSITIVE, *ORDINARY)}
    require(len(roster) == 34 and all(set(group) == roles for group in roster.values()), "Missing source cases")
    counts = {}
    for quality in QUALITIES:
        counts[str(quality)] = {"roles": {role: Counter(files=0, prior_flags=0, added=0, useful=0) for _, role in roles},
                                "moderate_or_strong_sources": set()}
    for key, row in zip(cases, rows, strict=True):
        require({field: row[field] for field in key} == key, "Measured pair differs from its freeze")
        role = row["role"]
        height, width = row["shape"]
        pixels, known, overlap = (row[k] for k in ("added_pixels", "changed", "overlap"))
        require(type(row["prior_reported"]) is bool and all(type(count) is int for count in (pixels, known, overlap))
                and 512 <= min(height, width) <= max(height, width) <= 768
                and 0 <= overlap <= min(pixels, known) and pixels + known - overlap <= height * width
                and bool(known) == (role in POSITIVE) and not (row["prior_reported"] and pixels), "Impossible area or routing")
        require((not pixels and row["measured_tiles"] == 0 and row["mask_sha256"] is None)
                or (pixels >= 1024 and not pixels % 256 and 0 < row["measured_tiles"] <= pixels // 256
                    and len(row["mask_sha256"]) == 64), "Invalid texture evidence")
        useful = bool(known and 2 * overlap >= known and 10 * overlap >= 3 * (pixels + known - overlap))
        record = counts[str(row["quality"])]
        record["roles"][role].update(files=1, prior_flags=int(row["prior_reported"]), added=int(bool(pixels)), useful=int(useful))
        if useful and role in POSITIVE[1:]:
            record["moderate_or_strong_sources"].add(row["source"])
    for record in counts.values():
        added_ordinary = sum(record["roles"][role]["added"] for role in ORDINARY)
        serious_flags = sum(record["roles"][role]["added"] for role in ORDINARY if role not in {"local_sharpen", "generated_feather"})
        groups = len(record["moderate_or_strong_sources"])
        record.update(moderate_or_strong_sources=groups, added_ordinary=added_ordinary, serious_flags=serious_flags,
                      qualifies=groups >= 8 and added_ordinary <= 2 and serious_flags == 0)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("output", type=Path)
    rosters = ("primary", "backup", "wavelet", "flux", "patch", "texture", "editors")
    for name in rosters:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = json.loads(args.output.read_bytes())
        result = recount(receipt["freeze"], receipt["rows"])
        require(receipt["result"] == result, "Totals differ")
        print(json.dumps(result, sort_keys=True))
        return

    import cv2

    from app.detectors.image import reference_jpeg_detail_loss
    from app.evidence import image_reference
    from scripts import image_reference_same_jpeg_confirmation as roster

    cv2.setNumThreads(1)
    require(all(getattr(args, name) for name in rosters), "Supply all source qualifications")
    sources = roster.sources(args)
    code = (Path(__file__), Path(reference_jpeg_detail_loss.__file__), Path(image_reference.__file__))
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        freeze = {"schema": SCHEMA, "design": DESIGN, "code": {path.name: sha(path.read_bytes()) for path in code},
                  "cases": [key for key, *_ in inputs(sources)]}
        write(args.output / "freeze.json", freeze)
        print(f"Locked {len(freeze['cases'])} deliveries", flush=True)
        return
    freeze = json.loads((args.output / "freeze.json").read_bytes())
    require(freeze["code"] == {path.name: sha(path.read_bytes()) for path in code}, "Code differs from frozen method")
    rows = []
    with TemporaryDirectory(dir=args.output) as directory:
        for key, case in zip(freeze["cases"], inputs(sources), strict=True):
            require(case[0] == key, "Delivery differs from the frozen input")
            rows.append(measured(case, Path(directory)))
            if len(rows) % 96 == 0:
                print(f"Measured {len(rows)} deliveries", flush=True)
    result = recount(freeze, rows)
    write(args.output / "receipt.json", {"freeze": freeze, "rows": rows, "result": result})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
