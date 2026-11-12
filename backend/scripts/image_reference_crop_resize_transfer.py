"""Validate exact earlier-copy crop and resize geometry on frozen source groups."""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, __version__ as pillow_version

from app.detectors.image import reference_alignment, reference_crop_resize, reference_resize
from scripts import image_edge_spread_transfer as panel

SCHEMA = "chipmunk.image-reference-crop-resize-transfer/1"
POSITIVE = ("sharp", "feathered")
ORDINARY = ("untouched_crop", "prior_jpeg", "contrast")
EDIT_CONTROLS = ("local_blur",)
INCOMPATIBLE = ("rotation", "fractional", "nonproportional", "different_photo")
ROLES = (*POSITIVE, *ORDINARY, *EDIT_CONTROLS, *INCOMPATIBLE)
SCALES = (.72, .86, 1.17)


def cases(slot, images, target, donor):
    source = images["original"]
    h, w = source.shape[:2]
    # Every view in a group uses the same asymmetric native crop.
    size_a, size_b = 8 + slot % 3, 6 + (slot + 1) % 3
    left = max(0, min(w * size_a // 100, target[0] - 12))
    top = max(0, min(h * size_b // 100, target[1] - 12))
    right = min(w, max(w - w * size_b // 100, target[2] + 12))
    bottom = min(h, max(h - h * size_a // 100, target[3] + 12))
    cropped = (left, top, right, bottom)
    known = {"x": left, "y": top, "w": right - left, "h": bottom - top}
    scale = SCALES[slot % 3]
    sw, sh = round(known["w"] * scale), round(known["h"] * scale)

    def crop(pixels):
        return Image.fromarray(pixels).crop(cropped)

    def resize(image):
        if slot % 3 == 0:
            return cv2.resize(np.asarray(image), (sw, sh), interpolation=cv2.INTER_AREA)
        kernel = Image.Resampling.BICUBIC if slot % 3 == 1 else Image.Resampling.LANCZOS
        return np.asarray(image.resize((sw, sh), kernel))

    original = crop(source)
    baseline = resize(original)
    blur = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(1.6)))
    sharp = resize(crop(images["edited"]))
    feathered = resize(crop(panel.blend.local(source, images["edited"], target, feather=True)))
    lossy = resize(crop(reference_resize.reference_compare._prior_jpeg(source, 83)))
    contrast = resize(ImageEnhance.Contrast(original).enhance(1.12))
    local_blur = resize(crop(panel.blend.local(source, blur, target, feather=True)))
    rotation = resize(original.rotate(2.0, resample=Image.Resampling.BICUBIC))
    fractional = cv2.warpAffine(baseline, np.array(((1, 0, .5 * sw / known["w"]), (0, 1, .5 * sh / known["h"]))),
                                (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    stretch = np.asarray(original.resize((round(sw * 1.06), sh), Image.Resampling.BICUBIC))
    unrelated = resize(Image.fromarray(donor))
    for role, view in zip(ROLES, (sharp, feathered, baseline, lossy, contrast, local_blur,
                                 rotation, fractional, stretch, unrelated), strict=True):
        truth = np.any(view != baseline, axis=2) if role in POSITIVE else np.zeros(view.shape[:2], dtype=bool)
        yield role, view, truth, None if role in INCOMPATIBLE else known


def profile(development, earlier, later):
    folders = {"development": development, "earlier": earlier, "later": later}
    plans = [panel.read(folder / "plan.json")["cases"] for folder in folders.values()]
    for key in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[key] for row in plan} for plan in plans]
        panel.require([len(g) for g in groups] == [8, 12, 12] and len(set.union(*groups)) == 32,
                      "Photo and creator groups must be disjoint")
    panel.require(all(Counter(row["generator"] for row in plan) == {"sdxl": 6, "kandinsky": 6} for plan in plans[1:]),
                  "Held generator balance changed")
    methods = (Path(__file__), Path(reference_crop_resize.__file__), Path(reference_alignment.__file__),
               Path(reference_resize.__file__), Path(reference_resize.reference_compare.__file__),
               Path(panel.blend.__file__), Path(panel.paired.__file__), Path(panel.scoring.__file__))
    return {
        "schema": SCHEMA,
        "code": {str(path.relative_to(panel.ROOT)): panel.sha(path) for path in methods},
        "intakes": {role: {file: panel.sha(folder / file) for file in ("plan.json", "attempts.jsonl", "result.json")}
                    for role, folder in folders.items()},
        "runtime": {"opencv": cv2.__version__, "pillow": pillow_version, "numpy": np.__version__},
        "protocol": {"roles": list(ROLES), "formats": list(panel.FORMATS), "scales": list(SCALES),
                     "geometry": "Every accepted four-edge native crop must match exactly. Incompatible controls must stay unassessed.",
                     "useful": "Native submitted IoU at least 0.30 and recall at least 0.50.",
                     "advance": "Per held panel and delivery: at least 4 additional useful feathered locations, at least 6 exact untouched "
                                "crops, at most 4 nuisance highlights across at most 3 source groups, zero wrong accepted crops, "
                                "zero assessed incompatible controls."},
    }


def sources(folder, name):
    if name != "development":
        return [(row["slot"], row["generator"], native, row["box"]) for row, native, _ in panel.paired.inputs(folder)]
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    result = []
    for slot, _, native, truth in panel.scoring.panels(folder, plan, attempts):
        ys, xs = np.nonzero(truth)
        target = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        result.append((slot, "sd15", native, target))
    return result


def run(folder, name):
    inputs = sources(folder, name)
    for ordinal, (slot, generator, images, target) in enumerate(inputs):
        original = images["original"]
        other = inputs[(ordinal + 1) % len(inputs)][2]["original"]
        for role, pixels, truth, expected in cases(slot, images, target, other):
            for fmt in panel.FORMATS:
                body, decoded = panel.paired.encoded(pixels, fmt)
                rgb = np.asarray(decoded)
                with Image.open(io.BytesIO(body)) as carrier:
                    baseline, prior_mask = reference_alignment.compare(original, rgb, carrier)
                    if baseline["status"] == "integer_alignment_unresolved":
                        baseline, prior_mask = reference_resize.compare(original, rgb, carrier)
                    candidate, mask = reference_crop_resize.compare(original, rgb, carrier)
                measured = candidate.get("reference_crop_bounds")
                crop = {key: measured[key] for key in ("x", "y", "w", "h")} if measured else None
                def stats(view):
                    return panel.scoring.measure(view if view is not None else np.zeros_like(truth), truth)
                yield {
                    "source": slot, "generator": generator, "format": fmt, "role": role, "truth": expected, "crop": crop,
                    "input_sha256": panel.paired.shared.digest(body), "truth_sha256": panel.paired.shared.digest(truth.tobytes()),
                    "mask_sha256": panel.paired.shared.digest(mask.tobytes()) if mask is not None else None,
                    "status": candidate["status"], "reference_status": baseline["status"],
                    "geometry_support": measured, "regions": candidate["regions"],
                    "candidate": stats(mask), "reference": stats(prior_mask),
                }


def summary(rows):
    results = {}
    for fmt in panel.FORMATS:
        views = [row for row in rows if row["format"] == fmt]
        ordinary = [row for row in views if row["role"] in ORDINARY and row["candidate"]["reported"]]
        result = {}
        for role in ROLES:
            result[role] = Counter()
            for row in (row for row in views if row["role"] == role):
                hit = panel.useful(row["candidate"])
                prior = panel.useful(row["reference"])
                result[role].update(views=1, exact_crop=row["crop"] is not None and row["crop"] == row["truth"],
                                    wrong_crop=row["crop"] is not None and row["crop"] != row["truth"],
                                    compatible=row["status"] == "comparable", incumbent_compatible=row["reference_status"] == "comparable",
                                    highlighted=bool(row["candidate"]["reported"]), incumbent_highlighted=bool(row["reference"]["reported"]),
                                    useful=hit, incumbent_useful=prior, added=hit and not prior)
        results[fmt] = {"by_role": result, "nuisance_highlights": len(ordinary),
                        "nuisance_source_groups": len({row["source"] for row in ordinary}),
                        "wrong_or_incompatible_crops": sum(role["wrong_crop"] for role in result.values())}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "confirm"))
    for field in ("development", "earlier", "later", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    locked = profile(args.development, args.earlier, args.later)
    path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(path, locked)
        return
    panel.require(panel.read(path) == locked, "Frozen source or method differs")
    cv2.setNumThreads(1)
    selection = ("development",) if args.stage == "development" else ("earlier", "later")
    results, commitments = {}, {}
    start = time.monotonic()
    for name in selection:
        rows = list(run(getattr(args, name), name))
        expected = 8 if name == "development" else 12
        panel.require(len(rows) == expected * len(ROLES) * len(panel.FORMATS), "Incomplete test panels")
        journal = args.output / f"{name}.json"
        panel.write(journal, rows)
        results[name], commitments[name] = summary(rows), panel.sha(journal)
        print(name, len(rows), "views", flush=True)
    decision = {}
    if args.stage == "confirm":
        decision = {fmt: all((view := result[fmt])["by_role"]["feathered"]["added"] >= 4
                             and view["by_role"]["untouched_crop"]["exact_crop"] >= 6
                             and view["wrong_or_incompatible_crops"] == 0 and view["nuisance_highlights"] <= 4
                             and view["nuisance_source_groups"] <= 3 for result in results.values()) for fmt in panel.FORMATS}
    panel.write(args.output / f"{args.stage}-summary.json", {"schema": SCHEMA, "freeze_sha256": panel.sha(path),
                "journal_sha256": commitments, "results": results, "advances": decision,
                "elapsed_s": round(time.monotonic() - start, 3)})


if __name__ == "__main__":
    main()
