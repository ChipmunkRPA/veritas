# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay scaled photo pairs in locally captured browser screenshots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.reference_heif import display_pixels
from app.detectors.image import reference_scaled_screenshot as check
from app.evidence.image_reference import inspect

HERE = Path(__file__).resolve().parent
LAYOUTS = (.66, .83, .91, .75, .59, .86)
DELIVERIES = {"png": None, "jpeg75": 75, "jpeg85": 85, "jpeg90": 90}
OPTIONS = {"allow_proportional_resize": True, "allow_oriented_crop": True,
           "allow_oriented_resize": True, "allow_embedded_photo": True}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def prepare(run, sources):
    (run / "photo-fixtures").mkdir(parents=True)
    plan, identities = [], []
    seen_photos, seen_authors = set(), set()
    for folder in sources:
        for row in read(folder / "qualification.json")["accepted"]:
            source = row["source"]
            photo, author = source["flickr_photo_id"], source["flickr_owner"]
            if photo in seen_photos or author in seen_authors:
                raise ValueError("Sources must use distinct photos and contributors")
            seen_photos.add(photo)
            seen_authors.add(author)
            index = len(plan)
            for variant, role in (("original", "photo"), ("edited", "edit")):
                body = (folder / row["folder"] / f"{variant}.png").read_bytes()
                if sha(body) != row["fingerprints"][variant]["encoded_sha256"]:
                    raise ValueError(f"Photo fingerprint changed at index {index}")
                (run / "photo-fixtures" / f"{index}-{role}.png").write_bytes(body)
            rw, rh = row["fingerprints"]["original"]["size"]
            scale = max(LAYOUTS[index % len(LAYOUTS)], 166 / min(rw, rh))
            x, y = 93 + (index * 17) % 111, 63 + (index * 23) % 117
            w, h = round(rw * scale), round(rh * scale)
            plan.append(dict(slot=index, x=x, y=y, w=w, h=h, layout=index % 4,
                             width=max(674, x + w + 84), height=max(530, y + h + 70)))
            identities.append({"slot": index, "photo": photo, "author": author,
                               "attribution": source["photo_attribution"], "fingerprints": row["fingerprints"]})
    write(run / "layout-plan.json", plan)
    write(run / "sources.json", identities)


def commitments(run):
    paths = [run / name for name in ("layout-plan.json", "sources.json", "captures.json")]
    paths.extend(sorted((run / "photo-fixtures").glob("*.png")))
    paths.extend(sorted((run / "captures").glob("*.png")))
    expected = read(run / "captures.json")["rows"]
    for row in expected:
        image = run / "captures" / f"{row['slot']}-{row['role']}.png"
        if sha(image.read_bytes()) != row["png_sha256"]:
            raise ValueError(f"Browser capture changed: {image.name}")
    return {"inputs": {str(path.relative_to(run)): sha(path.read_bytes()) for path in paths},
            "code": {Path(path).name: sha(Path(path).read_bytes()) for path in
                     (__file__, check.__file__, inspect.__code__.co_filename,
                      HERE / "capture_image_crop_multiscale_transfer.cjs")}}


def image(path):
    with Image.open(path) as source:
        return np.array(source.convert("RGB"))


def encode(pixels, quality):
    with io.BytesIO() as out:
        if quality is None:
            Image.fromarray(pixels).save(out, format="PNG")
        else:
            Image.fromarray(pixels).save(out, format="JPEG", quality=quality, subsampling=2)
        return out.getvalue()


def cases(run, slot, plan):
    x, y, w, h = (plan[slot][key] for key in ("x", "y", "w", "h"))
    before, after, chrome = (image(run / "captures" / f"{slot}-{role}.png") for role in ("photo", "edit", "template"))
    with Image.open(run / "photo-fixtures" / f"{slot}-photo.png") as source:
        earlier = display_pixels(source)
    view = np.s_[y:y+h, x:x+w]
    truth = np.max(np.abs(before.astype(np.int16) - after), axis=2) >= 2
    if not truth.any() or truth.sum() != truth[view].sum():
        raise ValueError(f"Edited capture lacks confined differences at index {slot}")
    blank = np.zeros_like(truth)
    yield "clean", before, blank
    yield "edit", after, truth
    yield "template", chrome, blank

    preview = chrome.copy()
    preview[view] = np.array(Image.fromarray(earlier).resize((w, h), Image.Resampling.LANCZOS))
    yield "resample", preview, blank

    local = before.copy()
    roi = np.s_[y+h//4:y+2*h//5, x+w//3:x+w//2]
    local[roi] = (local[roi] * .68 + 7).astype(np.uint8)
    yield "local_tone", local, np.max(np.abs(before.astype(np.int16) - local), axis=2) >= 2

    recolored = before.copy()
    recolored[view] = (recolored[view] * .82 + 12).astype(np.uint8)
    yield "global_tone", recolored, blank

    crop = chrome.copy()
    padding = w // 8
    crop[y:y+h, x:x+w-2*padding] = before[y:y+h, x+padding:x+w-padding]
    yield "cropped", crop, blank

    donor = image(run / "photo-fixtures" / f"{(slot + 1) % len(plan)}-photo.png")
    stranger = chrome.copy()
    stranger[view] = np.array(Image.fromarray(donor).resize((w, h)))
    yield "unrelated", stranger, blank

    size = (min(596, w), min(h, round(h * min(596, w) / w)))
    inset = np.array(Image.fromarray(earlier).resize(size, Image.Resampling.BICUBIC))
    dw, dh = size
    duplicate = np.full((dh + 104, 2 * dw + 76, 3), (226, 236, 238), np.uint8)
    duplicate[52:52+dh, 23:23+dw] = duplicate[52:52+dh, dw+53:2*dw+53] = inset
    yield "duplicate", duplicate, np.zeros(duplicate.shape[:2], dtype=bool)


def measure(run):
    plan = read(run / "layout-plan.json")
    with TemporaryDirectory(dir=run) as scratch:
        submitted = Path(scratch) / "submitted"
        for group in plan:
            slot = group["slot"]
            expected = {key: group[key] for key in ("x", "y", "w", "h")}
            earlier = run / "photo-fixtures" / f"{slot}-photo.png"
            reference = image(earlier)
            for role, pixels, target in cases(run, slot, plan):
                for delivery, quality in DELIVERIES.items():
                    body = encode(pixels, quality)
                    submitted.write_bytes(body)
                    with Image.open(earlier) as source, Image.open(io.BytesIO(body)) as carrier:
                        decoded = np.array(carrier.convert("RGB"))
                        start = time.monotonic()
                        result, mask, _ = check.compare(reference, decoded, source, carrier)
                        duration = time.monotonic() - start
                    old = inspect(earlier, submitted, **OPTIONS)
                    size = int(target.sum())
                    selected = int(mask.sum()) if mask is not None else 0
                    intersection = int((target & mask).sum()) if mask is not None else 0
                    placement = result and result["scaled_screenshot"]["submitted_bounds"]
                    exact = placement == expected
                    recall = intersection / size if size else None
                    iou = intersection / (size + selected - intersection) if size else None
                    yield {"slot": slot, "role": role, "delivery": delivery,
                           "sha256": sha(body), "seconds": round(duration, 6),
                           "old_status": old["status"], "old_regions": len(old["regions"]),
                           "placement": placement, "exact_placement": exact, "regions": result and result["regions"],
                           "changed_pixels": size, "selected_pixels": selected,
                           "recall": recall, "iou": iou,
                           "localized": bool(exact and size and recall >= .50 and iou >= .30)}


def summarize(rows):
    totals = {}
    for delivery in DELIVERIES:
        group = [row for row in rows if row["delivery"] == delivery]
        totals[delivery] = {}
        for role in dict.fromkeys(row["role"] for row in group):
            found = [row for row in group if row["role"] == role]
            totals[delivery][role] = {
                "pairs": len(found), "localized": sum(row["localized"] for row in found),
                "exact_placement": sum(row["exact_placement"] for row in found),
                "other_placement": sum(bool(row["placement"]) and not row["exact_placement"] for row in found),
                "highlighted": sum(bool(row["regions"]) for row in found),
                "previous_comparisons": sum(row["old_status"] == "comparable" for row in found),
                "previous_highlights": sum(bool(row["old_regions"]) for row in found),
                "median_seconds": round(median(row["seconds"] for row in found), 3),
            }
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "run"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("sources", type=Path, nargs="*")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    folder = args.directory
    if args.stage == "prepare":
        if not args.sources:
            parser.error("specify at least one qualified source directory")
        prepare(folder, args.sources)
    elif args.stage == "freeze":
        write(folder / "freeze.json", commitments(folder))
        print(sha((folder / "freeze.json").read_bytes()))
    else:
        frozen = read(folder / "freeze.json")
        if commitments(folder) != frozen:
            raise ValueError("Frozen image bytes or comparison code changed")
        rows = list(measure(folder))
        totals = summarize(rows)
        write(folder / "result.json", {"freeze_sha256": sha((folder / "freeze.json").read_bytes()),
                                       "rows": rows, "summary": totals})
        print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
