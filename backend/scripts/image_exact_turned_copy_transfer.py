# SPDX-License-Identifier: Apache-2.0
"""Replay exact orthogonal copy localization on a committed archive of photos."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-exact-turned-copy/1"
GEOMETRIES = ((512, 384), (576, 416), (704, 480), (512, 512))
EDIT_PROFILES = {
    "quarter_24x28": (1, 24, 28, 0),
    "half_36x28": (2, 36, 28, 0),
    "feather_clockwise_44x42": (3, 44, 42, 5),
}
CONTROLS = ("ordinary", "tone", "local_blur", "straight_copy", "mirrored_copy")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def overlap(first, second):
    a, b, aw, ah = first
    x, y, w, h = second
    intersection = max(0, min(a + aw, x + w) - max(a, x)) * max(0, min(b + ah, y + h) - max(b, y))
    return intersection / (aw * ah + w * h - intersection)


def useful(pairs, truth):
    return bool(truth) and any(
        any(min(overlap(a, x), overlap(b, y)) >= .5 for x, y in (truth, truth[::-1]))
        for a, b in pairs
    )


def summarize(rows):
    counts = {role: Counter(cases=0, prior_useful=0, rotated_useful=0, with_rotated_useful=0,
                            rotated_highlights=0, rotated_unassessed=0)
              for role in (*EDIT_PROFILES, *CONTROLS)}
    for row in rows:
        count = counts[row["role"]]
        count["cases"] += 1
        truth = row["truth"]
        prior = useful(row["prior_pairs"], truth)
        new = [item["pair"] for item in row["rotated"]["pairs"]]
        count["prior_useful"] += prior
        count["rotated_useful"] += useful(new, truth)
        # Exact translated pairs keep priority in the application.
        additional = [] if row["translation_status"] != "no_pair" else new
        count["with_rotated_useful"] += prior or useful(additional, truth)
        count["rotated_highlights"] += bool(additional)
        count["rotated_unassessed"] += row["rotated"]["status"] not in {"supported_pair", "no_pair"}
    return counts


def cases(pixels, slot):
    import cv2
    import numpy as np

    def paint(turns, width, height, feather, *, mirror=False):
        full_height, full_width = pixels.shape[:2]
        dx, dy = 35 + slot * 19 % 125, 30 + slot * 13 % 110
        w, h = (height, width) if turns % 2 else (width, height)
        sx, sy = full_width - w - 38 - slot * 17 % 90, full_height - h - 44 - slot * 29 % 85
        donor = pixels[sy:sy + h, sx:sx + w]
        donor = np.rot90(donor[:, ::-1] if mirror else donor, turns)
        result = pixels.copy()
        destination = result[dy:dy + height, dx:dx + width]
        if feather:
            y, x = np.indices((height, width))
            distance = np.minimum.reduce((x, y, width - 1 - x, height - 1 - y))
            alpha = np.minimum(distance / feather, 1.0)[..., None]
            donor = np.rint(destination * (1.0 - alpha) + donor * alpha).astype(np.uint8)
        destination[:] = donor
        return result, ((dx, dy, width, height), (sx, sy, w, h))

    yield "ordinary", pixels, None
    yield "tone", np.clip(pixels.astype(np.float32) * .92 + 12, 0, 255).astype(np.uint8), None
    blur = pixels.copy()
    blur[80:152, 96:168] = cv2.GaussianBlur(pixels, (0, 0), 1.2)[80:152, 96:168]
    yield "local_blur", blur, None
    yield "straight_copy", paint(0, 32, 28, 0)[0], None
    yield "mirrored_copy", paint(0, 32, 28, 0, mirror=True)[0], None
    for role, profile in EDIT_PROFILES.items():
        yield role, *paint(*profile)


def source_pixels(archive, source):
    import numpy as np

    from scripts.image_affine_native_scale import decode

    slot = source["slot"]
    data = (archive / source["name"]).read_bytes()
    if source["status"] != "qualified" or digest(data) != source["sha256"]:
        raise ValueError(f"Unqualified or changed source {slot}")
    native = decode(data)
    if digest(native.tobytes()) != source["rgb_sha256"] or list(native.size) != source["size"]:
        raise ValueError(f"Source decode differs for group {slot}")
    w, h = GEOMETRIES[slot % len(GEOMETRIES)]
    x, y = (native.width - w) // 2, (native.height - h) // 2
    if min(x, y) < 0:
        raise ValueError(f"Crop does not fit source {slot}")
    return np.array(native.crop((x, y, x + w, y + h)))


def run(archive):
    import cv2
    import numpy as np

    from app.detectors.image import copy_affine, exact_small_copy as exact, exact_turned_copy
    from app.evidence import image_copy_affine_binding as affine, image_packets_v7

    cv2.setNumThreads(1)
    manifest = (archive / "freeze.json").read_bytes()
    sources = json.loads(manifest)["sources"]
    if len(sources) != 38 or len({source["author"].casefold() for source in sources}) != len(sources):
        raise ValueError("The 38 attributed source groups are required")
    rows = []
    for source in sources:
        slot = source["slot"]
        original = source_pixels(archive, source)
        h, w = original.shape[:2]
        for role, pixels, truth in cases(original, slot):
            translated, baseline, _ = ("no_pair", [], None)
            if image_packets_v7.supported_size(w, h):
                translated, baseline, _ = image_packets_v7.selected_translation_pairs(pixels)
            if translated == "no_pair":
                measurement = exact.analyse(pixels)
                translated = measurement["status"]
                baseline = [pair["pair"] for pair in measurement["pairs"]]
            if truth and affine.supported(w, h):
                proposals = copy_affine.candidates(pixels)
                gray = copy_affine.luma(pixels) if proposals else None
                for proposal in proposals:
                    ax, ay, aw, ah = image_packets_v7._crop_box(proposal.box_a, w, h)
                    bounds = ax, ay, ax + aw, ay + ah
                    masks = copy_affine.dense_pair(gray, np.array(proposal.affine.matrix), bounds)
                    if masks is not None:
                        baseline.append(tuple(cv2.boundingRect(mask.astype(np.uint8)) for mask in masks))
            outcome = exact_turned_copy.analyse(pixels)
            for match in outcome["pairs"]:
                if exact.witness(pixels, *match["pair"], turns=match["turns"]) != match["measurement"]:
                    raise ValueError(f"Source-pixel verification failed for source {slot}")
            rows.append({"group": slot, "source_sha256": source["sha256"], "size": [w, h],
                         "role": role, "rgb_sha256": digest(pixels.tobytes()), "truth": truth,
                         "translation_status": translated, "prior_pairs": baseline, "rotated": outcome})
    return {"schema": SCHEMA, "source_freeze_sha256": digest(manifest), "rows": rows, "summary": summarize(rows)}


def review(document):
    rows = document["rows"]
    keys = {(row["group"], row["role"]) for row in rows}
    expected = {(slot, role) for slot in range(38) for role in (*EDIT_PROFILES, *CONTROLS)}
    if document["schema"] != SCHEMA or keys != expected or len(rows) != len(expected):
        raise ValueError("The held receipt has missing or duplicated cases")
    for row in rows:
        result = row["rotated"]
        status, pairs = result["status"], result["pairs"]
        if len(pairs) != (status == "supported_pair"):
            raise ValueError("The held receipt disagrees on whether a pair was selected")
        for item in pairs:
            (_, _, w, h), (_, _, u, v) = item["pair"]
            if (item["turns"] not in (1, 2) or (u, v) != ((h, w) if item["turns"] == 1 else (w, h))
                    or item["measurement"]["matching_pixels"] != w * h):
                raise ValueError("The held receipt has a wrong orientation or matched pixel count")
    actual = summarize(rows)
    if actual != document["summary"]:
        raise ValueError("Recorded counts differ from the source-coordinate recount")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--archive", type=Path, help="Recreate all deliveries from the attributed image archive")
    args = parser.parse_args()
    if args.archive:
        document = run(args.archive)
        with args.receipt.open("x") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        document = json.loads(args.receipt.read_bytes())
    print(json.dumps(review(document), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
