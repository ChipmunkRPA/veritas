# SPDX-License-Identifier: Apache-2.0
"""Evaluate right-angle JPEG copy localization against the existing image packet checks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts.image_exact_turned_copy_transfer import digest, overlap, useful

SCHEMA = "chipmunk.image-jpeg-turned-copy-transfer/1"
QUALITIES = (75, 90)
EDITS = ("quarter", "half", "quarter_soft")
CONTROLS = ("ordinary", "tone", "local_blur", "straight", "mirror")
# One matching source supplied a development crop. Keep every other attributed group.
SLOTS = tuple(slot for slot in range(38) if slot != 1)


def deliveries(pixels, slot):
    import cv2
    import numpy as np

    h, w = pixels.shape[:2]
    sx, sy = w - 115 - slot * 17 % 77, h - 125 - slot * 29 % 61
    dx, dy = 35 + slot * 19 % 113, 30 + slot * 13 % 97
    yield "ordinary", pixels, None
    yield "tone", np.clip(pixels.astype(np.float32) * .92 + 12, 0, 255).astype(np.uint8), None
    blur = pixels.copy()
    blur[80:152, 96:168] = cv2.GaussianBlur(pixels, (0, 0), 1.2)[80:152, 96:168]
    yield "local_blur", blur, None
    for role, turns, dw, dh, feather, mirror in [
        ("straight", 0, 56, 72, 0, False), ("mirror", 0, 56, 72, 0, True),
        ("quarter", 1, 56, 72, 0, False), ("half", 2, 64, 64, 0, False),
        ("quarter_soft", 3, 72, 64, 6, False),
    ]:
        sw, sh = (dh, dw) if turns % 2 else (dw, dh)
        donor = pixels[sy:sy + sh, sx:sx + sw]
        source = np.rot90(donor[:, ::-1] if mirror else donor, turns)
        changed = pixels.copy()
        target = changed[dy:dy + dh, dx:dx + dw]
        if feather:
            y, x = np.indices((dh, dw))
            alpha = np.minimum(np.minimum.reduce((x, y, dw - 1 - x, dh - 1 - y)) / feather, 1)[..., None]
            source = np.rint(target * (1 - alpha) + source * alpha).astype(np.uint8)
        target[:] = source
        truth = ((dx, dy, dw, dh), (sx, sy, sw, sh)) if turns else None
        yield role, changed, truth


def _native_pair(source, item, packet):
    """Reproduce the current geometric preview boxes from its accepted dense masks."""
    import cv2
    import numpy as np

    from app.detectors.image import copy_affine

    matrix = np.array(item.affine.matrix, np.float64)
    x, y, w, h = packet._crop_box(item.box_a, source.shape[1], source.shape[0])
    masks = copy_affine.dense_pair(copy_affine.luma(source), matrix, (x, y, x + w, y + h))
    if masks is None:
        return None
    return tuple(cv2.boundingRect(mask.astype(np.uint8)) for mask in masks)


def _existing(image, encoded):
    from app.detectors.image import copy_affine, copy_move, jpeg_copy_blocks
    from app.evidence import image_jpeg_offgrid_source, image_packets_v7

    try:
        geometrical = [boxes for item in copy_affine.candidates(image)
                       if (boxes := _native_pair(image, item, image_packets_v7)) is not None]
        affine_status = "supported_pair" if geometrical else "no_pair"
    except copy_move.CandidateBudgetExceeded:
        geometrical, affine_status = [], "unassessed_match_budget"

    status, pairs, basis = image_packets_v7.selected_translation_pairs(image)
    if status == "no_pair":
        block_status, pairs = jpeg_copy_blocks.analyse(encoded)
        if pairs:
            status, basis = "supported_pair", "jpeg_blocks"
        elif block_status == "no_pair":
            review, pairs = image_jpeg_offgrid_source.review(encoded, image)
            status = review["status"]
            basis = "jpeg_offgrid" if pairs else None
        else:
            status = block_status
    return {"status": status, "pairs": pairs, "basis": basis,
            "affine_status": affine_status, "affine_pairs": geometrical}


def source_pixels(archive, item):
    import numpy as np

    from scripts.image_affine_native_scale import decode

    slot = item["slot"]
    data = (archive / item["name"]).read_bytes()
    if item["status"] != "qualified" or digest(data) != item["sha256"]:
        raise ValueError(f"Source changed at group {slot}")
    frame = decode(data)
    if digest(frame.tobytes()) != item["rgb_sha256"] or list(frame.size) != item["size"]:
        raise ValueError(f"Decoded source changed at group {slot}")
    width, height = (704, 480) if slot % 2 else (576, 416)
    x, y = (frame.width - width) // 2, (frame.height - height) // 2
    return np.array(frame.crop((x, y, x + width, y + height)))


def encode_jpeg(pixels, quality):
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality, subsampling=2)
    return buffer.getvalue()


def _source_group(archive, item):
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from app.detectors.image.jpeg_turned_copy import analyse

    slot = item["slot"]
    pixels = source_pixels(archive, item)
    height, width = pixels.shape[:2]
    rows = []
    for role, altered, truth in deliveries(pixels, slot):
        for quality in QUALITIES:
            encoded = encode_jpeg(altered, quality)
            with Image.open(BytesIO(encoded)) as reopened:
                rgb = np.array(reopened)
            existing = _existing(rgb, encoded)
            status, matches = analyse(rgb)
            rows.append({"source": slot, "role": role, "quality": quality, "known_boxes": truth,
                         "shape": [width, height], "encoded_sha256": digest(encoded), "prior": existing,
                         "rotation": {"status": status, "matches": [
                             {"pair": match["pair"], "turns": match["turns"],
                              "interior_mean": match["measurement"]["interior_mean"],
                              "interior_p90": match["measurement"]["interior_p90"],
                              "exterior_means": [v for v in match["measurement"]["perimeter_means"] if v is not None]}
                             for match in matches]}})
    return rows


def _setup():
    import cv2

    cv2.setNumThreads(1)


def _run(archive, workers):
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    manifest = (archive / "freeze.json").read_bytes()
    items = [item for item in json.loads(manifest)["sources"] if item["slot"] in SLOTS]
    if len(items) != len(SLOTS) or len({item["author"].casefold() for item in items}) != len(SLOTS):
        raise ValueError("The held roster requires distinct author groups and every reserved slot")
    with ProcessPoolExecutor(max_workers=workers, initializer=_setup) as executor:
        rows = [
            row
            for group in executor.map(_source_group, repeat(archive), items)
            for row in group
        ]
    return {"schema": SCHEMA, "source_freeze_sha256": digest(manifest), "rows": rows, "summary": _summary(rows)}


def _summary(rows):
    summary = {}
    for quality in QUALITIES:
        counts = Counter(cases=0, edits=0, controls=0, prior_useful=0, added_useful=0,
                         misplaced=0, candidate_control_flags=0, added_control_flags=0,
                         ambiguous=0, baseline_unassessed=0)
        for row in (item for item in rows if item["quality"] == quality):
            truth, baseline, candidate_result = row["known_boxes"], row["prior"], row["rotation"]
            match = [item["pair"] for item in candidate_result["matches"]]
            counts["cases"] += 1
            counts["edits" if truth else "controls"] += 1
            counts["ambiguous"] += candidate_result["status"] == "ambiguous"
            counts["baseline_unassessed"] += baseline["status"] not in {"supported_pair", "no_pair"}
            old_pairs = baseline["pairs"] + baseline["affine_pairs"]
            new = match if baseline["status"] == "no_pair" else []
            if truth:
                old = useful(old_pairs, truth)
                counts["prior_useful"] += old
                counts["added_useful"] += not old and useful(new, truth)
                counts["misplaced"] += bool(new) and not useful(new, truth)
            else:
                counts["candidate_control_flags"] += bool(match)
                counts["added_control_flags"] += bool(new)
        summary[str(quality)] = dict(counts)
    return summary


def review(document):
    rows = document["rows"]
    expected = {(slot, q, role) for slot in SLOTS for q in QUALITIES for role in (*EDITS, *CONTROLS)}
    actual = {(row["source"], row["quality"], row["role"]) for row in rows}
    if document["schema"] != SCHEMA or len(rows) != len(expected) or actual != expected:
        raise ValueError("The JPEG receipt is missing frozen cases or contains duplicates")
    for row in rows:
        result = row["rotation"]
        if ((row["role"] in EDITS) != bool(row["known_boxes"])
                or result["status"] not in {"no_pair", "supported_pair", "ambiguous", "unassessed_match_budget"}
                or len(result["matches"]) != (result["status"] == "supported_pair")):
            raise ValueError("The JPEG receipt disagrees on whether a pair was selected")
        for item in result["matches"]:
            a, b = item["pair"]
            expected_size = a[2:][::-1] if item["turns"] == 1 else a[2:]
            width, height = row["shape"]
            if (item["turns"] not in {1, 2} or b[2:] != expected_size or overlap(a, b)
                    or any(x < 0 or y < 0 or not 32 <= min(w, h) <= max(w, h) <= 118
                           or x + w > width or y + h > height for x, y, w, h in (a, b))):
                raise ValueError("The selected JPEG crops have inconsistent rotation geometry")
            mean, p90, exterior = item["interior_mean"], item["interior_p90"], item["exterior_means"]
            if (not 0 <= mean <= 3.2 or not 0 <= p90 <= 6 or not 3 <= len(exterior) <= 4
                    or not all(math.isfinite(v) for v in exterior)
                    or sum(v >= max(7.0, 2.5 * mean) - 0.00002 for v in exterior) < 3):
                raise ValueError("The selected JPEG crops have insufficient color agreement")
    counts = _summary(rows)
    if document["summary"] != counts:
        raise ValueError("The recorded counts differ from the source-coordinate recount")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--workers", default=1, type=int, choices=range(1, 5))
    args = parser.parse_args()
    data = json.loads(args.receipt.read_bytes()) if args.archive is None else _run(args.archive, args.workers)
    print(json.dumps(review(data), indent=2, sort_keys=True))
    if args.archive is not None:
        with args.receipt.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
