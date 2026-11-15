# SPDX-License-Identifier: Apache-2.0
"""Evaluate right-angle JPEG copy localization against the existing image packet checks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.image_exact_turned_copy_transfer import digest, overlap, useful

SCHEMA = "chipmunk.image-jpeg-turned-copy-transfer/1"
QUALITIES = (75, 90)
EDITS = ("quarter", "half", "quarter_soft")
CONTROLS = ("ordinary", "tone", "local_blur", "straight", "mirror")
# One matching source supplied a development crop. Keep every other attributed group.
SLOTS = tuple(slot for slot in range(38) if slot != 1)


def sample(image, *, step):
    import numpy as np

    from app.detectors.image.jpeg_offgrid_copy import _descriptors

    h, w = image.shape
    y, x = np.mgrid[10:h - 10:step, 10:w - 10:step]
    x, y = x.ravel(), y.ravel()
    descriptors, contrast = _descriptors(image, x, y)
    texture = contrast >= 5
    return np.stack((x[texture], y[texture]), axis=1), descriptors[texture]


def alignments(canvas, rotated, turns):
    import cv2
    import numpy as np

    from app.detectors.image.exact_turned_copy import _source_box

    h, w = canvas.shape
    rh, rw = rotated.shape
    points, desc = sample(canvas, step=8)
    targets, rotated_desc = sample(rotated, step=2)
    if len(targets) < 12 or len(points) < 2:
        return []
    cv2.setRNGSeed(902)
    index = cv2.flann_Index(rotated_desc, {"algorithm": 1, "trees": 3})
    nearest, distances = index.knnSearch(desc, 12, params={"checks": 36})
    votes = defaultdict(set)
    for row, col in zip(*np.where(distances < .14)):
        x, y = map(int, points[row])
        a, b = map(int, targets[nearest[row, col]])
        u, v, _, _ = _source_box((a, b, 1, 1), h, w, turns)
        if max(abs(x - u), abs(y - v)) >= 24:
            votes[a - x, b - y].add((x, y))

    refined = defaultdict(set)
    ranked = sorted(votes.items(), key=lambda item: (-len(item[1]), item[0]))[:45]
    for (dx, dy), seeds in ranked:
        if len(seeds) < 2:
            break
        for x, y in sorted(seeds):
            left, top = max(8, x + dx - 2), max(8, y + dy - 2)
            right, bottom = min(rw - 9, x + dx + 2), min(rh - 9, y + dy + 2)
            tile = canvas[y - 8:y + 9, x - 8:x + 9]
            search = rotated[top - 8:bottom + 9, left - 8:right + 9]
            scores = cv2.matchTemplate(search, tile, cv2.TM_CCOEFF_NORMED)
            _, score, _, (a, b) = cv2.minMaxLoc(scores)
            if score >= .95:
                refined[left + a - x, top + b - y].add((x, y))
    return sorted(
        ((delta, seeds) for delta, seeds in refined.items() if len(seeds) >= 2),
        key=lambda item: (-len(item[1]), item[0]),
    )[:14]


def measures(left, right, boxes):
    import numpy as np

    (x, y, w, h), (a, b, _, _) = boxes
    inside = abs(left[y + 6:y + h - 6, x + 6:x + w - 6] - right[b + 6:b + h - 6, a + 6:a + w - 6]).mean(axis=2)
    mean, p90 = float(inside.mean()), float(np.quantile(inside, .90))
    if mean > 3.2 or p90 > 6:
        return None
    perimeters = []
    for l, t, r, d in [(-7, 4, -2, h - 4), (w + 2, 4, w + 7, h - 4), (4, -7, w - 4, -2), (4, h + 2, w - 4, h + 7)]:
        if (min(x + l, y + t, a + l, b + t) >= 0 and x + r <= left.shape[1] and a + r <= right.shape[1]
                and y + d <= left.shape[0] and b + d <= right.shape[0]):
            p = left[y + t:y + d, x + l:x + r]
            q = right[b + t:b + d, a + l:a + r]
            perimeters.append(float(np.abs(p - q).mean()))
    if sum(value >= max(7, 2.5 * mean) for value in perimeters) >= 3:
        return [mean, p90, perimeters]
    return None


def regions(gray, band, color, turns):
    import cv2
    import numpy as np

    from app.detectors.image.exact_turned_copy import _source_box
    from app.detectors.image.jpeg_offgrid_copy import _correlation_field

    h, w = gray.shape
    rgray, rband, rcolor = [np.ascontiguousarray(np.rot90(image, turns)) for image in (gray, band, color)]
    rh, rw = rgray.shape
    kernel = np.ones((9, 9), np.uint8)
    for (dx, dy), seeds in alignments(gray, rgray, turns):
        left, top, right, bottom = max(0, -dx), max(0, -dy), min(w, rw - dx), min(h, rh - dy)
        a = np.s_[top:bottom, left:right]
        b = np.s_[top + dy:bottom + dy, left + dx:right + dx]
        lo, spread = _correlation_field(gray[a], rgray[b])
        hi, detail = _correlation_field(band[a], rband[b])
        agree = (lo > .93) & (hi > .82) & (spread >= 3) & (detail >= 1.3)
        agree[:6] = agree[-6:] = agree[:, :6] = agree[:, -6:] = False
        closed = cv2.morphologyEx(agree.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for x, y, bw, bh, area in boxes[1:].tolist():
            if not (17 <= bw <= 110 and 17 <= bh <= 110 and area >= .5 * bw * bh):
                continue
            count = sum(x - 5 <= sx - left < x + bw + 5 and y - 5 <= sy - top < y + bh + 5 for sx, sy in seeds)
            if not count or agree[y:y + bh, x:x + bw].mean() < .58:
                continue
            l, t = max(0, x - 4), max(0, y - 4)
            r, d = min(right - left, x + bw + 4), min(bottom - top, y + bh + 4)
            first = left + l, top + t, r - l, d - t
            second = left + l + dx, top + t + dy, r - l, d - t
            source = _source_box(second, h, w, turns)
            if min(first[2:]) < 32 or overlap(first, source):
                continue
            if metrics := measures(color, rcolor, (first, second)):
                yield count, area, (first, source), turns, metrics


def candidate(pixels):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.15)
    band = cv2.GaussianBlur(gray, (0, 0), .8) - cv2.GaussianBlur(gray, (0, 0), 2.7)
    color = cv2.GaussianBlur(pixels.astype(np.float32), (0, 0), 1.25)
    matches = sorted([*regions(blurred, band, color, 1), *regions(blurred, band, color, 2)], reverse=True)
    if not matches:
        return "no_pair", []
    selected = matches[0]
    if any(not useful([other[2]], selected[2]) or other[3] != selected[3] for other in matches[1:]):
        return "ambiguous", []
    return "supported_pair", [selected]


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


def _source_group(archive, item):
    from io import BytesIO

    import numpy as np
    from PIL import Image

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
    pixels = np.array(frame.crop((x, y, x + width, y + height)))
    rows = []
    for role, altered, truth in deliveries(pixels, slot):
        for quality in QUALITIES:
            buffer = BytesIO()
            Image.fromarray(altered).save(buffer, format="JPEG", quality=quality, subsampling=2)
            encoded = buffer.getvalue()
            with Image.open(BytesIO(encoded)) as reopened:
                rgb = np.array(reopened)
            existing = _existing(rgb, encoded)
            status, matches = candidate(rgb)
            rows.append({"source": slot, "role": role, "quality": quality, "known_boxes": truth,
                         "shape": [width, height], "encoded_sha256": digest(encoded), "prior": existing,
                         "rotation": {"status": status, "matches": [
                             {"pair": pair, "turns": turns, "interior_mean": round(metrics[0], 5),
                              "interior_p90": round(metrics[1], 5),
                              "exterior_means": [round(v, 5) for v in metrics[2]]}
                             for _, _, pair, turns, metrics in matches]}})
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
                or result["status"] not in {"no_pair", "supported_pair", "ambiguous"}
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
