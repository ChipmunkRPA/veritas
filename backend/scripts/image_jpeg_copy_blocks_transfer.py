# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact JPEG block matching on small single-image edits and photo controls."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import copy_move, copy_similarity, jpeg_copy_blocks, jpeg_dq
from app.evidence import image_packets_v7
from scripts import image_translation_patch_transfer as reference

SCHEMA = "chipmunk.image-jpeg-copy-blocks-transfer/1"
LICENSES = {"https://creativecommons.org/licenses/by/2.0/", "https://creativecommons.org/licenses/by-sa/2.0/"}
QUALITIES = (75, 90)
GROUPS = 18
CONTROLS = ("original", "tone", "blur", "resize")
POSITIVE = ("aligned_small", "aligned_medium", "offgrid_small", "offgrid_medium")
GATE = {"minimum_added_aligned_small_groups": 3, "minimum_added_aligned_medium_groups": 3,
        "maximum_extra_control_groups": 1, "maximum_extra_original_groups": 0, "maximum_wrong_added_pairs": 1}
read, sha, write = reference.read, reference.sha, reference.write


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select(cache_path, exposed_path):
    cache = read(cache_path)
    recorded = read(exposed_path)
    require(sha(exposed_path) == read(Path("docs/evaluation-panels/image-translation-patch-transfer-2026-11-11.json"))["freeze_sha256"],
            "The earlier translation freeze changed")
    records = {row["key"]: row for row in cache["sources"]}
    hard = [item["source"] for item in recorded["hard_controls"]]
    prior = {row["key"] for row in recorded["sources"] + hard}
    eligible = []
    for component in cache["candidate_only_unlinked_components"]:
        if prior.intersection(component):
            continue
        options = [records[key] for key in component if records[key]["license_url"] in LICENSES
                   and min(records[key]["size"]) >= 480 and max(records[key]["size"]) >= 640]
        if options:
            eligible.append(min(options, key=rank))
    selected = sorted(eligible, key=rank)[:GROUPS]
    require(len(selected) == GROUPS and len(hard) == 8 and len({row["author_profile"] for row in selected + hard}) == GROUPS + len(hard),
            "Need a complete, contributor-separated comparison roster")
    return selected, hard


def rank(row):
    return digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def photo(record):
    path = Path(record["source_jpeg_file"])
    require(record["license_url"] in LICENSES and sha(path) == record["source_jpeg_sha256"], "Photo rights or bytes differ")
    with Image.open(path) as source:
        require(source.format == "JPEG" and source.getexif().get(274, 1) == 1 and list(source.size) == record["size"], "Source scope changed")
        rgb = source.convert("RGB")
    require(digest(rgb.tobytes()) == record["rgb_sha256"], "Native photo pixels differ")
    w, h = rgb.size
    left, top = max(0, (w - 768) // 2), max(0, (h - 768) // 2)
    return path, rgb.crop((left, top, left + min(w, 768), top + min(h, 768)))


def edit(image, slot, size, off_grid):
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    w, h = size
    ax, ay = 43 + (slot * 17) % 88, 37 + (slot * 13) % 66
    dx, dy = 8 * ((width - ax - w - 58) // 8), 8 * ((height - ay - h - 58) // 8)
    bx, by = ax + dx + (3 if off_grid else 0), ay + dy + (5 if off_grid else 0)
    copy = rgb.copy()
    copy[ay:ay+h, ax:ax+w] = rgb[by:by+h, bx:bx+w]
    changed = np.any(rgb[ay:ay+h, ax:ax+w] != copy[ay:ay+h, ax:ax+w], axis=2)
    require(changed.any(), "Authored patch did not alter the photo")
    return Image.fromarray(copy), ((ax, ay, w, h), (bx, by, w, h)), changed


def encode(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        return buffer.getvalue()


def cases(cache, exposed):
    selected, hard = select(cache, exposed)
    for slot, record in enumerate(selected + hard):
        path, image = photo(record)
        group = "photo" if slot < GROUPS else "repetition"
        yield (group, slot, "native", "original"), record, path.read_bytes(), None, None
        if group == "photo":
            local = image.copy()
            box = (69, 83, 128, 157)
            local.paste(image.filter(ImageFilter.GaussianBlur(1.6)).crop(box), box[:2])
            smaller = (round(image.width * .81), round(image.height * .81))
            resampled = image.resize(smaller, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)
            variants = [("original", image, None, None), ("tone", ImageEnhance.Contrast(image).enhance(1.16), None, None),
                        ("blur", local, None, None), ("resize", resampled, None, None)]
            for i, role in enumerate(POSITIVE):
                variants.append((role, *edit(image, slot, (40, 48) if i % 2 == 0 else (64, 64), i >= 2)))
        else:
            variants = [("original", image, None, None)]
        for quality in QUALITIES:
            for role, pixels, truth, changed in variants:
                yield (group, slot, str(quality), role), record, encode(pixels, quality), truth, changed


def bindings(cache, exposed):
    modules = (Path(__file__), Path(jpeg_copy_blocks.__file__), Path(jpeg_dq.__file__), Path(copy_similarity.__file__),
               Path(copy_move.__file__), cache, exposed)
    return {path.name if path not in (cache, exposed) else f"{i}-{path.name}": sha(path) for i, path in enumerate(modules)} | {
        fn.__name__: digest(inspect.getsource(fn).encode())
        for fn in (image_packets_v7.selected_translation_pairs, image_packets_v7.translation_pairs)
    }


def freeze(cache, exposed):
    return {"schema": SCHEMA, "gate": GATE, "bindings": bindings(cache, exposed),
            "useful_iou": .5, "confidence_weight": 0, "rows": [
                {"key": list(key), "encoded_sha256": digest(body), "source_sha256": record["source_jpeg_sha256"],
                 "truth": [list(box) for box in truth] if truth else None,
                 "changed_pixels": int(changed.sum()) if changed is not None else 0,
                 "changed_mask_sha256": digest(changed.tobytes()) if changed is not None else None}
                for key, record, body, truth, changed in cases(cache, exposed)]}


def match(pairs, truth, change):
    if not truth:
        return False
    first, second = truth
    x, y, w, h = first
    for boxes in pairs:
        for target, donor in (boxes, boxes[::-1]):
            if min(copy_move.iou(target, first), copy_move.iou(donor, second)) < .5:
                continue
            a, b, c, d = target
            recovered = change[max(0, b-y):min(h, b+d-y), max(0, a-x):min(w, a+c-x)]
            if 2 * recovered.sum() >= change.sum():
                return True
    return False


def verify_pair(body, pair):
    """Read every block in both stated areas and independently compare exact coefficients."""
    coefficient = jpeg_dq.read_coefficients(body)
    boxes = []
    for x, y, w, h in pair:
        require(not any(value % 8 for value in (x, y, w, h)) and x+w <= coefficient.width and y+h <= coefficient.height,
                "Pair is outside the encoded block grid")
        tile = coefficient.y[y//8:(y+h)//8, x//8:(x+w)//8]
        boxes.append(tile)
    require(np.array_equal(*boxes), "Cropped regions have different encoded luminance blocks")
    return digest(boxes[0].astype("<i4", copy=False).tobytes())


def measure(cache, exposed, committed):
    for case, expected in zip(cases(cache, exposed), committed["rows"], strict=True):
        key, _, body, truth, changed = case
        require(list(key) == expected["key"] and digest(body) == expected["encoded_sha256"], "Held view differs from frozen bytes")
        with Image.open(io.BytesIO(body)) as image:
            rgb = np.asarray(image.convert("RGB"))
        status, baseline, _ = image_packets_v7.selected_translation_pairs(rgb)
        new_status, blocks = jpeg_copy_blocks.analyse(body)
        combined = blocks if status == "no_pair" else baseline
        yield {"key": list(key), "baseline_status": status, "block_status": new_status,
               "baseline": [list(map(list, pair)) for pair in baseline], "blocks": [list(map(list, pair)) for pair in blocks],
               "paired_coefficients_sha256": [verify_pair(body, pair) for pair in blocks],
               "baseline_useful": bool(match(baseline, truth, changed)), "blocks_useful": bool(match(blocks, truth, changed)),
               "combined_useful": bool(match(combined, truth, changed)), "fallback_added": status == "no_pair" and bool(blocks)}


def report(rows):
    table, gained = {}, {key: set() for key in POSITIVE}
    extra_control_groups, extra_original_groups = set(), set()
    wrong = 0
    for row in rows:
        group, slot, quality, role = row["key"]
        bucket = table.setdefault(f"{group}/{quality}/{role}", Counter())
        bucket.update(views=1, baseline_pairs=bool(row["baseline"]), block_pairs=bool(row["blocks"]),
                      baseline_useful=row["baseline_useful"], block_useful=row["blocks_useful"],
                      combined_useful=row["combined_useful"], fallback_added=row["fallback_added"],
                      assessed=row["block_status"] in {"no_pair", "supported_pair"})
        if role in POSITIVE and row["fallback_added"]:
            if row["combined_useful"]:
                gained[role].add(slot)
            else:
                wrong += 1
        elif row["fallback_added"]:
            extra_control_groups.add(slot)
            if role == "original":
                extra_original_groups.add(slot)
    counts = {role: len(slots) for role, slots in gained.items()}
    passed = (counts["aligned_small"] >= GATE["minimum_added_aligned_small_groups"]
              and counts["aligned_medium"] >= GATE["minimum_added_aligned_medium_groups"]
              and len(extra_control_groups) <= GATE["maximum_extra_control_groups"]
              and len(extra_original_groups) <= GATE["maximum_extra_original_groups"]
              and wrong <= GATE["maximum_wrong_added_pairs"])
    return {"passed": passed, "views": len(rows), "strata": table, "added_useful_groups": counts, "wrong_added_pairs": wrong,
            "extra_control_groups": len(extra_control_groups), "extra_original_groups": len(extra_original_groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("exposed", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    committed = freeze(args.cache, args.exposed)
    commitment = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(commitment, committed)
        print(sha(commitment), flush=True)
        return
    require(committed == read(commitment), "Code, sources or truth differ from the pre-score freeze")
    observed = args.run / "rows.json"
    rows = list(measure(args.cache, args.exposed, committed))
    if args.stage == "review":
        require(rows == read(observed), "Replayed block and native pair evidence differs")
        print(f"Reproduced {len(rows)} held JPEG comparisons", flush=True)
        return
    require(not observed.exists(), "The held measurements already exist")
    write(observed, rows)
    receipt = {"schema": SCHEMA, "freeze_sha256": sha(commitment), "rows_sha256": sha(observed), "summary": report(rows),
               "confidence_weight": 0}
    write(args.run / "result.json", receipt)
    print({key: value for key, value in receipt["summary"].items() if key != "strata"}, flush=True)


if __name__ == "__main__":
    main()
