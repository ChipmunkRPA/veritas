# SPDX-License-Identifier: Apache-2.0
"""Frozen exact native RGB comparison for small within-image copies."""

from __future__ import annotations

import argparse
import inspect
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import exact_small_copy as detector
from app.evidence import image_packets_v7
from scripts import image_jpeg_offgrid_quality95_transfer as sources
from scripts import image_local_residual_covariance as regional

SCHEMA = "chipmunk.image-exact-small-copy/1"
CAL_COUNT, HELD_COUNT = 8, 16
REPETITION = (0, 2, 6, 8, 13, 18, 25, 29, 30, 33, 34, 35)
EDITS = frozenset({"sharp28", "sharp36", "feather44"})
CRITERIA = {"added_sharp": 8, "added_feather": 4, "control_groups": 1, "wrong_boxes": 0}
replay = sources.replay
need, read, write, sha = regional.need, regional.read, regional.write, regional.sha


def selections(args):
    graph = replay.offgrid.committed(args.cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    by_key = {row["key"]: row for row in graph["sources"]}
    occupied = set()
    for path in args.exclude:
        previous = read(path)
        for key in ("sources", "photo_sources", "public_sources", "repetition_sources"):
            occupied.update(row["author"].casefold() for row in previous.get(key, ()) if row.get("author"))
        for row in previous.get("rows", ()):
            occupied.update(source["author"].casefold() for source in graph["sources"]
                            if source.get("source_jpeg_sha256") == row.get("source_sha256"))
    with patch.object(replay, "SLOTS", REPETITION):
        repeated = replay.roster()
    occupied.update(row["author"].casefold() for row in repeated)
    ranked = []
    for group in graph["candidate_only_unlinked_components"]:
        rows = [by_key[key] for key in group]
        if any(row["author"].casefold() in occupied for row in rows):
            continue
        chosen = [row for row in rows if row.get("license_url") in replay.original.LICENSES
                  and min(row["size"]) >= 500 and max(row["size"]) >= 640]
        if chosen:
            ranked.append(min(chosen, key=rank))
    selected = []
    for item in sorted(ranked, key=rank):
        author = item["author"].casefold()
        if author not in occupied:
            selected.append(item)
            occupied.add(author)
        if len(selected) == CAL_COUNT + HELD_COUNT:
            break
    need(len(selected) == CAL_COUNT + HELD_COUNT, "Insufficient distinct attributed photo groups")
    return selected[:CAL_COUNT], selected[CAL_COUNT:], repeated


def rank(row):
    return sha(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def copy_patch(image, slot, side, feather=0):
    rgb = np.array(image)
    height, width = rgb.shape[:2]
    x, y = 48 + (slot * 37) % 140, 43 + (slot * 31) % 155
    a, b = width - side - (58 + (slot * 29) % 107), height - side - (58 + (slot * 13) % 93)
    first, second = rgb[y:y + side, x:x + side], rgb[b:b + side, a:a + side]
    if feather:
        yy, xx = np.mgrid[:side, :side]
        alpha = np.minimum.reduce((yy, xx, side - 1 - yy, side - 1 - xx))
        alpha = np.minimum(alpha / feather, 1.)[..., None]
        replacement = np.rint(first * (1. - alpha) + second * alpha).astype(np.uint8)
    else:
        replacement = second
    changed = np.any(first != replacement, axis=-1)
    first[:] = replacement
    need(changed.any(), "Copy recipe did not change pixels")
    return Image.fromarray(rgb), ((x, y, side, side), (a, b, side, side)), changed


def cases(args, part):
    calibration, held, repeated = selections(args)
    for slot, row in enumerate(calibration if part == "cal" else held):
        _, base = replay.original.photo(row)
        source = row["source_jpeg_sha256"]
        tone = ImageEnhance.Contrast(base).enhance(1.12)
        blurred = base.copy()
        bounds = (54, 62, 98, 106)
        blurred.paste(base.filter(ImageFilter.GaussianBlur(1.6)).crop(bounds), bounds[:2])
        for role, pixels in (("original", base), ("tone", tone), ("blur", blurred)):
            yield (source, role), pixels, None, None
        if part == "held":
            for role, side, feather in (("sharp28", 28, 0), ("sharp36", 36, 0), ("feather44", 44, 6)):
                yield (source, role), *copy_patch(base, slot, side, feather)
    if part == "held":
        for row in repeated:
            _, image = replay._source(args.public / f"source-{row['slot']:02d}.jpg", row)
            yield (row["source_sha256"], "repetition"), image, None, None
        for family, source, _, original, edited in regional.sources(args, "held"):
            if family == "flux":
                rgb = regional.common.panel.BASE.feather(original, edited, np.any(original != edited, axis=2))
                yield (source, "generated_feathered"), Image.fromarray(rgb), None, None


def measure(pixels, truth, changed, *, full):
    candidate = detector.analyse(pixels)
    current, current_pairs, _ = image_packets_v7.selected_translation_pairs(pixels) if full else ("not_run", [], None)
    pairs = [pair["pair"] for pair in candidate["pairs"]]
    proof = []
    for pair in candidate["pairs"]:
        first, second = pair["pair"]
        crops = [pixels[y:y + h, x:x + w] for x, y, w, h in (first, second)]
        need(np.array_equal(*crops) and sha(crops[0].tobytes()) == sha(crops[1].tobytes()), "Native crop pixels do not match")
        need(pair["measurement"] == detector.witness(pixels, *pair["pair"]), "Recomputed native measurement differs")
        proof.append(sha(crops[0].tobytes()))
    return {"candidate": candidate, "candidate_useful": bool(replay.original.match(pairs, truth, changed)),
            "current_status": current, "current_pairs": current_pairs,
            "current_useful": bool(replay.original.match(current_pairs, truth, changed)), "native_pair_rgb_sha256": proof}


def binding(item):
    key, image, truth, changed = item
    pixels = np.asarray(image)
    with io.BytesIO() as stream:
        image.save(stream, "PNG", compress_level=1)
        encoded = stream.getvalue()
    return {"key": list(key), "input_sha256": sha(encoded), "native_rgb_sha256": sha(pixels.tobytes()),
            "truth": [list(box) for box in truth] if truth else None, "changed_sha256": sha(changed.tobytes()) if truth else None}


def summarize(rows):
    groups = defaultdict(Counter)
    added = Counter()
    flagged = set()
    for row in rows:
        source, role = row["key"]
        base = bool(row["current_pairs"])
        new = bool(row["candidate"]["pairs"] and row["current_status"] == "no_pair")
        tally = groups[role]
        tally.update(views=1, current_flags=int(base), candidate_flags=int(new),
                     current_useful=int(row["current_useful"]),
                     combined_useful=int(row["current_useful"] or (new and row["candidate_useful"])),
                     candidate_only_useful=int(row["candidate_useful"]))
        if not new:
            continue
        if role not in EDITS:
            flagged.add(source)
        elif not row["candidate_useful"]:
            added["wrong_boxes"] += 1
        else:
            added["added_feather" if role == "feather44" else "added_sharp"] += 1
    actual = {key: int(added[key]) for key in CRITERIA} | {"control_groups": len(flagged)}
    passed = all(actual[key] >= CRITERIA[key] for key in ("added_sharp", "added_feather")) and all(
        actual[key] <= CRITERIA[key] for key in ("control_groups", "wrong_boxes"))
    return {"strata": groups, "gate_actual": actual, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held", "review"))
    parser.add_argument("run", type=Path)
    data = Path("backend/data/independent-evaluation")
    for name, path in (("cache", "image-fresh-source-cache-v39/private-groups.json"),
                       ("public", "image-affine-native-scale-2026-11-11"), ("lama", "image-bg-real-acquisition-v225"),
                       ("flux-earlier", "image-residual-patch-held-2026-11-11/plan.json"),
                       ("flux-cache", "image-fresh-source-cache-v39/private-groups.json")):
        parser.add_argument("--" + name, type=Path, default=data / path)
    parser.add_argument("--exclude", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    files = [Path(module.__file__) for module in (detector, regional, sources)]
    comparator = {fn.__name__: sha(inspect.getsource(fn).encode()) for fn in (
        image_packets_v7.selected_translation_pairs, image_packets_v7.translation_pairs)}
    frozen = {"schema": SCHEMA, "code": {path.name: sha(path.read_bytes()) for path in (*files, Path(__file__))} | comparator,
              "confidence_weight": 0, "criteria": CRITERIA,
              "exclusion_sha256": [sha(path.read_bytes()) for path in args.exclude],
              "views": {part: [binding(row) for row in cases(args, part)] for part in ("cal", "held")}}
    lock = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(lock, frozen)
        print(sha(lock.read_bytes()), {part: len(rows) for part, rows in frozen["views"].items()}, flush=True)
        return
    need(read(lock) == frozen, "Method or input bindings differ from the pre-score freeze")

    def observations(part):
        rows = []
        for item, expected in zip(cases(args, part), frozen["views"][part], strict=True):
            need(binding(item) == expected, "Prepared input differs")
            _, image, truth, changed = item
            result = measure(np.asarray(image), truth, changed, full=part == "held")
            # Read receipts through JSON to avoid treating list/tuple conversion as a pixel difference.
            rows.append(json.loads(json.dumps({**expected, **result})))
        return rows

    calibration = args.run / "calibration.json"
    if args.stage == "calibrate":
        rows = observations("cal")
        positives = {row["key"][0] for row in rows if row["candidate"]["pairs"]}
        need(len(positives) <= 1, "Ordinary qualification exceeded one source. Held inputs remain unopened.")
        write(calibration, {"freeze_sha256": sha(lock.read_bytes()), "ordinary_groups_with_pairs": len(positives), "rows": rows})
        print("Calibration sealed", sha(calibration.read_bytes()), flush=True)
        return
    cal = read(calibration)
    need(cal["freeze_sha256"] == sha(lock.read_bytes()), "Calibration did not use this exact freeze")
    if args.stage == "review":
        need(observations("cal") == cal["rows"], "Calibration replay differs")
    rows = observations("held")
    result = {"freeze_sha256": sha(lock.read_bytes()), "calibration_sha256": sha(calibration.read_bytes()),
              "rows": rows, "summary": summarize(rows)}
    output = args.run / "held.json"
    if args.stage == "review":
        need(read(output) == result, "Native held replay differs")
    else:
        write(output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
