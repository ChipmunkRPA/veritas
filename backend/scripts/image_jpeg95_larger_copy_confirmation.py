# SPDX-License-Identifier: Apache-2.0
"""Confirm the JPEG95 between-block path on larger held copy regions."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import image_jpeg_offgrid_quality95_transfer as prior

SCHEMA = "chipmunk.image-jpeg95-larger-copy-confirmation/1"
METHOD = {"profile_sha256": prior.JPEG95, "minimum_crop_side": 56, "groups": 16, "repetition_slots": [4, 5, 10, 11, 16, 17, 21, 22],
          "copy_size": [88, 96], "feather": 8, "minimum_added_useful_groups": 4, "minimum_added_useful_feathered": 2,
          "maximum_added_ordinary_groups": 1, "maximum_added_repetition": 0, "maximum_wrong_added_locations": 1}
EDITED = {"sharp", "feathered"}
replay = prior.replay
require = replay.require
sha = replay.original.sha


def exclusions(value, hashes, authors):
    if isinstance(value, list):
        for item in value:
            exclusions(item, hashes, authors)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "author" and isinstance(item, str):
                authors.add(item.casefold())
            elif key in {"source_sha256", "source_jpeg_sha256"} and isinstance(item, str):
                hashes.add(item)
            else:
                exclusions(item, hashes, authors)


def roster(cache, prior_files):
    graph = replay.offgrid.committed(cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    hashes, authors = set(), set()
    for path in prior_files:
        exclusions(json.loads(path.read_bytes()), hashes, authors)
    authors.update(row["author"].casefold() for row in graph["sources"] if row.get("source_jpeg_sha256") in hashes)
    with patch.object(replay, "SLOTS", tuple(METHOD["repetition_slots"])):
        repeat = replay.roster()
    authors.update(row["author"].casefold() for row in repeat)
    by_key = {row["key"]: row for row in graph["sources"]}
    eligible = []
    for component in graph["candidate_only_unlinked_components"]:
        group = [by_key[key] for key in component]
        if any(row["author"].casefold() in authors or row.get("source_jpeg_sha256") in hashes for row in group):
            continue
        candidates = [row for row in group if row.get("license_url") in replay.original.LICENSES
                      and min(row["size"]) >= 480 and max(row["size"]) >= 640]
        if candidates:
            eligible.append(min(candidates, key=rank))
    selected = []
    for row in sorted(eligible, key=rank):
        author = row["author"].casefold()
        if author not in authors:
            selected.append(row)
            authors.add(author)
        if len(selected) == METHOD["groups"]:
            break
    require(len(selected) == METHOD["groups"], "Insufficient contributor-separated input photos")
    return selected, repeat


def rank(row):
    return replay.digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def variants(original, slot):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    sharp, truth, sharp_mask = replay.offgrid.authored(original, slot + 311, METHOD["copy_size"])
    x, y, width, height = truth[0]
    bounds = (x, y, x + width, y + height)
    yy, xx = np.mgrid[:height, :width]
    distance = np.minimum.reduce((yy, height - 1 - yy, xx, width - 1 - xx))
    alpha = np.minimum(distance / METHOD["feather"], 1)[..., None]
    base = np.asarray(original.crop(bounds))
    yield "ordinary", original, None, None
    for name, patch_image in (("local_tone", ImageEnhance.Brightness(original.crop(bounds)).enhance(.85)),
                              ("local_blur", original.filter(ImageFilter.GaussianBlur(1.6)).crop(bounds)),
                              ("feathered", sharp.crop(bounds))):
        pixels = np.rint(base * (1 - alpha) + np.asarray(patch_image) * alpha).astype(np.uint8)
        delivery = original.copy()
        delivery.paste(Image.fromarray(pixels), bounds[:2])
        yield name, delivery, truth if name == "feathered" else None, np.any(base != pixels, axis=2) if name == "feathered" else None
    yield "sharp", sharp, truth, sharp_mask


def inputs(photos, repetition, public):
    def encode(image, slot, treatment, source_hash, truth=None, mask=None):
        body = replay.original.encode(image, 95)
        view = {"slot": slot, "quality": 95, "treatment": treatment, "source_sha256": source_hash,
                "input_sha256": replay.digest(body), "truth": [list(box) for box in truth] if truth else None,
                "changed_pixels": int(mask.sum()) if mask is not None else 0,
                "changed_mask_sha256": replay.digest(mask.tobytes()) if mask is not None else None}
        return view, body, mask

    for slot, photo in enumerate(photos):
        _, original = replay.original.photo(photo)
        for role, image, truth, mask in variants(original, slot):
            yield encode(image, slot, role, photo["source_jpeg_sha256"], truth, mask)
    for source in repetition:
        _, image = replay._source(public / f"source-{source['slot']:02d}.jpg", source)
        yield encode(image, source["slot"], "repetition", source["source_sha256"])


def lock(args, photos, repeated):
    from app.evidence import image_packets_v7

    modules = (prior, replay, replay.offgrid, replay.original, replay.copy_similarity, replay.copy_move,
               replay.jpeg_copy_blocks, replay.jpeg_offgrid_copy, replay.image_jpeg_offgrid_source, image_packets_v7)
    scripts = [Path(__file__), *(Path(module.__file__) for module in modules)]
    return {"schema": SCHEMA, "method": METHOD, "confidence_weight": 0,
            "implementation": {path.name: sha(path) for path in scripts}, "cache_sha256": sha(args.cache),
            "exclusions": [sha(path) for path in args.exclusions],
            "photo_sources": [{key: row[key] for key in ("author", "photo_page", "license_url", "source_jpeg_sha256",
                                                            "rgb_sha256", "size")} for row in photos],
            "repetition_sources": repeated, "views": [view for view, _, _ in inputs(photos, repeated, args.public)]}


def summary(rows):
    by_treatment = defaultdict(Counter)
    useful_groups = set()
    ordinary_groups = set()
    for row in rows:
        role, added, useful = row["treatment"], row["basis"] == replay.OFFGRID_BASIS, row["serving_useful"]
        cell = by_treatment[role]
        cell.update(views=1, current_useful=useful and not added, candidate_useful=useful,
                    current_flags=bool(row["verified_pairs"]) and not added, added_flags=added,
                    wrong_added=added and role in EDITED and not useful,
                    crops=len(row["verified_pairs"]) * 2)
        if added and useful:
            useful_groups.add(row["slot"])
        if added and role not in EDITED:
            ordinary_groups.add(("repetition" if role == "repetition" else "photo", row["slot"]))
    feather = by_treatment["feathered"]
    gates = {"useful_groups": len(useful_groups) >= METHOD["minimum_added_useful_groups"],
             "feathered": feather["candidate_useful"] - feather["current_useful"] >= METHOD["minimum_added_useful_feathered"],
             "ordinary": len(ordinary_groups) <= METHOD["maximum_added_ordinary_groups"],
             "repetition": by_treatment["repetition"]["added_flags"] <= METHOD["maximum_added_repetition"],
             "wrong": sum(cell["wrong_added"] for cell in by_treatment.values()) <= METHOD["maximum_wrong_added_locations"]}
    return {"counts": dict(by_treatment), "added_useful_source_groups": len(useful_groups),
            "added_ordinary_source_groups": len(ordinary_groups), "gate": gates, "passed": all(gates.values())}


def measure(args, photos, repeated, frozen):
    from PIL import Image

    source = replay.image_jpeg_offgrid_source
    minimum_side = source._minimum_crop_side
    profiles = source._QUANTIZATION_SHA256 | {METHOD["profile_sha256"]}

    def candidate(data):
        ordinary_minimum = minimum_side(data)
        if ordinary_minimum is None:
            return None
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            quantization = bytes(image.quantization[0]) + bytes(image.quantization[1])
        return METHOD["minimum_crop_side"] if replay.digest(quantization) == METHOD["profile_sha256"] else ordinary_minimum

    rows, tamper = [], set()
    with (patch.object(source, "_QUANTIZATION_SHA256", profiles), patch.object(source, "_minimum_crop_side", candidate),
          TemporaryDirectory(dir=args.run) as folder):
        for (view, body, mask), expected in zip(inputs(photos, repeated, args.public), frozen["views"], strict=True):
            require(view == expected and source.supported(body), "Frozen JPEG profile differs")
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
                rows.append(replay.score(image.convert("RGB"), view, body, mask, Path(folder) / "input.jpg", tamper))
            if len(rows) % 12 == 0:
                print(f"Completed {len(rows)}/{len(frozen['views'])}", flush=True)
    return {"freeze": frozen, "rows": rows, "summary": summary(rows), "source_tamper_rejected": sorted(tamper)}


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("exclusions", type=Path, nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    photos, repeated = roster(args.cache, args.exclusions)
    frozen = lock(args, photos, repeated)
    plan = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        replay._save(plan, frozen)
        print(replay.digest(plan.read_bytes()))
        return
    require(plan.read_bytes() == replay._json(frozen), "Frozen confirmation inputs changed")
    record = measure(args, photos, repeated, frozen)
    output = args.run / "receipt.json"
    if args.stage == "run":
        replay._save(output, record)
    else:
        require(output.read_bytes() == replay._json(record), "Native confirmation differs")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
