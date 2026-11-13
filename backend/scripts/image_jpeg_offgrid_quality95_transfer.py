# SPDX-License-Identifier: Apache-2.0
"""Test the existing exact between-block JPEG evidence on a held quality-95 profile."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_jpeg_offgrid_quality85_confirmation as prior

SCHEMA = "chipmunk.image-jpeg-offgrid-quality95-transfer/1"
JPEG95 = "b40e056724db43022fcc08475520e2fda95cb91c31551a5dc0f9f9f6b50b1ddc"
MIN_SIDE = 56
EDIT_GROUPS = 12
HARD_SLOTS = (0, 1, 3, 6, 8, 14, 15, 18, 23, 29, 34, 35)
GATE = {"minimum_added_edit_groups": 4, "minimum_added_64_groups": 2, "minimum_added_72_groups": 2,
        "maximum_added_ordinary_groups": 1, "maximum_added_repetition_groups": 0,
        "maximum_wrong_sharp_views": 1}
replay, source_review = prior.replay, prior.source_review


def roster(args):
    graph = replay.offgrid.committed(args.cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    earlier = [json.loads(path.read_bytes()) for path in args.exclusions]
    excluded = {row["source_sha256"] for plan in earlier for row in plan.get("rows", ())}
    excluded_authors = set()
    for plan in earlier:
        for key in ("sources", "photo_sources", "public_sources"):
            for row in plan.get(key, ()):
                excluded.add(row.get("source_sha256", row.get("source_jpeg_sha256")))
                excluded_authors.add(row["author"].casefold())
    excluded_authors.update(row["author"].casefold() for row in graph["sources"]
                            if row.get("source_jpeg_sha256") in excluded)
    with patch.object(replay, "SLOTS", HARD_SLOTS):
        repetition = replay.roster()
    for row in repetition:
        excluded_authors.add(row["author"].casefold())
    by_key = {row["key"]: row for row in graph["sources"]}
    eligible = []
    for group in graph["candidate_only_unlinked_components"]:
        rows = [by_key[key] for key in group]
        if any(row.get("source_jpeg_sha256") in excluded or row["author"].casefold() in excluded_authors for row in rows):
            continue
        candidates = [row for row in rows if row.get("license_url") in replay.original.LICENSES
                      and min(row["size"]) >= 480 and max(row["size"]) >= 640]
        if candidates:
            eligible.append(min(candidates, key=rank))
    photos = []
    for row in sorted(eligible, key=rank):
        author = row["author"].casefold()
        if author not in excluded_authors:
            excluded_authors.add(author)
            photos.append(row)
        if len(photos) == EDIT_GROUPS:
            break
    replay.require(len(photos) == EDIT_GROUPS, "Insufficient distinct commercially permitted photo groups")
    return photos, repetition


def rank(row):
    return replay.digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def variants(original, index):
    sharp = {label: replay.offgrid.authored(original, index + 87, size)
             for label, size in (("copy64", (64, 64)), ("copy72", (72, 80)))}
    x, y, width, height = sharp["copy72"][1][0]
    bounds = (x, y, x + width, y + height)
    blurred = original.copy()
    blurred.paste(original.filter(ImageFilter.GaussianBlur(1.6)).crop(bounds), (x, y))
    toned = original.copy()
    toned.paste(ImageEnhance.Brightness(original.crop(bounds)).enhance(.85), (x, y))
    yield "ordinary", original, None, None
    yield "local_tone", toned, None, None
    yield "local_blur", blurred, None, None
    for label, (image, truth, mask) in sharp.items():
        yield label, image, truth, mask
    pixels = np.array(original)
    patch_pixels = pixels[y:y + height, x:x + width]
    donor = np.asarray(sharp["copy72"][0])[y:y + height, x:x + width]
    dy, dx = np.mgrid[:height, :width]
    distance = np.minimum.reduce((dy, height - 1 - dy, dx, width - 1 - dx))
    alpha = np.minimum(distance / 8, 1)[..., None]
    blended = np.rint(patch_pixels * (1 - alpha) + donor * alpha).astype(np.uint8)
    mask = np.any(blended != patch_pixels, axis=2)
    patch_pixels[:] = blended
    yield "feather72", Image.fromarray(pixels), sharp["copy72"][1], mask


def inputs(args, photos, repetition):
    def encode(image, slot, treatment, source, truth=None, changed=None):
        body = replay.original.encode(image, 95)
        view = {"slot": slot, "quality": 95, "treatment": treatment, "source_sha256": source,
                "input_sha256": replay.digest(body), "truth": [list(box) for box in truth] if truth else None,
                "changed_mask_sha256": replay.digest(changed.tobytes()) if changed is not None else None}
        return view, body, changed

    for index, photo in enumerate(photos):
        _, original = replay.original.photo(photo)
        for label, image, truth, changed in variants(original, index):
            yield encode(image, index, label, photo["source_jpeg_sha256"], truth, changed)
    for source in repetition:
        _, image = replay._source(args.public / f"source-{source['slot']:02d}.jpg", source)
        yield encode(image, source["slot"], "repetition", source["source_sha256"])


def summarize(rows):
    by_treatment = {}
    for treatment in ("copy64", "copy72", "feather72", "ordinary", "local_tone", "local_blur", "repetition"):
        subset = [row for row in rows if row["treatment"] == treatment]
        by_treatment[treatment] = {"views": len(subset), "current_useful": sum(row["current_useful"] for row in subset),
                                  "candidate_useful": sum(row["serving_useful"] for row in subset),
                                  "current_flags": sum(bool(row["verified_pairs"]) and not row["added"] for row in subset),
                                  "added_flags": sum(row["added"] for row in subset)}
    added = [row for row in rows if row["added"]]
    useful = [row for row in added if row["serving_useful"] and row["treatment"] in {"copy64", "copy72"}]
    actual = {"minimum_added_edit_groups": len({row["slot"] for row in useful}),
              "minimum_added_64_groups": sum(row["treatment"] == "copy64" for row in useful),
              "minimum_added_72_groups": sum(row["treatment"] == "copy72" for row in useful),
              "maximum_added_ordinary_groups": len({row["slot"] for row in added
                                                      if row["truth"] is None and row["treatment"] != "repetition"}),
              "maximum_added_repetition_groups": sum(row["treatment"] == "repetition" for row in added),
              "maximum_wrong_sharp_views": sum(row["treatment"] in {"copy64", "copy72"}
                                                 and not row["serving_useful"] for row in added)}
    return {"by_treatment": by_treatment, "gate_actual": actual,
            "gate_passed": all(actual[key] >= limit if key.startswith("minimum") else actual[key] <= limit
                               for key, limit in GATE.items()),
            "verified_crop_assets": 2 * sum(len(row["verified_pairs"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("exclusions", type=Path, nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    photos, repetition = roster(args)
    paths = [Path(__file__), *(Path(module.__file__) for module in
             (prior, replay, replay.copy_move, replay.copy_similarity, replay.jpeg_copy_blocks,
              replay.jpeg_dq, replay.jpeg_offgrid_copy, replay.verify_image_source, replay.original, replay.offgrid))]
    frozen = {"schema": SCHEMA, "minimum_side": MIN_SIDE, "gate": GATE, "confidence_weight": 0,
              "profile_sha256": JPEG95, "code": {path.name: replay.digest(path.read_bytes()) for path in paths},
              "cache_sha256": replay.digest(args.cache.read_bytes()),
              "exclusion_sha256": [replay.digest(path.read_bytes()) for path in args.exclusions],
              "repetition_sources": repetition, "source_exposure": "Reused public architecture control images",
              "photo_sources": [{key: row[key] for key in ("author", "photo_page", "license_url",
                                  "source_jpeg_sha256", "rgb_sha256", "size")} for row in photos],
              "views": [row for row, _, _ in inputs(args, photos, repetition)],
              "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": replay.version("pillow"),
                          "jpeglib": replay.version("jpeglib"), "jpeg": replay.features.version_codec("jpg")}}
    freeze = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        replay._save(freeze, frozen)
        print(f"Frozen {len(frozen['views'])} inputs: {replay.digest(freeze.read_bytes())}", flush=True)
        return
    replay.require(replay._json(frozen) == freeze.read_bytes(), "Frozen source, runtime or detector differs")
    incumbent_profile = source_review._minimum_crop_side
    # Permit only the committed table pair and its fixed footprint while using the real native verifier.
    profiles = source_review._QUANTIZATION_SHA256 | {JPEG95}

    def candidate_profile(data):
        if incumbent_profile(data) is None:
            return None
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            tables = bytes(image.quantization[0]) + bytes(image.quantization[1])
        return MIN_SIDE if replay.digest(tables) == JPEG95 else incumbent_profile(data)

    rows, altered = [], set()
    with (patch.object(source_review, "_QUANTIZATION_SHA256", profiles),
          patch.object(source_review, "_minimum_crop_side", candidate_profile),
          TemporaryDirectory(dir=args.run) as folder):
        for (view, body, changed), expected in zip(inputs(args, photos, repetition), frozen["views"], strict=True):
            replay.require(view == expected and source_review.supported(body), "Candidate input differs")
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as decoded:
                row = replay.score(decoded.convert("RGB"), view, body, changed, Path(folder) / "submitted.jpg", altered)
            row["added"] = row["basis"] == replay.OFFGRID_BASIS
            row["current_useful"] = row["serving_useful"] and not row["added"]
            rows.append(row)
            if len(rows) % 12 == 0:
                print(f"Scored {len(rows)}/{len(frozen['views'])}", flush=True)
    output = args.run / "result.json"
    result = {"schema": SCHEMA, "freeze_sha256": replay.digest(freeze.read_bytes()), "rows": rows,
              "alterations_rejected": sorted(altered), **summarize(rows)}
    if args.stage == "review":
        replay.require(replay._json(result) == output.read_bytes(), "Candidate native replay differs")
    else:
        replay._save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
