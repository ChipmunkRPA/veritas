# SPDX-License-Identifier: Apache-2.0
"""Confirm a larger crop footprint on a source-separated JPEG85 panel."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import cv2
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_jpeg_offgrid_quality85 as earlier

replay = earlier.replay
source_review = earlier.source_review
SCHEMA = "chipmunk.image-jpeg-offgrid-quality85-confirmation/1"
PRIOR_FREEZE = "baf205c87eb0c71daeb0384d992ea24bfb6bfdb1c795e0423b8bcacdadaeffca"
MIN_SIDE = 56
HARD_SLOTS = (0, 2, 6, 8, 29, 30, 33, 34, 35)
EDIT_COUNT = 12
GATE = {"minimum_added_edit_groups": 4, "minimum_added_64_groups": 2, "minimum_added_72_groups": 2,
        "maximum_added_ordinary_groups": 1, "maximum_added_repetition_groups": 0,
        "maximum_wrong_edit_views": 1}


def photo_sources(cache, previous):
    graph = replay.offgrid.committed(cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    excluded_hashes = {row["source_sha256"] for row in previous["sources"]}
    excluded_authors = {row["author"].casefold() for row in previous["sources"]}
    by_key = {row["key"]: row for row in graph["sources"]}
    choices = []
    for group in graph["candidate_only_unlinked_components"]:
        members = [by_key[key] for key in group]
        if any(row.get("source_jpeg_sha256") in excluded_hashes or row["author"].casefold() in excluded_authors
               for row in members):
            continue
        eligible = [row for row in members if row["license_url"] in replay.original.LICENSES
                    and min(row["size"]) >= 480 and max(row["size"]) >= 640]
        if eligible:
            choices.append(min(eligible, key=_rank))
    selected = []
    for row in sorted(choices, key=_rank):
        if row["author"].casefold() in excluded_authors:
            continue
        excluded_authors.add(row["author"].casefold())
        selected.append(row)
        if len(selected) == EDIT_COUNT:
            break
    replay.require(len(selected) == EDIT_COUNT, "Need twelve source-separated contributors")
    return selected


def _rank(row):
    return replay.digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def repetition_sources(earlier_plan):
    with patch.object(replay, "SLOTS", HARD_SLOTS):
        rows = replay.roster()
    for row in rows:
        replay.require(all(row[key].casefold() != old[key].casefold()
                           for old in earlier_plan["sources"] for key in ("author", "source_sha256")),
                       "Repeated-structure control overlaps the first quality-85 panel")
    return [{**row, "file": f"source-{row['slot']:02d}.jpg"} for row in rows]


def prepared(photos, repetition, root):
    for index, row in enumerate(photos):
        _, original = replay.original.photo(row)
        softened = original.copy()
        blur_box = (73, 82, 151, 164)
        softened.paste(original.filter(ImageFilter.GaussianBlur(1.6)).crop(blur_box), blur_box[:2])
        controls = [("ordinary", original, None, None),
                    ("contrast", ImageEnhance.Contrast(original).enhance(1.12), None, None),
                    ("blur", softened, None, None)]
        edits = [(label, *replay.offgrid.authored(original, index + 41, size))
                 for label, size in (("copy64", (64, 64)), ("copy72", (72, 80)))]
        for treatment, image, truth, changed in controls + edits:
            yield _encode(image, index, treatment, row["source_jpeg_sha256"], truth, changed)
    for row in repetition:
        _, image = replay._source(root / row["file"], row)
        yield _encode(image, row["slot"], "repetition", row["source_sha256"], None, None)


def _encode(image, slot, treatment, source_sha256, truth, changed):
    body = replay.original.encode(image, 85)
    return ({"slot": slot, "quality": 85, "treatment": treatment, "source_sha256": source_sha256,
             "input_sha256": replay.digest(body), "truth": [list(box) for box in truth] if truth else None,
             "changed_mask_sha256": replay.digest(changed.tobytes()) if changed is not None else None},
            body, changed)


def results(rows):
    by_treatment = {}
    for treatment in ("copy64", "copy72", "ordinary", "contrast", "blur", "repetition"):
        selected = [row for row in rows if row["treatment"] == treatment]
        by_treatment[treatment] = {
            "views": len(selected), "current_useful": sum(row["current_useful"] for row in selected),
            "candidate_useful": sum(row["serving_useful"] for row in selected),
            "current_flags": sum(bool(row["verified_pairs"]) and not row["added"] for row in selected),
            "added_flags": sum(row["added"] for row in selected),
            "assessed_by_candidate": sum((row["offgrid"] or {}).get("status") in {"no_pair", "supported_pair"}
                                         for row in selected),
        }
    added = [row for row in rows if row["added"]]
    ordinary = {row["slot"] for row in added if row["treatment"] in {"ordinary", "contrast", "blur"}}
    useful = [row for row in added if row["serving_useful"]]
    actual = {"minimum_added_edit_groups": len({row["slot"] for row in useful}),
              "minimum_added_64_groups": sum(row["treatment"] == "copy64" for row in useful),
              "minimum_added_72_groups": sum(row["treatment"] == "copy72" for row in useful),
              "maximum_added_ordinary_groups": len(ordinary),
              "maximum_added_repetition_groups": sum(row["treatment"] == "repetition" for row in added),
              "maximum_wrong_edit_views": sum(row["truth"] is not None and not row["serving_useful"] for row in added)}
    passed = all(actual[key] >= bound if key.startswith("minimum") else actual[key] <= bound
                 for key, bound in GATE.items())
    return {"by_treatment": by_treatment, "gate_actual": actual, "gate_passed": passed,
            "verified_crop_assets": sum(2 * len(row["verified_pairs"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "review"))
    for name in ("previous", "cache", "public", "run"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    previous = args.previous.read_bytes()
    replay.require(replay.digest(previous) == PRIOR_FREEZE, "Earlier trial freeze differs")
    previous = json.loads(previous)
    photos = photo_sources(args.cache, previous)
    repetition = repetition_sources(previous)
    authors = {row["author"].casefold() for row in photos + repetition}
    replay.require(len(authors) == EDIT_COUNT + len(HARD_SLOTS), "Confirmation needs distinct contributors")
    paths = [Path(__file__), *(Path(module.__file__) for module in
              (earlier, replay, replay.copy_move, replay.copy_similarity, replay.jpeg_copy_blocks,
               replay.jpeg_dq, replay.jpeg_offgrid_copy, replay.verify_image_source, replay.original, replay.offgrid,
               source_review))]
    frozen = {"schema": SCHEMA, "minimum_side": MIN_SIDE, "gate": GATE, "confidence_weight": 0,
              "previous_freeze": PRIOR_FREEZE, "cache_sha256": replay.digest(args.cache.read_bytes()),
              "roster_sha256": replay.digest(replay.ROSTER.read_bytes()),
              "code": {path.name: replay.digest(path.read_bytes()) for path in paths},
              "public_sources": repetition,
              "photo_sources": [{key: row[key] for key in ("author", "photo_page", "license_url",
                                 "source_jpeg_sha256", "rgb_sha256", "size")} for row in photos],
              "views": [row for row, _, _ in prepared(photos, repetition, args.public)],
              "runtime": {"numpy": replay.np.__version__, "opencv": cv2.__version__,
                          "pillow": replay.version("pillow"), "jpeglib": replay.version("jpeglib"),
                          "jpeg": replay.features.version_codec("jpg")}}
    path = args.run / "freeze.json"
    if args.action == "freeze":
        args.run.mkdir(mode=0o700)
        replay._save(path, frozen)
        print(f"Frozen {len(frozen['views'])} inputs: {replay.digest(path.read_bytes())}", flush=True)
        return
    replay.require(replay._json(frozen) == path.read_bytes(), "Confirmation scope differs from its freeze")
    rows, altered = [], set()
    profiles = source_review._QUANTIZATION_SHA256
    original_review = source_review.review

    def candidate(data, rgb):
        review, pairs = original_review(data, rgb)
        if pairs and min(pairs[0][0][2:]) < MIN_SIDE:
            return {"status": "no_pair", "measurement": None}, []
        return review, pairs

    with (patch.object(source_review, "_QUANTIZATION_SHA256", profiles | {earlier.QUALITY_85}),
          patch.object(source_review, "review", candidate)):
        for (view, body, changed), expected in zip(prepared(photos, repetition, args.public), frozen["views"], strict=True):
            replay.require(view == expected and source_review.supported(body), "Confirmation input changed")
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as decoded:
                row = replay.score(decoded.convert("RGB"), view, body, changed, args.run / "submitted.jpg", altered)
            row["added"] = row["basis"] == replay.OFFGRID_BASIS
            row["current_useful"] = row["serving_useful"] and not row["added"]
            rows.append(row)
    output = args.run / "result.json"
    document = {"schema": SCHEMA, "freeze_sha256": replay.digest(path.read_bytes()),
                "rows": rows, "alterations_rejected": sorted(altered), **results(rows)}
    if args.action == "review":
        replay.require(replay._json(document) == output.read_bytes(), "Confirmation native measurements changed")
    else:
        replay._save(output, document)
    print(json.dumps({key: value for key, value in document.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
