# SPDX-License-Identifier: Apache-2.0
"""Confirm the 32-pixel minimum evidence footprint on a separate JPEG panel."""

from __future__ import annotations

import argparse
import inspect
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, features

from app.evidence import image_jpeg_offgrid_source
from app.evidence.image_copy_similarity_binding import OFFGRID_MIN_SIDE
from scripts import image_offgrid_patch_transfer as prior

SCHEMA = "chipmunk.image-offgrid-patch-confirmation/1"
FIRST_INPUTS_SHA256 = "10f3aec114df1156dbe43b1128d9133a672c5da689824e3ccd8ed343687a9f2a"
PHOTOS = 16
CATEGORIES = {"abandoned", "cityscapes"}
reference = prior.reference
require, sha, digest, read, write = prior.require, prior.sha, prior.digest, prior.read, prior.write


def additional_sources(inputs):
    cache, earlier, translation, first, transfer, expansion, root = inputs
    graph = prior.committed(cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    block = prior.committed(earlier, "image-jpeg-copy-blocks-transfer-2026-11-12.json", "freeze_sha256")
    old = prior.committed(translation, "image-translation-patch-transfer-2026-11-11.json", "freeze_sha256")
    exposed = {row["source_sha256"] for source in (block, read(first)) for row in source["rows"]}
    exposed.update(row["source_jpeg_sha256"] for row in old["sources"])
    exposed.update(row["source"]["source_jpeg_sha256"] for row in old["hard_controls"])
    inventories = (transfer.read_text(), expansion.read_text())
    excluded = set()
    for row in graph["sources"]:
        h = row.get("source_jpeg_sha256")
        if h in exposed or (h and any(h in text or f'"{row["key"]}"' in text for text in inventories)):
            excluded.add(row["author"])
    controls = hard_controls(root, exposed)
    excluded.update(row["author"].casefold() for row in controls)
    by_key = {row["key"]: row for row in graph["sources"]}

    def rank(row):
        return digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())

    candidates = []
    for component in graph["candidate_only_unlinked_components"]:
        members = [by_key[key] for key in component]
        if any(row["author"] in excluded for row in members):
            continue
        options = [row for row in members if row["license_url"] in reference.LICENSES
                   and min(row["size"]) >= 480 and max(row["size"]) >= 640]
        if options:
            candidates.append(min(options, key=rank))
    selected = sorted(candidates, key=rank)[:PHOTOS]
    require(len(selected) == len({row["author"] for row in selected}) == PHOTOS, "Need sixteen separately attributed groups")
    return selected, controls


def hard_controls(root, excluded):
    panel = read(prior.PANELS / "image-affine-native-scale-2026-11-11.json")
    require(read(root / "plan.json") == panel["source_roster"]
            and read(root / "acquisition.json") == panel["source_intake"], "Repeated-structure roster differs")
    intake = {row["slot"]: row for row in panel["source_intake"]["attempts"]}
    controls = []
    for row in panel["source_roster"]["sources"]:
        if row["category"] not in CATEGORIES:
            continue
        item = intake[row["slot"]]
        path = root / item["name"]
        require(row["licence_url"].startswith(("https://creativecommons.org/licenses/by/", "https://creativecommons.org/licenses/by-sa/"))
                and item["status"] == "qualified" and sha(path) == item["sha256"] and item["sha256"] not in excluded,
                "Source rights or hash differ, or the source appeared on the earlier panel")
        controls.append({**row, "path": path, "jpeg_sha256": item["sha256"]})
    require(len(controls) == len({row["author"] for row in controls}) == 14, "Need fourteen separate architecture controls")
    return controls


def cases(inputs):
    sources, controls = additional_sources(inputs)
    for slot, row in enumerate(sources):
        path, image = reference.photo(row)
        source_sha = row["source_jpeg_sha256"]
        yield ("photo", slot, "native", "original"), source_sha, path.read_bytes(), None, None
        softened = image.copy()
        box = (183, 94, 246, 151)
        softened.paste(image.filter(ImageFilter.GaussianBlur(1.4)).crop(box), box[:2])
        recipes = [("original", image, None, None), ("tone", ImageEnhance.Contrast(image).enhance(1.13), None, None),
                   ("blur", softened, None, None)]
        recipes.extend((role, *prior.authored(image, slot + 21, size)) for role, size in zip(prior.ROLES, prior.SIZES, strict=True))
        for quality in prior.QUALITY:
            for role, pixels, truth, mask in recipes:
                yield ("photo", slot, str(quality), role), source_sha, reference.encode(pixels, quality), truth, mask
    for slot, row in enumerate(controls):
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        require(min(width, height) >= 384, "Repeated-structure source is too small")
        scale = max(768 / max(width, height), 384 / min(width, height))
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        width, height = image.size
        left, top = max(0, (width - 768) // 2), max(0, (height - 768) // 2)
        image = image.crop((left, top, left + min(width, 768), top + min(height, 768)))
        for quality in prior.QUALITY:
            yield ("repetition", slot, str(quality), "original"), row["jpeg_sha256"], reference.encode(image, quality), None, None


def freeze(inputs):
    first_rows = json.dumps(read(inputs[3])["rows"], sort_keys=True, separators=(",", ":")).encode()
    require(digest(first_rows) == FIRST_INPUTS_SHA256, "Original frozen inputs differ")
    code = (Path(__file__), Path(prior.__file__), Path(prior.cue.__file__), Path(reference.__file__),
            Path(prior.copy_move.__file__), Path(prior.copy_similarity.__file__), Path(prior.jpeg_copy_blocks.__file__),
            Path(prior.jpeg_dq.__file__), Path(image_jpeg_offgrid_source.__file__))
    files = code + inputs[:-1] + (inputs[-1] / "plan.json", inputs[-1] / "acquisition.json")
    rows = [{"key": key, "sha256": digest(data), "source_sha256": source, "truth": truth,
             "mask_sha256": digest(mask.tobytes()) if mask is not None else None}
            for key, source, data, truth, mask in cases(inputs)]
    baseline = {fn.__name__: digest(inspect.getsource(fn).encode()) for fn in (
        prior.image_packets_v7.selected_translation_pairs, prior.image_packets_v7.translation_pairs)}
    return {"schema": SCHEMA, "min_display_side": OFFGRID_MIN_SIDE, "gate": prior.GATE, "confidence_weight": 0,
            "bindings": {str(i): sha(path) for i, path in enumerate(files)} | baseline, "rows": rows,
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "jpeg": features.version_codec("jpg")}}


def observations(inputs, frozen):
    for (key, source, body, truth, mask), pin in zip(cases(inputs), frozen["rows"], strict=True):
        require(list(key) == pin["key"] and digest(body) == pin["sha256"] and source == pin["source_sha256"]
                and (digest(mask.tobytes()) if mask is not None else None) == pin["mask_sha256"],
                "Prepared confirmation differs from frozen bytes")
        with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        baseline_status, baseline, _ = prior.image_packets_v7.selected_translation_pairs(rgb)
        if baseline_status == "no_pair":
            baseline_status, baseline = prior.jpeg_copy_blocks.analyse(body)
        review, pairs = image_jpeg_offgrid_source.review(body, rgb)
        for pair in pairs:
            require(prior.cue.witness(rgb, pair) == review["measurement"], "Exact native measurement did not reproduce")
        added = baseline_status == "no_pair" and bool(pairs)
        yield {"key": key, "current_status": baseline_status, "candidate_status": review["status"], "current": baseline,
               "candidate": [{"pair": pair, "measurement": review["measurement"]} for pair in pairs], "added": added,
               "current_useful": reference.match(baseline, truth, mask), "candidate_useful": reference.match(pairs, truth, mask),
               "combined_useful": reference.match(pairs if added else baseline, truth, mask)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    for name in ("cache", "earlier", "translation", "first", "transfer", "expansion", "controls"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    inputs = tuple(getattr(args, name) for name in ("cache", "earlier", "translation", "first", "transfer", "expansion", "controls"))
    plan = args.run / "freeze.json"
    if args.action == "freeze":
        args.run.mkdir(mode=0o700)
        write(plan, freeze(inputs))
        print(sha(plan), flush=True)
        return
    frozen = read(plan)
    require(prior.json_ready(freeze(inputs)) == frozen, "Source, truth, or code changed after the freeze")
    rows = []
    start = time.monotonic()
    for index, row in enumerate(observations(inputs, frozen), 1):
        rows.append(row)
        if index % 30 == 0:
            print(f"{index}/{len(frozen['rows'])} views measured in {time.monotonic() - start:.1f} seconds", flush=True)
    output = args.run / "rows.json"
    if args.action == "review":
        require(prior.json_ready(rows) == read(output), "Confirmation measurements changed on replay")
    else:
        require(not output.exists(), "Refusing to overwrite the previous confirmation")
        write(output, rows)
    print(json.dumps(prior.summarize(rows), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
