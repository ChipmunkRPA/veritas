# SPDX-License-Identifier: Apache-2.0
"""Freeze and compare small off-grid JPEG copies with source-separated controls."""

from __future__ import annotations

import argparse
import inspect
import io
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, features

from app.detectors.image import copy_move, copy_similarity, jpeg_copy_blocks, jpeg_dq
from app.detectors.image import jpeg_offgrid_copy as cue
from app.evidence import image_packets_v7
from scripts import image_jpeg_copy_blocks_transfer as reference

SCHEMA = "chipmunk.image-offgrid-patch-transfer/1"
PANELS = Path("docs/evaluation-panels")
PHOTOS = 20
SIZES = ((40, 48), (64, 64))
QUALITY = (75, 90)
ROLES = ("small", "medium")
CATEGORIES = {"architecture", "building-exteriors"}
GATE = {"minimum_added_small_source_groups": 4, "minimum_added_medium_source_groups": 4,
        "maximum_added_ordinary_source_groups": 1, "maximum_added_repeat_source_groups": 0,
        "maximum_wrong_added_edit_views": 2}
read, sha, write, require, digest = reference.read, reference.sha, reference.write, reference.require, reference.digest


def committed(path, panel_name, key):
    expected = read(PANELS / panel_name)[key]
    require(sha(path) == expected, f"{path.name} differs from its earlier public commitment")
    return read(path)


def source_order(row):
    return digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())


def photos(cache, prior, translation, controls):
    graph = committed(cache, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    exposed = committed(prior, "image-jpeg-copy-blocks-transfer-2026-11-12.json", "freeze_sha256")
    before = committed(translation, "image-translation-patch-transfer-2026-11-11.json", "freeze_sha256")
    excluded = {row["source_sha256"] for row in exposed["rows"]}
    excluded.update(row["source_jpeg_sha256"] for row in before["sources"])
    excluded.update(item["source"]["source_jpeg_sha256"] for item in before["hard_controls"])
    by_key = {row["key"]: row for row in graph["sources"]}
    exposed_authors = {row.get("author") for row in graph["sources"] if row.get("source_jpeg_sha256") in excluded}
    exposed_authors.update(row["author"].lower() for row in controls)
    pool = []
    for component in graph["candidate_only_unlinked_components"]:
        members = [by_key[key] for key in component]
        if any(row.get("source_jpeg_sha256") in excluded or row.get("author") in exposed_authors for row in members):
            continue
        options = [row for row in members if row["license_url"] in reference.LICENSES
                   and min(row["size"]) >= 480 and max(row["size"]) >= 640]
        if options:
            pool.append(min(options, key=source_order))
    selected = sorted(pool, key=source_order)[:PHOTOS]
    require(len(selected) == PHOTOS and len({row["author"] for row in selected}) == PHOTOS,
            "Need twenty contributor-separated source groups")
    return selected


def hard_controls(root):
    receipt = read(PANELS / "image-affine-native-scale-2026-11-11.json")
    for name, key in (("plan.json", "source_roster"), ("acquisition.json", "source_intake")):
        require(read(root / name) == receipt[key], "Hard control source plan differs")
    intake = {item["slot"]: item for item in receipt["source_intake"]["attempts"]}
    rows = []
    for row in receipt["source_roster"]["sources"]:
        if row["category"] not in CATEGORIES:
            continue
        item = intake[row["slot"]]
        path = root / item["name"]
        require(row["licence"] in {"CC BY 3.0", "CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 4.0", "CC BY-SA 2.0", "CC BY 2.0"}
                and item["status"] == "qualified" and sha(path) == item["sha256"], "Hard control source or license differs")
        rows.append({**row, "jpeg_sha256": item["sha256"], "path": path, "delivery": item["delivery"]})
    require(len(rows) == len({row["author"] for row in rows}) == 10, "Need ten attribution-separated architecture controls")
    return rows


def authored(image, slot, size):
    pixels = np.asarray(image)
    height, width = pixels.shape[:2]
    w, h = size
    x, y = 53 + (slot * 23) % 117, 52 + (slot * 19) % 114
    if slot % 4 == 0:
        dx, dy = w + 29, 11
    elif slot % 4 == 1:
        dx, dy = 13, h + 33
    elif slot % 4 == 2:
        dx, dy = 8 * ((width - x - w - 47) // 8) + 1, 8 * ((height - y - h - 45) // 8) + 3
    else:
        dx, dy = 8 * ((width - x - w - 54) // 8) + 7, -19
    a, b = x + dx, y + dy
    require((dx % 8 or dy % 8) and (abs(dx) >= w or abs(dy) >= h)
            and 0 <= a < a + w <= width and 0 <= b < b + h <= height, "Authored placement is outside its scope")
    changed = (pixels[y:y + h, x:x + w] != pixels[b:b + h, a:a + w]).any(axis=2)
    require(changed.any(), "Authored copy must alter pixels")
    out = pixels.copy()
    out[y:y + h, x:x + w] = pixels[b:b + h, a:a + w]
    return Image.fromarray(out), ((x, y, w, h), (a, b, w, h)), changed


def cases(cache, prior, translation, control_root):
    controls = hard_controls(control_root)
    for slot, row in enumerate(photos(cache, prior, translation, controls)):
        path, image = reference.photo(row)
        yield ("photo", slot, "native", "original"), row["source_jpeg_sha256"], path.read_bytes(), None, None
        blurred = image.copy()
        box = (67, 83, 132, 149)
        blurred.paste(image.filter(ImageFilter.GaussianBlur(1.6)).crop(box), box[:2])
        recipes = [("original", image, None, None), ("tone", ImageEnhance.Contrast(image).enhance(1.16), None, None),
                   ("blur", blurred, None, None)]
        recipes.extend((role, *authored(image, slot, size)) for role, size in zip(ROLES, SIZES, strict=True))
        for quality in QUALITY:
            for role, pixels, truth, mask in recipes:
                yield ("photo", slot, str(quality), role), row["source_jpeg_sha256"], reference.encode(pixels, quality), truth, mask
    for slot, row in enumerate(controls):
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        require(min(image.size) >= 384, "Repeated-structure control is outside the measured shape")
        for quality in QUALITY:
            yield ("repetition", slot, str(quality), "original"), row["jpeg_sha256"], reference.encode(image, quality), None, None


def scope(body):
    from PIL import JpegImagePlugin

    with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
        if image.mode != "RGB" or JpegImagePlugin.get_sampling(image) != 2 or image.getexif().get(274, 1) != 1:
            return None
        return np.asarray(image.convert("RGB"))


def freeze(inputs):
    files = (Path(__file__), Path(cue.__file__), Path(reference.__file__), Path(copy_move.__file__),
             Path(copy_similarity.__file__), Path(jpeg_copy_blocks.__file__), Path(jpeg_dq.__file__),
             inputs[0], inputs[1], inputs[2], inputs[3] / "plan.json", inputs[3] / "acquisition.json")
    code = {func.__name__: digest(inspect.getsource(func).encode())
            for func in (image_packets_v7.selected_translation_pairs, image_packets_v7.translation_pairs)}
    return {"schema": SCHEMA, "gate": GATE, "bindings": {str(i): sha(path) for i, path in enumerate(files)} | code,
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "jpeg": features.version_codec("jpg")},
            "min_iou_each": .5, "min_changed_recall": .5, "confidence_weight": 0,
            "rows": [{"key": key, "sha256": digest(data), "source_sha256": source,
                      "truth": truth, "mask_sha256": digest(mask.tobytes()) if mask is not None else None}
                     for key, source, data, truth, mask in cases(*inputs)]}


def measure(inputs, frozen):
    for (key, source, body, truth, mask), pin in zip(cases(*inputs), frozen["rows"], strict=True):
        require(list(key) == pin["key"] and digest(body) == pin["sha256"] and source == pin["source_sha256"]
                and (digest(mask.tobytes()) if mask is not None else None) == pin["mask_sha256"],
                "Prepared holdout differs from frozen bytes")
        with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        status, baseline, _ = image_packets_v7.selected_translation_pairs(rgb)
        if status == "no_pair":
            status, baseline = jpeg_copy_blocks.analyse(body)
        eligible = scope(body)
        candidate_status, candidate = cue.analyse(eligible) if eligible is not None else ("unsupported_profile", [])
        pairs = [item["pair"] for item in candidate]
        for item in candidate:
            require(cue.witness(rgb, item["pair"]) == item["measurement"], "Native witness did not reproduce")
        added = status == "no_pair" and bool(candidate)
        yield {"key": key, "current_status": status, "candidate_status": candidate_status, "current": baseline,
               "candidate": candidate, "added": added, "current_useful": reference.match(baseline, truth, mask),
               "candidate_useful": reference.match(pairs, truth, mask),
               "combined_useful": reference.match(pairs if added else baseline, truth, mask)}


def summarize(rows):
    table = {}
    groups = {role: set() for role in ROLES}
    extra, repeated = set(), set()
    wrong = 0
    for row in rows:
        population, slot, quality, role = row["key"]
        counts = table.setdefault(f"{population}/{quality}/{role}", Counter())
        counts.update(views=1, current_useful=row["current_useful"], candidate_useful=row["candidate_useful"],
                      combined_useful=row["combined_useful"], added=row["added"],
                      current_flag=bool(row["current"]), candidate_flag=bool(row["candidate"]),
                      assessed=row["candidate_status"] in {"no_pair", "supported_pair"})
        if row["added"]:
            if role in ROLES:
                if row["combined_useful"]:
                    groups[role].add(slot)
                else:
                    wrong += 1
            elif population == "repetition":
                repeated.add(slot)
            else:
                extra.add(slot)
    outcome = {"added_small_source_groups": len(groups["small"]), "added_medium_source_groups": len(groups["medium"]),
               "added_ordinary_source_groups": len(extra), "added_repeat_source_groups": len(repeated),
               "wrong_added_edit_views": wrong}
    return {"views": len(rows), "passed": all(outcome[key.removeprefix("minimum_")] >= value
            if key.startswith("minimum_") else outcome[key.removeprefix("maximum_")] <= value for key, value in GATE.items()),
            **outcome, "strata": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("translation", type=Path)
    parser.add_argument("controls", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    inputs = args.cache, args.prior, args.translation, args.controls
    plan = args.run / "freeze.json"
    if args.action == "freeze":
        args.run.mkdir(mode=0o700)
        write(plan, freeze(inputs))
        print(sha(plan))
        return
    frozen = read(plan)
    require(json_ready(freeze(inputs)) == frozen, "Code, truth or sources differ from the pre-score freeze")
    started = time.monotonic()
    rows = []
    for index, item in enumerate(measure(inputs, frozen), 1):
        rows.append(item)
        if index % 30 == 0:
            print(f"{index}/{len(frozen['rows'])} views measured in {time.monotonic() - started:.1f} seconds", flush=True)
    path = args.run / "rows.json"
    if args.action == "review":
        require(json_ready(rows) == read(path), "Replayed measurements differ")
    else:
        require(not path.exists(), "Refusing to overwrite prior results")
        write(path, rows)
    print(summarize(rows), flush=True)


def json_ready(value):
    import json

    return json.loads(json.dumps(value))


if __name__ == "__main__":
    main()
