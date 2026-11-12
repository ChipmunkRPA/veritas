# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay mirrored EXIF thumbnail deliveries at their exact source coordinates."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.evidence import image_thumbnail_evidence as evidence
from app.provenance import image_thumbnail_localization as detector
from scripts import image_thumbnail_localization_transfer as prior
from scripts import image_thumbnail_orientation_transfer as rotation

SCHEMA = "chipmunk.image-thumbnail-mirror-transfer/1"
OPERATIONS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 4: Image.Transpose.FLIP_TOP_BOTTOM,
              5: Image.Transpose.TRANSPOSE, 7: Image.Transpose.TRANSVERSE}
GATE = {"minimum_useful_per_branch": 7, "minimum_upright_useful_retained_numerator": 3,
        "minimum_upright_useful_retained_denominator": 4, "maximum_unedited_flags": 0,
        "maximum_genuine_flags": 0, "maximum_other_clean_groups": 1, "maximum_wrong_matches": 0}
CONVENTIONS = ("raw_axes", "display_axes")
ROOT = prior.ROOT
require, read, write, sha = prior.REQUIRE, prior.read, prior.write, prior.sha


def flux_sources(path):
    sealed = read(path / "summary.json")
    require(sealed["status"] == "complete" and sealed["source_groups"] == 6, "Incomplete edit archive")
    for batch in range(1, 4):
        output = path / f"batch-{batch:02}" / "authored" / "output"
        receipt = output / "result.json"
        require(sha(receipt.read_bytes()) == sealed["batch_receipts"][str(batch)]["result_sha256"], "Edit receipt changed")
        for case in read(receipt)["cases"]:
            name = case["slot"]
            original = (path / name / "source.png").read_bytes()
            edited = (output / name / "composite.png").read_bytes()
            require(sha(edited) == case["composite_png_sha256"], "Edited file changed")
            before, after = (Image.open(io.BytesIO(data)).convert("RGB") for data in (original, edited))
            require(sha(before.tobytes()) == case["source_rgb_sha256"]
                    and sha(after.tobytes()) == case["composite_rgb_sha256"], "Edited RGB changed")
            yield name, before, after, np.any(np.asarray(before) != np.asarray(after), axis=2)


def cases(args):
    families = {"feathered": list(rotation.edit_photos(args)), "compact": list(flux_sources(args.flux))}
    identities = {sha(before.tobytes()) for sources in families.values() for _, before, _, _ in sources}
    require(len(identities) == 18, "Expected 18 distinct original RGB identities")
    for family, sources in families.items():
        for slot, (name, before, after, truth) in enumerate(sources):
            edge = (144, 168, 192)[slot % 3]
            upright = prior.thumbnail(before, edge)
            yield [family, name, 1, "display_axes", "edit"], prior.jpeg(after, preview=upright), truth
            tone = Image.fromarray(np.rint(np.clip(np.asarray(before).astype(np.float32) * .91 + 12, 0, 255)).astype(np.uint8))
            foreign = prior.thumbnail(sources[(slot + 1) % len(sources)][1].resize(before.size), edge)
            for tag, operation in OPERATIONS.items():
                raw = prior.thumbnail(before.transpose(operation), edge)
                for convention, thumbnail in zip(CONVENTIONS, (raw, upright), strict=True):
                    for role, image in (("edit", after), ("unedited", before), ("global_tone", tone)):
                        key = [family, name, tag, convention, role]
                        yield key, rotation.encode(image.transpose(operation), tag, thumbnail), truth if role == "edit" else None
                for role, thumbnail in (("foreign", foreign), ("missing", None)):
                    yield [family, name, tag, "none", role], rotation.encode(before.transpose(operation), tag, thumbnail), None
    for slot, (_, original, _) in enumerate(prior.genuine_inputs(args)):
        preview = prior.preview_from(original)
        with Image.open(io.BytesIO(preview)) as stored, Image.open(io.BytesIO(original)) as main:
            display = ImageOps.exif_transpose(main).convert("RGB")
            display.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            for tag, operation in OPERATIONS.items():
                raw = prior.jpeg(stored.transpose(operation), quality=90)
                for convention, thumbnail in zip(CONVENTIONS, (raw, preview), strict=True):
                    yield ["genuine", slot, tag, convention, "unedited"], rotation.encode(display.transpose(operation), tag, thumbnail), None


def plan(args):
    rows = [{"key": key, "sha256": sha(data), "truth_sha256": None if truth is None else sha(truth.tobytes())}
            for key, data, truth in cases(args)]
    modules = (Path(__file__), Path(prior.__file__), Path(rotation.__file__))
    return {"schema": SCHEMA, "criteria": GATE, "confidence_weight": 0,
            "code": {p.name: sha(p.read_bytes()) for p in modules}, "records": rows}


def pixel_bounds(region, size, tag):
    width, height = size
    pixels = np.arange(width * height, dtype=np.int32).reshape(height, width)
    labelled = Image.fromarray(pixels)
    if tag:
        labelled = labelled.transpose(OPERATIONS[tag])
    x, y, w, h = (region[key] for key in evidence.BOX_KEYS)
    selected = np.asarray(labelled)[y:y + h, x:x + w].ravel()
    ys, xs = np.divmod(selected, width)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    require(np.array_equal(np.sort(selected), pixels[top:bottom, left:right].ravel()), "Preview pixel bounds changed")
    return dict(zip(evidence.BOX_KEYS, (left, top, right - left, bottom - top), strict=True))


def mirror(data, tag):
    preview = prior.preview_from(data)
    if preview is None:
        return None, None
    with Image.open(io.BytesIO(data)) as raw, Image.open(io.BytesIO(preview)) as stored:
        native = raw.size[::-1] if tag in (5, 7) else raw.size
        edge = max(stored.size)
        raw.draft("RGB", tuple(round(3 * edge * n / max(raw.size)) for n in raw.size))
        main = ImageOps.exif_transpose(raw).convert("RGB")
        before = stored.convert("RGB")
        matches = [(operation, match) for operation, image in ((0, before), (tag, before.transpose(OPERATIONS[tag])))
                   if (match := rotation._candidate(main, native, image, 0)) is not None]
        if len(matches) != 1:
            return None, None
        operation, (record, mask) = matches[0]
        return {"operation": operation, "regions": [{"main_image": region["main_image"],
                    "stored_preview": pixel_bounds(region["aligned_preview"], stored.size, operation)}
                    for region in record["regions"]]}, mask


def observe(case, *, runtime):
    key, data, truth = case
    _, _, tag, convention, role = key
    measure = detector.measure
    floor = detector.CONFIRMATION_COMPONENT_FRACTION
    if tag == 1:
        _, masks = measure(data, minimum_component_fraction=floor)
        mask, record = (None if masks is None else masks[0]), None
    else:
        record, mask = mirror(data, tag)
    regions_checked = 0
    if runtime and tag != 1:
        actual, masks = measure(data, minimum_component_fraction=floor)
        require((mask is None) == (masks is None), "Runtime eligibility disagrees with native audit")
        if masks is not None:
            require(np.array_equal(mask, masks[0]) and actual["main_orientation"] == tag
                    and actual["preview_orientation"] == record["operation"], "Runtime orientation or mask changed")
            for expected, region in zip(record["regions"], actual["regions"], strict=True):
                require(expected["main_image"] == region["submitted"]
                        and expected["stored_preview"] == evidence._stored_preview_bounds(region, actual), "Native crop bounds changed")
                regions_checked += 1
    expected_operation = {"raw_axes": tag, "display_axes": 0}.get(convention)
    return {"key": key, "comparison": mask is not None, "flag": bool(mask is not None and mask.any()),
            "wrong_match": bool(record is not None and record["operation"] != expected_operation),
            "overlap": prior.area(mask, truth), "target_pixels": 0 if truth is None else int(truth.sum()),
            "mask_sha256": None if mask is None else sha(mask.tobytes()), "regions_checked": regions_checked,
            "role": role}


def summarize(rows):
    upright = {(row["key"][0], row["key"][1]) for row in rows if row["key"][2] == 1 and row["overlap"]["useful"]}
    counts = Counter()
    branches = {}
    clean_groups = set()
    for row in rows:
        family, group, tag, convention, role = row["key"]
        if tag == 1:
            continue
        counts["wrong_matches"] += row["wrong_match"]
        counts["mapped_regions"] += row["regions_checked"]
        if row["flag"] and role != "edit":
            counts["clean_flags"] += 1
            if family == "genuine":
                counts["genuine_flags"] += 1
            elif role == "unedited":
                counts["unedited_flags"] += 1
            else:
                clean_groups.add((family, group))
        if role == "edit":
            item = branches.setdefault(f"{tag}/{convention}", Counter())
            item["useful"] += row["overlap"]["useful"]
            item["upright_retained"] += row["overlap"]["useful"] and (family, group) in upright
            item[family] += row["overlap"]["useful"]
    for key in ("wrong_matches", "mapped_regions", "clean_flags", "genuine_flags", "unedited_flags"):
        counts.setdefault(key, 0)
    passed = (all(item["useful"] >= GATE["minimum_useful_per_branch"]
                  and item["upright_retained"] * GATE["minimum_upright_useful_retained_denominator"]
                  >= len(upright) * GATE["minimum_upright_useful_retained_numerator"] for item in branches.values())
              and counts["wrong_matches"] <= GATE["maximum_wrong_matches"]
              and counts["genuine_flags"] <= GATE["maximum_genuine_flags"]
              and counts["unedited_flags"] <= GATE["maximum_unedited_flags"]
              and len(clean_groups) <= GATE["maximum_other_clean_groups"])
    return {"upright_useful": len(upright), "branches": branches, "counts": counts,
            "other_clean_groups_flagged": len(clean_groups), "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "held", "runtime"))
    parser.add_argument("run", type=Path)
    for key, value in {"primary": "image-bg-real-acquisition-v225", "backup": "image-bg-real-backup-intake-v262",
                       "legacy": "image-residual-patch-held-2026-11-11", "commons": "image-affine-native-scale-2026-11-11",
                       "authentic": "image-authentic-thumbnails-2026-11-12", "flux": "image-flux-photo-expansion-v285"}.items():
        parser.add_argument(f"--{key}", type=Path, default=ROOT / value)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frozen = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, plan(args))
        print(sha(frozen.read_bytes()), flush=True)
        return
    require(read(frozen) == json.loads(json.dumps(plan(args))), "Frozen inputs or test code changed")
    rows = [observe(case, runtime=args.stage == "runtime") for case in cases(args)]
    summary = summarize(rows)
    if args.stage == "runtime":
        held = read(args.run / "held.json")
        require([{**row, "regions_checked": 0} for row in rows] == held["rows"], "Runtime test changed frozen held results")
    write(args.run / f"{args.stage}.json", {"freeze_sha256": sha(frozen.read_bytes()), "summary": summary, "rows": rows})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
