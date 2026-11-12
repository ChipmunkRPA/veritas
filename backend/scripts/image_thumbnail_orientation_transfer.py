# SPDX-License-Identifier: Apache-2.0
"""Qualify the embedded JPEG preview comparison for common EXIF rotations."""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.evidence import image_thumbnail_evidence as evidence
from app.provenance import image_thumbnail_localization as local
from scripts import image_thumbnail_localization_transfer as prior

SCHEMA = "chipmunk.image-thumbnail-orientation-transfer/1"
ORIENTATIONS = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}
INVERSE = {3: 3, 6: 8, 8: 6}
CONVENTIONS = ("same_raw_axes", "preview_display_upright")
GATE = {"minimum_localized": 48, "minimum_per_orientation_convention": 6, "maximum_clean_groups": 1,
        "maximum_new_genuine_highlights": 0, "maximum_wrong_correspondence": 0}
ROOT = Path("backend/data/independent-evaluation")
write, read, sha, require = prior.write, prior.read, prior.sha, prior.REQUIRE


def orient_pixels(preview, native, operation):
    oriented = np.asarray(preview.transpose(ORIENTATIONS[operation]) if operation else preview)
    h, w = oriented.shape[:2]
    aspect = native[0] / native[1]
    if abs((w / h) / aspect - 1) <= local.MAX_ASPECT_DELTA:
        return oriented, (0, 0, w, h)
    cropped, trim = local._trim(oriented)
    h, w = cropped.shape[:2]
    if min(w, h) >= local.MIN_SIDE and abs((w / h) / aspect - 1) <= local.MAX_ASPECT_DELTA:
        return cropped, trim
    return None


def to_raw_preview(box, size, operation):
    width, height = size
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    if operation == 3:
        return {"x": width - x - w, "y": height - y - h, "w": w, "h": h}
    if operation == 6:
        return {"x": y, "y": height - x - w, "w": h, "h": w}
    if operation == 8:
        return {"x": width - y - h, "y": x, "w": h, "h": w}
    return box


def _candidate(main, native, preview, operation):
    adjusted = orient_pixels(preview, native, operation)
    if adjusted is None:
        return None
    before, trim = adjusted
    h, w = before.shape[:2]
    supported, support = [], []
    for kernel in local.KERNELS:
        after = np.asarray(main.resize((w, h), kernel))
        alignment = local._align(before, after)
        if alignment is None:
            return None
        registered, shift = alignment
        luma, chroma, detail = local._differences(registered, after)
        if np.mean(luma[3:-3, 3:-3] <= 12.) < local.MIN_BACKGROUND:
            return None
        bright, _ = local._over(luma, detail, 9.)
        color, _ = local._over(chroma, detail, 10.)
        supported.append(bright | color)
        support.append({"shift_and_response": shift})
    regions, mask = local._mask(supported[0] & supported[1], local.CONFIRMATION_COMPONENT_FRACTION)
    public = []
    for region in local._mapped(regions, trim, (w, h), native):
        aligned_box = evidence._preview_bounds(region, {"support": support, "trim": trim})
        public.append({"main_image": region["submitted"], "aligned_preview": aligned_box,
                       "raw_preview": to_raw_preview(aligned_box, preview.size, operation)})
    return {"operation": operation, "native_size": native, "preview_size": preview.size, "regions": public}, mask


def measure(data):
    with Image.open(io.BytesIO(data)) as main:
        raw = main.size
        orientation = main.getexif().get(274, 1)
        if (main.format != "JPEG" or orientation not in ORIENTATIONS or min(raw) < 160
                or raw[0] * raw[1] > local.MAX_MAIN_PIXELS):
            return None
        blob = local._pillow_thumbnail(main, main.getexif())
        if blob is None:
            return None
        with Image.open(io.BytesIO(blob)) as thumbnail:
            if thumbnail.format != "JPEG" or min(thumbnail.size) < local.MIN_SIDE or max(thumbnail.size) > local.MAX_SIDE:
                return None
            preview = thumbnail.convert("RGB")
        # Draft at the retained working resolution on the camera's stored axes, then rotate once for display.
        native = raw if orientation == 3 else raw[::-1]
        edge = max(preview.size)
        main.draft("RGB", tuple(round(3 * edge * n / max(raw)) for n in raw))
        display = main.convert("RGB").transpose(ORIENTATIONS[orientation])
    candidates = [match for operation in (0, orientation) if (match := _candidate(display, native, preview, operation)) is not None]
    return candidates[0] if len(candidates) == 1 else None


def encode(main, orientation, preview):
    opts = {}
    if preview is not None:
        original = prior.exif_bytes(preview)
        opts["exif"] = original[:24] + struct.pack("<H", orientation) + original[26:]
    else:
        exif = Image.Exif()
        exif[274] = orientation
        opts["exif"] = exif
    with io.BytesIO() as out:
        main.save(out, "JPEG", quality=90, subsampling=2, **opts)
        body = out.getvalue()
    with Image.open(io.BytesIO(body)) as carrier:
        require(carrier.getexif().get(274) == orientation and local._pillow_thumbnail(carrier, carrier.getexif()) == preview,
                "The exact preview and main orientation were not retained")
    return body


def edit_photos(args):
    native = prior.sources.prior.old.primary_sources(args)
    groups = []
    for slot, _, original, edited, known in native:
        feathered = prior.sources.prior.BASE.feather(original, edited, known)
        groups.append((slot, Image.fromarray(original), Image.fromarray(feathered), np.any(original != feathered, axis=2)))
    require(len(groups) == 12, "Expected 12 disjoint primary photo groups")
    return groups


def cases(args):
    groups = edit_photos(args)
    for slot, clean, edit, truth in groups:
        thumb_edge = (144, 168, 192)[slot % 3]
        upright_thumb = prior.thumbnail(clean, thumb_edge)
        donor = groups[(slot + 1) % len(groups)][1].resize(clean.size)
        yield ("edit", slot, 1, "upright", "edited"), prior.jpeg(edit, preview=upright_thumb), truth, 0
        for orientation, operation in INVERSE.items():
            raw = ORIENTATIONS[operation]
            rotated_thumb = prior.thumbnail(clean.transpose(raw), thumb_edge)
            for convention, preview in zip(CONVENTIONS, (rotated_thumb, upright_thumb), strict=True):
                expected = orientation if convention == CONVENTIONS[0] else 0
                for role, picture in (("clean", clean), ("edited", edit)):
                    yield (("edit", slot, orientation, convention, role), encode(picture.transpose(raw), orientation, preview),
                           truth if role == "edited" else None, expected)
            yield ("edit", slot, orientation, "mismatch", "foreign"), encode(clean.transpose(raw), orientation, prior.thumbnail(donor, thumb_edge)), None, None
            yield ("edit", slot, orientation, "missing", "no_preview"), encode(clean.transpose(raw), orientation, None), None, None
    for slot, (_, genuine, _) in enumerate(prior.genuine_inputs(args)):
        retained = prior.preview_from(genuine)
        with Image.open(io.BytesIO(retained)) as thumbnail, Image.open(io.BytesIO(genuine)) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")
            photo.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            for orientation, operation in INVERSE.items():
                raw = ORIENTATIONS[operation]
                rotated = prior.jpeg(thumbnail.transpose(raw), quality=90)
                for convention, preview in zip(CONVENTIONS, (rotated, retained), strict=True):
                    expected = orientation if convention == CONVENTIONS[0] else 0
                    yield ("genuine", slot, orientation, convention, "clean"), encode(photo.transpose(raw), orientation, preview), None, expected


def freeze(args):
    records = [{"key": key, "file": sha(data), "truth": sha(mask.tobytes()) if mask is not None else None, "operation": expected}
               for key, data, mask, expected in cases(args)]
    return {"schema": SCHEMA, "confidence_weight": 0, "criteria": GATE,
            "code": {p.name: sha(p.read_bytes()) for p in (Path(__file__), Path(prior.__file__), Path(local.__file__), Path(evidence.__file__))},
            "records": records}


def audit_raw_boxes(record):
    width, height = record["preview_size"]
    values = np.arange(width * height, dtype=np.int32).reshape(height, width)
    canvas = Image.fromarray(values)
    if record["operation"]:
        canvas = canvas.transpose(ORIENTATIONS[record["operation"]])
    for region in record["regions"]:
        x, y, w, h = (region["aligned_preview"][key] for key in ("x", "y", "w", "h"))
        a, b, m, n = (region["raw_preview"][key] for key in ("x", "y", "w", "h"))
        assert np.array_equal(np.sort(np.asarray(canvas)[y:y+h, x:x+w].ravel()), np.sort(values[b:b+n, a:a+m].ravel()))


def summarize(rows):
    counts = {}
    for orientation in ORIENTATIONS:
        for convention in CONVENTIONS:
            branch = [row for row in rows if row["key"][2:4] == [orientation, convention]]
            counts[f"{orientation}/{convention}"] = {"useful_edits": sum(row["useful"] for row in branch if row["key"][-1] == "edited"),
                                                       "clean_flags": sum(row["has_region"] for row in branch if row["key"][-1] == "clean"),
                                                       "clean_comparable": sum(row["compared"] for row in branch if row["key"][-1] == "clean")}
    total = sum(value["useful_edits"] for value in counts.values())
    clean = {row["key"][1] for row in rows if row["key"][0] == "edit" and row["key"][-1] == "clean" and row["has_region"]}
    wrong = sum(row["wrong_correspondence"] for row in rows)
    genuine = sum(row["has_region"] for row in rows if row["key"][0] == "genuine")
    passed = (total >= GATE["minimum_localized"] and min(value["useful_edits"] for value in counts.values()) >= GATE["minimum_per_orientation_convention"]
              and len(clean) <= GATE["maximum_clean_groups"] and genuine <= GATE["maximum_new_genuine_highlights"] and wrong <= GATE["maximum_wrong_correspondence"])
    return {"branches": counts, "useful_rotated_total": total, "useful_upright": sum(row["useful"] for row in rows if row["key"][2] == 1),
            "authored_clean_groups_flagged": len(clean), "genuine_derived_views_flagged": genuine,
            "wrong_correspondences": wrong, "passed": passed}


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("stage", choices=("freeze", "held"))
    cmd.add_argument("run", type=Path)
    for key, value in {"primary": "image-bg-real-acquisition-v225", "backup": "image-bg-real-backup-intake-v262",
                       "legacy": "image-residual-patch-held-2026-11-11", "commons": "image-affine-native-scale-2026-11-11",
                       "authentic": "image-authentic-thumbnails-2026-11-12"}.items():
        cmd.add_argument(f"--{key}", type=Path, default=ROOT / value)
    args = cmd.parse_args()
    cv2.setNumThreads(1)
    frozen = args.run / "freeze.json"
    plan = freeze(args)
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, plan)
        print(sha(frozen.read_bytes()), len(plan["records"]), flush=True)
        return
    require(read(frozen) == json.loads(json.dumps(plan)), "Frozen source or transformation changed")
    rows = []
    runtime_checks = 0
    mapped_regions = 0
    for key, data, truth, expected in cases(args):
        if key[2] == 1:
            record, masks = local.measure(data, minimum_component_fraction=local.CONFIRMATION_COMPONENT_FRACTION)
            mask = masks[0] if masks is not None else None
            found, wrong = mask is not None, False
        elif match := measure(data):
            record, mask = match
            found, wrong = True, record["operation"] != expected
            audit_raw_boxes(record)
        else:
            record, mask = None, None
            found, wrong = False, False
        measured, runtime = local.measure(data, minimum_component_fraction=local.CONFIRMATION_COMPONENT_FRACTION)
        require((runtime is not None) == found, "Runtime eligibility differs from the frozen orientation check")
        if runtime is not None:
            require(np.array_equal(runtime[0], mask), "Runtime mask differs from the frozen orientation check")
            if key[2] != 1:
                require(measured["main_orientation"] == key[2] and measured["preview_orientation"] == record["operation"],
                        "Runtime orientation differs from the frozen orientation check")
                for matched, candidate in zip(record["regions"], measured["regions"], strict=True):
                    require(matched["main_image"] == candidate["submitted"]
                            and matched["raw_preview"] == evidence._stored_preview_bounds(candidate, measured),
                            "Runtime native crop coordinates differ from independently replayed orientation bounds")
                    mapped_regions += 1
        runtime_checks += 1
        rows.append({"key": list(key), "compared": found, "wrong_correspondence": wrong,
                     "has_region": bool(mask is not None and mask.any()), "record": record,
                     "useful": prior.area(mask, truth)["useful"], "mask": sha(mask.tobytes()) if mask is not None else None})
    summary = summarize(rows)
    write(args.run / "held.json", {"freeze": sha(frozen.read_bytes()), "rows": rows, "summary": summary,
                                  "runtime_checks": runtime_checks, "mapped_regions_checked": mapped_regions})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
