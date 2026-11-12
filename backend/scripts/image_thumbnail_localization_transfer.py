# SPDX-License-Identifier: Apache-2.0
"""Score single-file thumbnail localization with exact masks and untouched photos."""

from __future__ import annotations

import argparse
import hashlib
import io
import struct
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.provenance import exiftool, image_thumbnail_localization as candidate, metadata_rules
from scripts import image_reference_embedded_screenshot_transfer as sources
from scripts import image_reference_orientation_transfer as commons
from scripts import image_thumbnail_authentic_intake as intake

ROOT = Path("backend/data/independent-evaluation")
SCHEMA = "chipmunk.image-thumbnail-localization-transfer/1"
SCOPE = {"generated_groups": 12, "minimum_authentic_groups": 5,
         "minimum_aligned_authentic_originals": 4, "minimum_useful_generated": 6,
         "minimum_useful_local_color": 8, "minimum_color_adds": 3, "maximum_authentic_highlights": 0,
         "maximum_incompatible_comparisons": 0, "maximum_clean_highlights": 0}
CONFIRMATION = {"lama": 12, "sdxl": 3, "kandinsky": 3, "minimum_useful_lama": 6,
                "minimum_useful_alternate": 3, "minimum_useful_local_color": 12, "minimum_color_adds": 6,
                "maximum_clean_control_groups": 1, "maximum_unchanged_highlights": 0}
REQUIRE, read, write, sha, file_sha = sources.require, sources.read, sources.write, sources.sha, sources.file_sha


def exif_bytes(thumbnail):
    # Little-endian TIFF with orientation in IFD0 and the exact JPEG byte range in IFD1.
    start = 8 + 2 + 12 + 4
    offset = start + 2 + 3 * 12 + 4
    def entry(tag, kind, value):
        return struct.pack("<HHII", tag, kind, 1, value)
    return (b"Exif\0\0II\x2a\0" + struct.pack("<IH", 8, 1) + entry(274, 3, 1) + struct.pack("<IH", start, 3)
            + entry(0x0103, 3, 6) + entry(0x0201, 4, offset) + entry(0x0202, 4, len(thumbnail))
            + b"\0\0\0\0" + thumbnail)


def jpeg(image, quality=90, preview=None):
    args = {} if preview is None else {"exif": exif_bytes(preview)}
    with io.BytesIO() as out:
        image.save(out, "JPEG", quality=quality, subsampling=2, **args)
        return out.getvalue()


def thumbnail(image, edge=192):
    factor = edge / max(image.size)
    shape = tuple(round(n * factor) for n in image.size)
    return jpeg(image.resize(shape, Image.Resampling.LANCZOS), quality=76)


def preview_from(body):
    with Image.open(io.BytesIO(body)) as image:
        return exiftool._pillow_thumbnail(image, image.getexif())


def color_patch(pixels, slot):
    height, width = pixels.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_x, center_y = (.31 + .17 * (slot % 3)) * width, (.37 + .15 * (slot % 2)) * height
    radius = ((x - center_x) / (.11 * width)) ** 2 + ((y - center_y) / (.095 * height)) ** 2
    alpha = np.clip((1. - radius) / .25, 0., 1.)
    base = pixels.astype(np.float32)
    # luma-weighted mean of the three shifts is within one tenth of a JPEG code value.
    delta = np.array([30., -8., -37.5], dtype=np.float32)
    if slot % 2:
        delta *= -1.
    allowed = np.where(delta > 0, (255 - base) / np.abs(delta), base / np.abs(delta)).min(axis=2)
    return np.rint(base + delta * np.minimum(alpha, allowed)[:, :, None]).astype(np.uint8)


def qualified(args):
    native = sources.earlier_photos(argparse.Namespace(confirmation=not args.confirmation, backup=args.backup,
                                                        primary=args.primary, legacy=args.legacy))
    roster = [("lama", source, original, edit) for source, original, edit in native]
    REQUIRE(len(roster) == SCOPE["generated_groups"], "Unexpected number of masked LaMa edit source groups")
    if args.confirmation:
        alternate = args.alternate
        REQUIRE(alternate is not None, "Separate SDXL/Kandinsky inputs are needed for confirmation")
        binding = sources.prior.BASE.REGIONAL
        credits = {row["slot"]: row for row in read(alternate / "plan.json")["cases"]}
        candidates = list(binding.paired.inputs(alternate))
        for family in ("sdxl", "kandinsky"):
            selected = sorted((row for row in candidates if row[0]["generator"] == family), key=lambda row: row[0]["slot"])[:CONFIRMATION[family]]
            REQUIRE(len(selected) == CONFIRMATION[family], "Incomplete alternate-generator roster")
            for result, pixels, _ in selected:
                variants = binding.native_variants(pixels, result["box"])
                roster.append((family, credits[result["slot"]], variants["original"], variants["feathered_edit"]))
        development = list(sources.earlier_photos(argparse.Namespace(confirmation=True, backup=args.backup)))
        for key in ("flickr_owner", "flickr_photo_id"):
            values = [str(row[key]).casefold() for _, row, _, _ in roster] + [str(row[key]).casefold() for row, _, _ in development]
            REQUIRE(len(set(values)) == len(values), "Confirmation and pilot source groups or contributors overlap")
        rgb = [sha(pixels.tobytes()) for _, _, pixels, _ in roster] + [sha(pixels.tobytes()) for _, pixels, _ in development]
        REQUIRE(len(set(rgb)) == len(rgb), "Confirmation and pilot exact photo pixels overlap")
    return roster


def generated(args):
    roster = qualified(args)
    for slot, (family, source, earlier, edited) in enumerate(roster):
        original = Image.fromarray(earlier)
        thumb = thumbnail(original, (144, 168, 192)[slot % 3])
        colored = color_patch(earlier, slot)
        tone = np.rint(np.clip(earlier.astype(np.float32) * .91 + 12, 0, 255)).astype(np.uint8)
        donor = Image.fromarray(roster[(slot + 1) % len(roster)][2]).resize(original.size)
        w, h = original.size
        zoom = original.crop((round(w * .08), round(h * .08), round(w * .94), round(h * .94))).resize(original.size)
        resample = original.resize((round(w * .66), round(h * .66)), Image.Resampling.LANCZOS).resize(original.size, Image.Resampling.BICUBIC)
        claims = (("generated", Image.fromarray(edited), thumb, 90, edited),
                  ("local_chroma", Image.fromarray(colored), thumb, 90, colored),
                  ("unchanged", original, thumb, 90, None), ("jpeg75", original, thumb, 75, None),
                  ("global_tone", Image.fromarray(tone), thumb, 90, None), ("resampled", resample, thumb, 90, None),
                  ("foreign", original, thumbnail(donor, (144, 168, 192)[slot % 3]), 90, None),
                  ("zoomed", zoom, thumb, 90, None), ("missing", original, None, 90, None))
        for role, image, embedded, quality, changed in claims:
            data = jpeg(image, quality, embedded)
            truth = None if changed is None else np.any(earlier != changed, axis=2)
            REQUIRE(preview_from(data) == embedded, "The actual embedded thumbnail differs from the fixture plan")
            yield (family if args.confirmation else "generated", slot, role), data, truth, source["flickr_photo_id"]


def genuine_inputs(args):
    qualified = read(args.authentic / "qualified.json")["rows"]
    for row in qualified:
        path = args.authentic / "objects" / f"{row['sha256']}.jpg"
        data = path.read_bytes()
        REQUIRE(sha(data) == row["sha256"] and hashlib.sha1(data).hexdigest() == row["source_sha1"]
                and row["license"] in intake.LICENSES and sha(preview_from(data)) == row["thumbnail_sha256"], "Public original binding changed")
        yield row["author"], data, {key: row[key] for key in ("title", "author", "license", "page", "url", "source_sha1", "sha256", "camera_model")}
    reference = commons.binding(args.commons)
    roster = read(args.commons / "plan.json")["sources"]
    deliveries = read(args.commons / "acquisition.json")["attempts"]
    for accepted, listed, delivery in zip(reference["sources"], roster, deliveries, strict=True):
        if delivery["delivery"] != "original":
            continue
        data = (args.commons / accepted["file"]).read_bytes()
        if preview_from(data) is not None:
            yield listed["author"], data, listed | {"sha256": accepted["sha256"]}


def genuine(args):
    roster = list(genuine_inputs(args))
    authors = {author.casefold() for author, _, _ in roster}
    REQUIRE(len(roster) == len(authors) >= SCOPE["minimum_authentic_groups"], "Too few distinct genuinely embedded thumbnails")
    for slot, (author, body, _) in enumerate(roster):
        old_thumb = preview_from(body)
        with Image.open(io.BytesIO(body)) as original:
            original.draft("RGB", (1200, 800))
            copy = original.convert("RGB")
        yield ("authentic", slot, "untouched"), body, None, author
        yield ("authentic", slot, "jpeg75"), jpeg(copy, 75, old_thumb), None, author
        if args.confirmation:
            values = np.asarray(copy, dtype=np.float32)
            balanced = Image.fromarray(np.rint(np.clip(values * (1.07, .98, .94) + (2, 4, 6), 0, 255)).astype(np.uint8))
            yield ("authentic", slot, "white_balance"), jpeg(balanced, 85, old_thumb), None, author


def cases(args):
    yield from generated(args)
    yield from genuine(args)


def case_binding(case):
    key, body, truth, source = case
    return {"case": list(key), "source_sha256": sha(str(source).encode()), "input_sha256": sha(body),
            "truth_sha256": None if truth is None else sha(truth.tobytes()), "target": 0 if truth is None else int(truth.sum())}


def freeze(args):
    bindings = [case_binding(case) for case in cases(args)]
    credits = [credit for _, _, credit in genuine_inputs(args)]
    code = (Path(__file__), Path(candidate.__file__), Path(intake.__file__), Path(metadata_rules.__file__), Path(exiftool.__file__))
    return {"schema": SCHEMA, "gate": SCOPE, "records": bindings, "genuine_credits": credits,
            "confirmation_gate": CONFIRMATION if args.confirmation else None,
            "minimum_component_fraction": component_floor(args),
            "development_commitment": file_sha(args.development / "freeze.json") if args.confirmation else None,
            "code": {p.name: file_sha(p) for p in code}, "confidence_weight": 0}


def area(mask, truth):
    if mask is None or truth is None:
        return {"selected_thumb": 0 if mask is None else int(mask.sum()), "native_overlap": 0, "selected_native": None, "useful": False}
    h, w = truth.shape
    y, x = (np.arange(n) * m // n for n, m in zip((h, w), mask.shape, strict=True))
    expanded = mask[y[:, None], x]
    selected, target, overlap = (int(a.sum()) for a in (expanded, truth, expanded & truth))
    useful = bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (selected + target - overlap))
    return {"selected_thumb": int(mask.sum()), "selected_native": selected, "native_overlap": overlap, "useful": useful}


def component_floor(args):
    return candidate.CONFIRMATION_COMPONENT_FRACTION if args.confirmation else candidate.PILOT_COMPONENT_FRACTION


def observe(case, floor):
    key, body, truth, _ = case
    start = time.monotonic()
    result, masks = candidate.measure(body, minimum_component_fraction=floor)
    elapsed = time.monotonic() - start
    old_thumb = preview_from(body)
    prior = None if old_thumb is None else metadata_rules.compare_thumbnail(old_thumb, body)
    current = None if prior is None else prior.ssim
    combined, luma = (None, None) if masks is None else masks
    return {"case": list(key), "result": result, "combined": area(combined, truth), "luma": area(luma, truth),
            "mask_sha256": None if combined is None else sha(combined.tobytes()),
            "luma_mask_sha256": None if luma is None else sha(luma.tobytes()), "incumbent_global_gray_ssim": current,
            "elapsed_seconds": round(elapsed, 6)}


def summarize(rows, confirmation=False):
    roles = defaultdict(lambda: dict(views=0, comparable=0, localized=0, useful=0, luma_useful=0, global_anomalies=0, color_added=0))
    for row in rows:
        panel, _, role = row["case"]
        tally = roles[f"{panel}/{role}"]
        tally["views"] += 1
        tally["comparable"] += row["result"]["status"] == "comparable"
        tally["localized"] += row["combined"]["selected_thumb"] > 0
        tally["useful"] += row["combined"]["useful"]
        tally["luma_useful"] += row["luma"]["useful"]
        score = row["incumbent_global_gray_ssim"]
        tally["global_anomalies"] += score is not None and score < metadata_rules.SSIM_ANOMALY
        tally["color_added"] += row["combined"]["useful"] and not row["luma"]["useful"]
    authentic = sum(tally["localized"] for key, tally in roles.items() if key.startswith("authentic/"))
    panels = ("lama", "sdxl", "kandinsky") if confirmation else ("generated",)
    incompatible = sum(roles[f"{panel}/{role}"]["comparable"] for panel in panels for role in ("missing", "foreign", "zoomed"))
    ordinary_roles = ("unchanged", "jpeg75", "global_tone", "resampled")
    clean = sum(roles[f"{panel}/{role}"]["localized"] for panel in panels for role in ordinary_roles)
    clean_groups = len({tuple(row["case"][:2]) for row in rows if row["case"][0] in panels and row["case"][2] in ordinary_roles
                        and row["combined"]["selected_thumb"]})
    safe = (authentic <= SCOPE["maximum_authentic_highlights"] and incompatible <= SCOPE["maximum_incompatible_comparisons"]
            and roles["authentic/untouched"]["comparable"] >= SCOPE["minimum_aligned_authentic_originals"])
    if confirmation:
        unchanged = sum(roles[f"{panel}/unchanged"]["localized"] for panel in panels)
        passed = (safe and clean_groups <= CONFIRMATION["maximum_clean_control_groups"]
                  and unchanged <= CONFIRMATION["maximum_unchanged_highlights"]
                  and roles["lama/generated"]["useful"] >= CONFIRMATION["minimum_useful_lama"]
                  and sum(roles[f"{panel}/generated"]["useful"] for panel in panels[1:]) >= CONFIRMATION["minimum_useful_alternate"]
                  and sum(roles[f"{panel}/local_chroma"]["useful"] for panel in panels) >= CONFIRMATION["minimum_useful_local_color"]
                  and sum(roles[f"{panel}/local_chroma"]["color_added"] for panel in panels) >= CONFIRMATION["minimum_color_adds"])
    else:
        passed = (safe and clean <= SCOPE["maximum_clean_highlights"]
                  and roles["generated/generated"]["useful"] >= SCOPE["minimum_useful_generated"]
                  and roles["generated/local_chroma"]["useful"] >= SCOPE["minimum_useful_local_color"]
                  and roles["generated/local_chroma"]["color_added"] >= SCOPE["minimum_color_adds"])
    runtime = sorted(row["elapsed_seconds"] for row in rows)
    return {"roles": dict(roles), "genuine_highlights": authentic, "clean_highlights": clean, "clean_groups": clean_groups,
            "incompatible_comparisons": incompatible,
            "median_seconds": runtime[len(runtime) // 2], "max_seconds": runtime[-1], "passed": passed}


def parser():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("stage", choices=("freeze", "run", "review"))
    cmd.add_argument("run", type=Path)
    cmd.add_argument("--authentic", type=Path, default=ROOT / "image-authentic-thumbnails-2026-11-12")
    cmd.add_argument("--commons", type=Path, default=ROOT / "image-affine-native-scale-2026-11-11")
    cmd.add_argument("--backup", type=Path, default=ROOT / "image-bg-real-backup-intake-v262")
    cmd.add_argument("--primary", type=Path, default=ROOT / "image-bg-real-acquisition-v225")
    cmd.add_argument("--legacy", type=Path, default=ROOT / "image-residual-patch-held-2026-11-11")
    cmd.add_argument("--development", type=Path, default=ROOT / "image-thumbnail-localization-2026-11-12")
    cmd.add_argument("--confirmation", action="store_true")
    cmd.add_argument("--alternate", type=Path)
    return cmd


def main():
    args = parser().parse_args()
    cv2.setNumThreads(1)
    frozen = args.run / "freeze.json"
    if args.stage == "freeze":
        record = freeze(args)
        args.run.mkdir(mode=0o700)
        write(frozen, record)
        print(file_sha(frozen), flush=True)
        return
    REQUIRE(freeze(args) == read(frozen), "Frozen code, source or derived asset changed")
    held = args.run / "held.json"
    receipt = args.run / "result.json"
    if args.stage == "review":
        from scripts import review_image_thumbnail_localization_transfer as reviewer
        review = reviewer.verify(args, read(held), read(receipt))
        write(args.run / "native-review.json", review)
        print(review, flush=True)
        return
    rows = [observe(case, component_floor(args)) for case in cases(args)]
    write(held, rows)
    report = summarize(rows, args.confirmation)
    write(receipt, {"schema": SCHEMA, "freeze_sha256": file_sha(frozen), "held_sha256": file_sha(held), "summary": report})
    print(report, flush=True)


if __name__ == "__main__":
    main()
