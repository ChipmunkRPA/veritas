# SPDX-License-Identifier: Apache-2.0
"""Compare color continuity and confined periodicity on exact image boundary proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.detectors.image import sharp_boundary
from app.engine.fusion.content_kinds import flat_block_share
from scripts import image_boundary_context_cues as cues
from scripts import image_reference_orientation_transfer as sources
from scripts import image_small_boundary_control_stress as base
from scripts.image_small_boundary_screenshot_admission import save_new, sha

SCHEMA = "chipmunk.image-boundary-context-transfer/1"
SIGN_PANEL = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/image-small-boundary-control-stress-2026-11-12.json"
TARGETS = 12
SIGN_SLOTS = (0, 3, 5, 7)
CONTROLS = ("original", "global_tone", "feathered_blur", "resize_roundtrip")
POSITIVE = ("sharp_paste", "repeat_paste", "tint_edit", "feathered_paste")
PANEL_TYPES = ("opaque", "alpha", "periodic")
POLICY = {
    "minimum_incumbent_negative_screen_groups": 8,
    "minimum_broad_flags_per_challenge_type": 3,
    "minimum_extra_context_resolved_screen_groups": 4,
    "maximum_residual_screen_groups": 2,
    "maximum_lost_standard_pastes": 1,
    "maximum_lost_repeat_or_tint_edits": 0,
    "minimum_broad_repeat_and_tint_locations": 2,
    "maximum_new_ordinary_groups": 1,
    "previous_interior_flat_floor": .35,
}
require = base.require


def source_rows(archive):
    inventory = sources.binding(archive)["sources"]
    # Slots 7 and 19 have credited author aliases. Slot 27 depicts children.
    ranked = sorted((row for row in inventory if row["slot"] not in {19, 27}),
                    key=lambda row: base.digest(f"image-boundary-context/{row['sha256']}".encode()))
    sign_index = {row["slot"]: row for row in base.read(SIGN_PANEL)["freeze"]["roster"]["sources"]}
    return ranked, [sign_index[slot] for slot in SIGN_SLOTS]


def photo(archive, row, size):
    with Image.open(archive / row["file"]) as raw, ImageOps.exif_transpose(raw) as oriented:
        return ImageOps.fit(oriented.convert("RGB"), size, Image.Resampling.LANCZOS)


def prepare(archive, run):
    ordinary, _ = source_rows(archive)
    path = run / "screen-photos"
    path.mkdir()
    for slot, row in enumerate(ordinary[2 * TARGETS:]):
        photo(archive, row, (768, 544)).save(path / f"{slot}.jpg", quality=92, subsampling=0)


def freeze(archive, cache, run):
    roster, signs = source_rows(archive)
    for sign in signs:
        base.photo(cache, sign, (704, 544))
    plan = base.read(run / "capture-plan.json")
    captures = base.read(run / "captures.json")
    expected = {(group["slot"], view["type"]): (group, view) for group in plan["groups"] for view in group["views"]}
    rgb_hashes = {}
    for row in captures["rows"]:
        slot, panel = row["slot"], row["type"]
        group, view = expected.pop((slot, panel))
        require(row["size"] == group["size"] and row["caption"] == view["caption"] and row["filename"] == f"{slot}-{panel}.png"
                and row["rect"] == [view["x"], view["y"] + 38, view["x"] + view["width"], view["y"] + view["height"] + 38],
                "A captured view does not follow the predeclared layout")
        require(row["photo_sha256"] == sha(run / "screen-photos" / f"{slot}.jpg"), "The capture background changed")
        path = run / "captures" / row["filename"]
        require(sha(path) == row["png_sha256"], "Captured PNG bytes changed")
        with Image.open(path) as image:
            require(list(image.size) == row["size"] and sharp_boundary.supported_size(*image.size), "The native capture size changed")
            rgb_hashes[row["filename"]] = base.digest(image.convert("RGB").tobytes())
    require(not expected and len(set(rgb_hashes.values())) == 3 * TARGETS, "Complete unique capture set required")
    code = [Path(__file__), Path(__file__).with_name("capture_image_boundary_context.cjs"), Path(cues.__file__),
            Path(base.__file__), Path(sharp_boundary.__file__), Path(base.prior.__file__)]
    return {"schema": SCHEMA, "source_receipt_sha256": sha(sources.SOURCE), "sign_receipt_sha256": sha(SIGN_PANEL),
            "photo_roster": roster, "printed_signs": signs, "capture_plan": plan, "captures": captures,
            "capture_rgb_sha256": rgb_hashes, "alpha": cues.ALPHA, "periodic": cues.PERIOD, "criteria": POLICY,
            "fixed_point": sharp_boundary.REVIEW_POINT, "fallback_point": base.POINT,
            "minimum_side": base.fallback.MIN_SHORT_EDGE, "maximum_axis_fraction": base.fallback.MAX_AXIS_FRACTION,
            "code_sha256": {path.name: sha(path) for path in code}}


def incumbent(frozen, run):
    rows = []
    for record in frozen["captures"]["rows"]:
        with Image.open(run / "captures" / record["filename"]) as capture:
            result = sharp_boundary.measure(np.asarray(capture.convert("RGB")))
        rows.append({key: record[key] for key in ("slot", "type", "caption", "filename", "size")}
                    | {"score": result.score, "proposal": list(result.box)})
    return {"freeze_sha256": sha(run / "freeze.json"), "all_views": rows,
            "eligible": [[row["slot"], row["type"]] for row in rows if row["score"] <= frozen["fixed_point"]]}


def repeated_patch(donor, size, slot):
    width, height = size
    tw, th = (7, 5) if slot % 2 == 0 else (6, 8)
    offsets = [(37 + i * 29, 47 + j * 33) for i in range(4) for j in range(3)]
    tiles = [np.asarray(donor.crop((x, y, x + tw, y + th))) for x, y in offsets]
    selected = max(tiles, key=lambda tile: float(tile.astype(np.float64).std(axis=(0, 1)).mean()))
    return Image.fromarray(np.tile(selected, ((height + th - 1) // th, (width + tw - 1) // tw, 1))[:height, :width])


def known_edits(host, donor, slot):
    x, y, right, bottom = box = base.roi(host, slot, "other_paste")
    width, height = right - x, bottom - y
    patch = donor.crop((53, 66, 53 + width, 66 + height))
    repeated = repeated_patch(donor, patch.size, slot)
    color = Image.new("RGB", patch.size, (227, 238, 241))
    tint = Image.blend(host.crop(box), color, .52)
    for role, region in zip(POSITIVE, (patch, repeated, tint, patch), strict=True):
        edited = host.copy()
        if role == "feathered_paste":
            inset = 9
            mask = Image.new("L", patch.size)
            mask.paste(255, (inset, inset, width - inset, height - inset))
            mask = mask.filter(ImageFilter.GaussianBlur(4))
            edited.paste(region, (x, y), mask)
        else:
            edited.paste(region, (x, y))
        yield role, edited, box


def measurements(row, image):
    selected = row["combined"]["selected"]
    if selected is None or row["fixed"]["selected"] is not None:
        return row | {"context": None}
    box = selected["box"]
    pixels = np.asarray(image)
    x, y, right, bottom = box
    return row | {"context": {
        "alpha_sides": cues.alpha_measurement(pixels, box),
        "periodic": cues.periodic_measurement(pixels, box),
        "flat_share": flat_block_share(pixels[y + 4:bottom - 4, x + 4:right - 4]),
    }}


def measure(archive, cache, run, frozen, seal):
    (run / "inputs").mkdir()
    roster = frozen["photo_roster"]
    rows = []
    for slot in range(TARGETS):
        size = base.SIZES[slot % len(base.SIZES)]
        host = photo(archive, roster[slot], size)
        donor = photo(archive, roster[TARGETS + slot], size)
        controls = ((role, image, None) for role, image, _ in base.photo_views(host, donor, slot) if role in CONTROLS)
        for role, image, target in (*controls, *known_edits(host, donor, slot)):
            rows.append(measurements(base.observe(run, "photo", slot, role, image, host, target), image))
    for slot, item in enumerate(frozen["printed_signs"]):
        sign = base.photo(cache, item, (704, 544))
        for role, image in (("printed_original", sign), ("printed_resize", base.resize(sign))):
            rows.append(measurements(base.observe(run, "sign", slot, role, image, sign, None), image))
    eligible = {tuple(pair) for pair in seal["eligible"]}
    for record in seal["all_views"]:
        if (record["slot"], record["type"]) not in eligible:
            continue
        with Image.open(run / "captures" / record["filename"]) as capture:
            image = capture.convert("RGB")
        row = base.observe(run, "screenshot", record["slot"], record["type"], image, image, None)
        require(row["fixed"]["selected"] is None and row["fixed"]["proposal"] == record["proposal"]
                and row["fixed"]["score"] == record["score"], "Sealed incumbent measurement changed")
        rows.append(measurements(row, image) | {"caption": record["caption"]})
    return {"seal_sha256": sha(run / "incumbent.json"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "incumbent", "candidate"))
    for key in ("archive", "cache", "run"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "prepare":
        prepare(args.archive, args.run)
        return
    frozen = freeze(args.archive, args.cache, args.run)
    if args.stage == "freeze":
        save_new(args.run / "freeze.json", frozen)
        print(sha(args.run / "freeze.json"), flush=True)
        return
    require(base.read(args.run / "freeze.json") == frozen, "The held image context plan changed")
    sealed = incumbent(frozen, args.run)
    if args.stage == "incumbent":
        save_new(args.run / "incumbent.json", sealed)
        print(json.dumps({"views": len(sealed["eligible"]), "groups": len({slot for slot, _ in sealed["eligible"]})}), flush=True)
        return
    require(base.read(args.run / "incumbent.json") == sealed, "The incumbent-only seal changed")
    candidate = measure(args.archive, args.cache, args.run, frozen, sealed)
    save_new(args.run / "candidate.json", candidate)
    print(f"{len(candidate['rows'])} compared views", flush=True)


if __name__ == "__main__":
    main()
