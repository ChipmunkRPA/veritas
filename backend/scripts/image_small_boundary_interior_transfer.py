# SPDX-License-Identifier: Apache-2.0
"""Test a frozen interior flatness cue on conventional pastes and screenshot panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import sharp_boundary
from app.engine.fusion.content_kinds import flat_block_share
from scripts import image_small_boundary_control_stress as support
from scripts.image_small_boundary_screenshot_admission import save_new, sha

SCHEMA = "chipmunk.image-small-boundary-interior-transfer/1"
TARGETS = 9
FLOOR, INSET = .35, 4
PASTES = ("low_texture_paste", "normal_texture_paste")
CONTROLS = ("original", "global_tone", "feathered_blur", "resize_roundtrip")
POLICY = {
    "interior_floor": FLOOR, "inset_pixels": INSET, "minimum_negative_screen_groups": 6,
    "minimum_broad_screen_types_flagged": 2, "minimum_broad_low_texture_locations": 3,
    "minimum_gated_normal_locations_added": 3, "maximum_extra_screen_groups": 0,
    "maximum_extra_photo_control_groups": 0, "maximum_lost_low_texture_locations": 0,
}
require = support.require


def donor_plan(donor, width, height, slot):
    rgb = np.asarray(donor, dtype=np.int16)
    candidates = []
    for y in range(20, donor.height - height - 19, 24):
        for x in range(20, donor.width - width - 19, 24):
            tile = rgb[y:y + height, x:x + width]
            roughness = (np.abs(np.diff(tile, axis=0)).mean() + np.abs(np.diff(tile, axis=1)).mean()) / 2
            candidates.append((float(roughness), y, x))
    candidates.sort()
    selected = {}
    for role, (roughness, y, x) in zip(PASTES, (candidates[0], candidates[len(candidates) * 3 // 5]), strict=True):
        selected[role] = {"box": [x, y, x + width, y + height], "source_gradient": roughness,
                          "colors": 4 if role == PASTES[0] and slot % 2 == 0 else None}
    return selected


def paste_patch(donor, spec):
    patch = donor.crop(spec["box"])
    if spec["colors"]:
        patch = patch.quantize(colors=spec["colors"], dither=Image.Dither.NONE).convert("RGB")
    return patch


def roles(cache, roster, slot):
    size = support.SIZES[slot % len(support.SIZES)]
    return (support.photo(cache, roster["photos"][offset + slot], size) for offset in (0, TARGETS))


def prepare(cache, roster, run):
    output = run / "screen-photos"
    output.mkdir()
    for slot, row in enumerate(roster["photos"][2 * TARGETS:]):
        image = support.photo(cache, row, (768, 544))
        image.save(output / f"{slot}.jpg", quality=92, subsampling=0)


def freeze(cache, roster, run):
    photos = roster["photos"]
    require(len(photos) == TARGETS * 3 and len({row["author_profile"] for row in photos}) == len(photos)
            and len({key for row in photos for key in row["source_component"]}) == sum(len(row["source_component"]) for row in photos)
            and all(row["license_url"] in {"https://creativecommons.org/licenses/by/2.0/", "https://creativecommons.org/licenses/by-sa/2.0/"}
                    for row in photos), "Need distinct commercially reusable photo source components")
    plan = support.read(run / "plan.json")
    captures = support.read(run / "capture.json")
    expected = {(row["slot"], view): row["type"] for row in plan["screens"] for view in range(2)}
    rgb_hashes = {}
    for row in captures["rows"]:
        slot, view = row["slot"], row["view"]
        require(row["type"] == expected.pop((slot, view)) and row["filename"] == f"{slot}-{view}.png"
                and sha(run / "screen-photos" / f"{slot}.jpg") == row["photo_jpeg_sha256"], "The screenshot source changed")
        path = run / "captures" / row["filename"]
        require(sha(path) == row["sha256"], "The screenshot bytes changed")
        with Image.open(path) as image:
            rgb_hashes[row["filename"]] = support.digest(image.convert("RGB").tobytes())
            require(list(image.size) == row["size"] and sharp_boundary.supported_size(*image.size), "Unexpected screenshot geometry")
    require(not expected and len(set(rgb_hashes.values())) == 2 * TARGETS, "Distinct native screenshots required")
    for row in photos[2 * TARGETS:]:
        support.photo(cache, row, (768, 544))
    locations = []
    for slot in range(TARGETS):
        host, donor = roles(cache, roster, slot)
        x, y, right, bottom = support.roi(host, slot, "other_paste")
        selected = donor_plan(donor, right - x, bottom - y, slot)
        for item in selected.values():
            item["known_patch_interior_flat_share"] = flat_block_share(np.asarray(paste_patch(donor, item))[INSET:-INSET, INSET:-INSET])
        locations.append({"slot": slot, "target": [x, y, right, bottom], "source_patches": selected})
    paths = (Path(__file__), Path(__file__).with_name("capture_image_small_boundary_interiors.cjs"),
             Path(support.__file__), Path(sharp_boundary.__file__), Path(support.prior.__file__))
    return {"schema": SCHEMA, "policy": POLICY, "fixed_point": sharp_boundary.REVIEW_POINT,
            "fallback_point": support.POINT, "minimum_side": support.fallback.MIN_SHORT_EDGE,
            "maximum_axis_fraction": support.fallback.MAX_AXIS_FRACTION, "roster": roster, "locations": locations,
            "capture_plan": plan, "captures": captures, "capture_rgb_sha256": rgb_hashes,
            "code_sha256": {path.name: sha(path) for path in paths}}


def incumbent(run, frozen):
    rows = []
    for row in frozen["captures"]["rows"]:
        with Image.open(run / "captures" / row["filename"]) as image:
            response = sharp_boundary.measure(np.asarray(image.convert("RGB")))
        rows.append({"slot": row["slot"], "view": row["view"], "type": row["type"], "filename": row["filename"],
                     "score": response.score, "proposal": list(response.box)})
    return {"freeze_sha256": sha(run / "freeze.json"), "all_screens": rows,
            "eligible": [[row["slot"], row["view"]] for row in rows if row["score"] <= frozen["fixed_point"]]}


def admit(row, image):
    selected = row["combined"]["selected"]
    share = None
    if selected and row["fixed"]["selected"] is None:
        x, y, right, bottom = selected["box"]
        share = flat_block_share(np.asarray(image)[y + INSET:bottom - INSET, x + INSET:right - INSET])
        require(share is not None, "Interior RGB geometry missing")
    row["interior_flat_share"] = share
    row["admitted"] = selected if share is None or share < FLOOR else None
    return row


def score(cache, run, frozen, seal):
    (run / "inputs").mkdir()
    rows = []
    for slot, geometry in enumerate(frozen["locations"]):
        host, donor = roles(cache, frozen["roster"], slot)
        for role, image, _ in support.photo_views(host, donor, slot):
            if role in CONTROLS:
                rows.append(admit(support.observe(run, "photo", slot, role, image, host, None), image))
        for role in PASTES:
            edited = host.copy()
            edited.paste(paste_patch(donor, geometry["source_patches"][role]), tuple(geometry["target"][:2]))
            rows.append(admit(support.observe(run, "photo", slot, role, edited, host, geometry["target"]), edited))
    eligible = {tuple(pair) for pair in seal["eligible"]}
    for before in seal["all_screens"]:
        slot, view = before["slot"], before["view"]
        if (slot, view) not in eligible:
            continue
        with Image.open(run / "captures" / before["filename"]) as image:
            rgb = image.convert("RGB")
            result = support.observe(run, "screenshot", slot, f"viewport_{view}", rgb, rgb, None)
        require(result["fixed"]["selected"] is None and result["fixed"]["proposal"] == before["proposal"]
                and result["fixed"]["score"] == before["score"], "Incumbent result changed")
        rows.append(admit(result, rgb) | {"view": view, "type": before["type"]})
    return {"seal_sha256": sha(run / "incumbent.json"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "incumbent", "candidate"))
    for key in ("cache", "roster", "run"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    roster = support.read(args.roster)
    if args.stage == "prepare":
        prepare(args.cache, roster, args.run)
        return
    frozen = freeze(args.cache, roster, args.run)
    if args.stage == "freeze":
        save_new(args.run / "freeze.json", frozen)
        print(sha(args.run / "freeze.json"), flush=True)
        return
    require(support.read(args.run / "freeze.json") == frozen, "The interior comparison plan changed")
    seal = incumbent(args.run, frozen)
    if args.stage == "incumbent":
        save_new(args.run / "incumbent.json", seal)
        print(json.dumps({"views": len(seal["eligible"]), "groups": len({slot for slot, _ in seal["eligible"]})}), flush=True)
        return
    require(support.read(args.run / "incumbent.json") == seal, "The incumbent seal changed")
    result = score(args.cache, args.run, frozen, seal)
    save_new(args.run / "candidate.json", result)
    print(f"{len(result['rows'])} scored views", flush=True)


if __name__ == "__main__":
    main()
