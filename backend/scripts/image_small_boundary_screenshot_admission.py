# SPDX-License-Identifier: Apache-2.0
"""Freeze an incumbent-negative screenshot stratum before measuring an image fallback."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import sharp_boundary
from app.engine.fusion import content_kinds
from scripts import image_small_boundary_control_stress as support

SCHEMA = "chipmunk.image-small-boundary-screenshot-admission/1"
PHOTO_GROUPS = 12
LICENSES = {"https://creativecommons.org/licenses/by/2.0/", "https://creativecommons.org/licenses/by-sa/2.0/"}
FLOOR = .25
POLICY = {
    "flat_share_skip_at_or_above": FLOOR,
    "minimum_screen_groups": 20,
    "minimum_screen_families": 3,
    "minimum_broad_added_screen_groups": 4,
    "minimum_broad_added_screen_families": 2,
    "maximum_remaining_screen_groups": 2,
    "minimum_screen_flag_reduction_fraction": .75,
    "minimum_new_useful": 6,
    "minimum_new_useful_groups": 5,
    "minimum_new_useful_per_paste_type": 2,
    "maximum_lost_useful": 1,
    "maximum_new_photo_control_groups": 1,
    "maximum_new_photo_control_views": 2,
    "maximum_new_wrong_pastes": 2,
}
REQUIRE = support.require


def sha(path: Path) -> str:
    return support.digest(path.read_bytes())


def save_new(path: Path, data) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2) + "\n")


def sources(roster):
    photos = roster["photos"]
    slots = list(range(3 * PHOTO_GROUPS))
    REQUIRE([row["slot"] for row in photos] == slots
            and len({row["author_profile"] for row in photos}) == len(slots)
            and len({key for row in photos for key in row["source_component"]}) == sum(len(row["source_component"]) for row in photos)
            and len({row["encoded_sha256"] for row in photos}) == len(slots)
            and len({row["rgb_sha256"] for row in photos}) == len(slots)
            and all(row["license_url"] in LICENSES for row in photos), "Need distinct rights-qualified photo components")
    return photos


def composite_photos(cache: Path, roster, run: Path) -> None:
    output = run / "composite-photos"
    output.mkdir()
    for slot, row in enumerate(sources(roster)[2 * PHOTO_GROUPS:]):
        image = support.photo(cache, row, (768, 544))
        image.save(output / f"{slot}.jpg", quality=92, subsampling=0)


def merge_captures(initial: Path, reserve: Path, output: Path) -> None:
    REQUIRE(not (initial / "candidate.json").exists(), "The reserve must be opened before any candidate scoring")
    plan_name, captures_name = "screenshot-plan.json", "screenshot-captures.json"
    initial_plan, reserve_plan = (support.read(path / plan_name) for path in (initial, reserve))
    initial_captures, reserve_captures = (support.read(path / captures_name) for path in (initial, reserve))
    REQUIRE(initial_plan["variants"] == reserve_plan["variants"] and initial_captures["browser"] == reserve_captures["browser"],
            "Both intake stages require the same capture settings")
    source_freeze, source_seal = initial / "freeze.json", initial / "incumbent.json"
    initial_seal = support.read(source_seal)
    REQUIRE(initial_seal["freeze_sha256"] == sha(source_freeze), "The initial incumbent seal changed")
    output.mkdir(mode=0o700)
    (output / "captures").mkdir()
    for root, catalogue in ((initial, initial_captures), (reserve, reserve_captures)):
        for group in catalogue["groups"]:
            for view in group["views"]:
                filename = view["filename"]
                src, dst = root / "captures" / filename, output / "captures" / filename
                REQUIRE(src.name == filename and not dst.exists() and sha(src) == view["sha256"], "A reserve capture changed")
                shutil.copyfile(src, dst)
    save_new(output / plan_name, initial_plan | {"groups": initial_plan["groups"] + reserve_plan["groups"]})
    save_new(output / captures_name, initial_captures | {"groups": sorted(initial_captures["groups"] + reserve_captures["groups"],
                                                                         key=lambda row: row["id"])})
    save_new(output / "reserve-receipt.json", {"initial_plan_sha256": sha(initial / plan_name),
                                              "initial_captures_sha256": sha(initial / captures_name),
                                              "initial_freeze_sha256": sha(source_freeze), "initial_seal_sha256": sha(source_seal),
                                              "initial_seal": initial_seal, "initial_capture_filenames": sorted(
                                                  view["filename"] for group in initial_captures["groups"] for view in group["views"]),
                                              "reserve_plan_sha256": sha(reserve / plan_name),
                                              "reserve_captures_sha256": sha(reserve / captures_name)})


def frozen_plan(cache: Path, roster, run: Path):
    REQUIRE(content_kinds.FLOOR == FLOOR, "The preexisting flat-block floor changed")
    for row in sources(roster):
        support.photo(cache, row, (768, 544))
    original_path = run / "screenshot-plan.json"
    capture_path = run / "screenshot-captures.json"
    plan = support.read(original_path)
    captures = support.read(capture_path)
    groups = {row["id"]: row for row in plan["groups"]}
    observed = {row["id"]: row for row in captures["groups"]}
    REQUIRE(len(groups) == len(observed) >= 60 and groups.keys() == observed.keys(), "The screenshot roster changed")
    native = {}
    for group in groups.values():
        row = observed[group["id"]]
        REQUIRE(row["family"] == group["family"] and len(row["views"]) <= 2, "Unexpected captured group")
        for view in row["views"]:
            file = f"{group['id']}-{view['view']}.png"
            REQUIRE(view["view"] in (0, 1) and view["filename"] == file and file not in native, "Unexpected captured view")
            capture = run / "captures" / file
            REQUIRE(sha(capture) == view["sha256"], "The captured PNG changed")
            with Image.open(capture) as image:
                REQUIRE(image.format == "PNG" and image.size == (view["width"], view["height"])
                        and sharp_boundary.supported_size(*image.size), "Unexpected native capture dimensions")
                native[file] = support.digest(image.convert("RGB").tobytes())
    receipt_path = run / "reserve-receipt.json"
    reserve_receipt = support.read(receipt_path) if receipt_path.exists() else None
    original_names = set(reserve_receipt["initial_capture_filenames"]) if reserve_receipt else set()
    first_by_pixels, duplicates = {}, {}
    for filename in sorted(native, key=lambda name: (name not in original_names, name)):
        digest = native[filename]
        first = first_by_pixels.setdefault(digest, filename)
        if first != filename:
            duplicates[filename] = first
    code = [Path(__file__), Path(support.__file__), Path(sharp_boundary.__file__), Path(content_kinds.__file__),
            Path(support.prior.__file__), Path(support.fallback.__file__), Path(__file__).with_name("capture_image_small_boundary_screenshots.cjs")]
    return {"schema": SCHEMA, "policy": POLICY, "fixed_point": sharp_boundary.REVIEW_POINT,
            "fallback_point": support.POINT, "minimum_side": support.fallback.MIN_SHORT_EDGE,
            "maximum_axis_fraction": support.fallback.MAX_AXIS_FRACTION, "roster": roster,
            "screenshot_plan_sha256": sha(original_path), "screenshot_captures_sha256": sha(capture_path),
            "screenshot_plan": plan, "screenshot_captures": captures, "capture_rgb_sha256": native, "excluded_duplicate_views": duplicates,
            "reserve_receipt": reserve_receipt,
            "code_sha256": {file.name: sha(file) for file in code}}


def fixed_on_screenshots(run: Path, freeze):
    groups = freeze["screenshot_captures"]["groups"]
    lineage = {group["id"]: group.get("lineage", group["id"]) for group in freeze["screenshot_plan"]["groups"]}
    rows = []
    for slot, group in enumerate(groups):
        for view in group["views"]:
            if view["filename"] in freeze["excluded_duplicate_views"]:
                continue
            with Image.open(run / "captures" / view["filename"]) as source:
                rgb = np.asarray(source.convert("RGB"))
                incumbent = sharp_boundary.measure(rgb)
            rows.append({"slot": slot, "group": group["id"], "lineage": lineage[group["id"]], "family": group["family"], "view": view["view"],
                         "filename": view["filename"], "size": [view["width"], view["height"]],
                         "score": incumbent.score, "proposal": list(incumbent.box)})
    eligible = [row for row in rows if row["score"] <= freeze["fixed_point"]]
    if receipt := freeze["reserve_receipt"]:
        by_file = {row["filename"]: row for row in rows}
        for old in receipt["initial_seal"]["all_views"]:
            REQUIRE(all(by_file[old["filename"]][key] == value for key, value in old.items() if key != "slot"),
                    "An initial incumbent result changed after adding the reserve")
    return {"schema": SCHEMA, "freeze_sha256": sha(run / "freeze.json"), "all_views": rows,
            "stratum": [[row["group"], row["view"]] for row in eligible]}


def admit(row, rgb):
    share = content_kinds.flat_block_share(np.asarray(rgb))
    REQUIRE(share is not None, "Flat-block share unavailable")
    row["flat_block_share"] = share
    row["admitted"] = row["fixed"]["selected"] if share >= FLOOR else row["combined"]["selected"]
    return row


def measure(cache: Path, run: Path, roster, sealed):
    (run / "inputs").mkdir()
    photos = roster["photos"]
    rows = []
    for slot in range(PHOTO_GROUPS):
        size = support.SIZES[slot % len(support.SIZES)]
        original = support.photo(cache, photos[slot], size)
        donor = support.photo(cache, photos[slot + PHOTO_GROUPS], size)
        rows.extend(admit(support.observe(run, "photo", slot, role, image, original, target), image)
                    for role, image, target in support.photo_views(original, donor, slot))
    eligible = {tuple(item) for item in sealed["stratum"]}
    for before in sealed["all_views"]:
        if (before["group"], before["view"]) not in eligible:
            continue
        with Image.open(run / "captures" / before["filename"]) as source:
            rgb = source.convert("RGB")
            record = support.observe(run, "screenshot", before["slot"], f"viewport_{before['view']}", rgb, rgb, None)
        REQUIRE(record["fixed"]["selected"] is None and record["fixed"]["score"] == before["score"]
                and record["fixed"]["proposal"] == before["proposal"], "The measurement differs from the sealed incumbent")
        rows.append(admit(record, rgb) | {"group": before["group"], "lineage": before["lineage"],
                                         "family": before["family"], "view": before["view"]})
    return {"schema": SCHEMA, "seal_sha256": sha(run / "incumbent.json"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "merge", "freeze", "incumbent", "candidate"))
    for argument in ("cache", "roster", "run"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "merge":
        merge_captures(args.cache, args.roster, args.run)
        return
    roster = support.read(args.roster)
    if args.stage == "prepare":
        composite_photos(args.cache, roster, args.run)
        return
    freeze = frozen_plan(args.cache, roster, args.run)
    freeze_path = args.run / "freeze.json"
    if args.stage == "freeze":
        save_new(freeze_path, freeze)
        print(sha(freeze_path), flush=True)
        return
    REQUIRE(support.read(freeze_path) == freeze, "The screenshot or photo plan changed after freeze")
    seal_path = args.run / "incumbent.json"
    sealed = fixed_on_screenshots(args.run, freeze)
    if args.stage == "incumbent":
        save_new(seal_path, sealed)
        chosen = {tuple(pair) for pair in sealed["stratum"]}
        print(json.dumps({"distinct_views": len(sealed["all_views"]), "stratum_views": len(chosen),
                          "stratum_groups": len({row["lineage"] for row in sealed["all_views"] if (row["group"], row["view"]) in chosen}),
                          "seal_sha256": sha(seal_path)}), flush=True)
        return
    REQUIRE(support.read(seal_path) == sealed, "The incumbent-only screenshot seal changed")
    chosen = {tuple(pair) for pair in sealed["stratum"]}
    by_group = {row["lineage"]: row["family"] for row in sealed["all_views"] if (row["group"], row["view"]) in chosen}
    REQUIRE(len(by_group) >= POLICY["minimum_screen_groups"] and len(set(by_group.values())) >= POLICY["minimum_screen_families"],
            "The incumbent-negative stratum did not reach its frozen minimum. No candidate score was produced.")
    scored = measure(args.cache, args.run, roster, sealed)
    save_new(args.run / "candidate.json", scored)
    print(f"{len(scored['rows'])} photo and branch-eligible screenshot views", flush=True)


if __name__ == "__main__":
    main()
