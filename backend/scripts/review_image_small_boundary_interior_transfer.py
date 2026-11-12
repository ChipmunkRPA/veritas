# SPDX-License-Identifier: Apache-2.0
"""Independently check interior-screening decisions and native evidence bindings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_image_small_boundary_control_stress import audit_images, encoded, geometry, require, sha  # noqa: E402
from review_image_small_boundary_screenshot_admission import check_box, constant_blocks  # noqa: E402

PASTES = {"low_texture_paste", "normal_texture_paste"}
CONTROLS = {"original", "global_tone", "feathered_blur", "resize_roundtrip"}


def decision(row, frozen):
    fixed, flexible = row["fixed"], row["combined"]
    for arm in (fixed, flexible):
        check_box(arm["proposal"], row["size"], arm["score"])
    old = fixed["proposal"] if fixed["score"] > frozen["fixed_point"] else None
    x, y, right, bottom = flexible["proposal"]
    width, height = row["size"]
    w, h = right - x, bottom - y
    qualifies = (flexible["score"] > frozen["fallback_point"] and min(w, h) >= frozen["minimum_side"]
                 and w <= width * frozen["maximum_axis_fraction"] and h <= height * frozen["maximum_axis_fraction"])
    broad = old or (flexible["proposal"] if qualifies else None)
    share = row["interior_flat_share"]
    if broad is not None and old is None:
        require(type(share) is float and math.isfinite(share) and 0 <= share <= 1, "Invalid measured candidate interior")
    else:
        require(share is None, "Unexpected interior measurement")
    gated = None if share is not None and share >= frozen["policy"]["interior_floor"] else broad
    actual = (fixed["selected"], flexible["selected"], row["admitted"])
    useful = [geometry(row, observed, box) for observed, box in zip(actual, (old, broad, gated), strict=True)]
    require(old is None or actual[0] == actual[1] == actual[2], "Existing crop changed")
    require(gated is None or actual[1] == actual[2], "Gated crop changed")
    return tuple(box is not None for box in (old, broad, gated)), useful


def summarize(frozen, sealed, candidate):
    require(sealed["freeze_sha256"] == sha(encoded(frozen)) and candidate["seal_sha256"] == sha(encoded(sealed)), "Invalid phase commitment")
    screen_by_view = {(row["slot"], row["view"]): row for row in frozen["captures"]["rows"]}
    expected = {}
    screen_types = defaultdict(Counter)
    for row in sealed["all_screens"]:
        key = (row["slot"], row["view"])
        source = screen_by_view.pop(key)
        require((row["filename"], row["type"]) == (source["filename"], source["type"]), "Screenshot row differs from capture")
        check_box(row["proposal"], source["size"], row["score"])
        screen_types[row["type"]]["captured_views"] += 1
        if row["score"] <= frozen["fixed_point"]:
            expected[key] = row
            screen_types[row["type"]]["eligible_views"] += 1
        else:
            screen_types[row["type"]]["incumbent_flags"] += 1
    require(not screen_by_view and sealed["eligible"] == [list(key) for key in expected], "The incumbent-only stratum differs")
    negative_groups = {key[0] for key in expected}
    locations = {row["slot"]: row for row in frozen["locations"]}
    photo_expected = {(slot, role) for slot in locations for role in CONTROLS | PASTES}
    by_role = defaultdict(Counter)
    flagged_screen_groups, flagged_control_groups, broad_screen_types = set(), set(), set()
    for row in candidate["rows"]:
        slot, role = row["slot"], row["role"]
        flags, useful = decision(row, frozen)
        fixed, broad, gated = flags
        if row["cohort"] == "screenshot":
            source = expected.pop((slot, row["view"]))
            require(row["type"] == source["type"] and not fixed and row["fixed"]["score"] == source["score"]
                    and row["fixed"]["proposal"] == source["proposal"] and row["target"] is None, "The sealed screenshot differs")
            screen_types[row["type"]].update({"broad_added_flags": broad, "gated_added_flags": gated})
            if broad:
                broad_screen_types.add(row["type"])
            if gated:
                flagged_screen_groups.add(slot)
        else:
            require(row["cohort"] == "photo" and (slot, role) in photo_expected, "Unexpected photo case")
            photo_expected.remove((slot, role))
            is_paste = role in PASTES
            require(row["target"] == (locations[slot]["target"] if is_paste else None)
                    and (row["target_pixels"] > 0) == is_paste, "Unexpected known target")
            by_role[role]["views"] += 1
            for index, label in enumerate(("fixed", "broad", "gated")):
                by_role[role].update({f"{label}_flags": flags[index], f"{label}_useful": useful[index],
                                      f"{label}_added_useful": useful[index] and not fixed})
            by_role[role]["lost_useful"] += useful[1] and not useful[2]
            if not is_paste and gated and not fixed:
                flagged_control_groups.add(slot)
    require(not expected and not photo_expected, "Required comparison views missing")
    policy = frozen["policy"]
    low, normal = by_role["low_texture_paste"], by_role["normal_texture_paste"]
    criteria = {
        "sealed_screenshot_stratum": (len(negative_groups) >= policy["minimum_negative_screen_groups"]
                                      and all(row["eligible_views"] for row in screen_types.values())),
        "challenged_screens": len(broad_screen_types) >= policy["minimum_broad_screen_types_flagged"],
        "challenged_low_texture": low["broad_useful"] >= policy["minimum_broad_low_texture_locations"],
        "normal_texture_utility": normal["gated_added_useful"] >= policy["minimum_gated_normal_locations_added"],
        "no_new_screen_groups": len(flagged_screen_groups) <= policy["maximum_extra_screen_groups"],
        "no_new_photo_control_groups": len(flagged_control_groups) <= policy["maximum_extra_photo_control_groups"],
        "preserved_low_texture_utility": low["lost_useful"] <= policy["maximum_lost_low_texture_locations"],
    }
    return {"screens_by_type": screen_types, "photo_roles": by_role, "incumbent_negative_screenshot_groups": len(negative_groups),
            "new_screen_groups": len(flagged_screen_groups), "new_photo_control_groups": len(flagged_control_groups),
            "criteria": criteria, "qualifies": all(criteria.values())}


def native_audit(run, frozen, candidate):
    from PIL import Image

    rows = candidate["rows"]
    stats = audit_images(run, rows)
    for row in frozen["captures"]["rows"]:
        path = run / "captures" / row["filename"]
        with Image.open(path) as image:
            require(sha(path.read_bytes()) == row["sha256"]
                    and sha(image.convert("RGB").tobytes()) == frozen["capture_rgb_sha256"][row["filename"]], "Captured source pixels differ")
    measured = 0
    for row in rows:
        path = run / "inputs" / f"{row['cohort']}-{row['slot']:02d}-{row['role']}.png"
        with Image.open(path) as image:
            if row["cohort"] == "screenshot":
                capture = f"{row['slot']}-{row['view']}.png"
                require(sha(image.tobytes()) == frozen["capture_rgb_sha256"][capture], "Candidate screenshot pixels differ")
            if row["interior_flat_share"] is not None:
                x, y, right, bottom = row["combined"]["selected"]["box"]
                inset = frozen["policy"]["inset_pixels"]
                interior = image.crop((x + inset, y + inset, right - inset, bottom - inset))
                require(constant_blocks(interior.tobytes(), *interior.size) == row["interior_flat_share"], "Independent interior flatness differs")
                measured += 1
    return stats | {"independent_interiors": measured, "native_capture_bindings": len(frozen["captures"]["rows"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["candidate"])) == package["audit"]["candidate_sha256"], "Candidate hash differs")
        result = summarize(package["freeze"], package["incumbent"], package["candidate"])
        require(result == package["summary"], "Published decision differs")
    else:
        frozen, sealed, candidate = (json.loads((args.source / filename).read_bytes())
                                     for filename in ("freeze.json", "incumbent.json", "candidate.json"))
        result = summarize(frozen, sealed, candidate)
        audit = native_audit(args.source, frozen, candidate) | {"candidate_sha256": sha(encoded(candidate))}
        if args.publish:
            with args.publish.open("xb") as stream:
                stream.write(encoded({"schema": frozen["schema"], "freeze": frozen, "incumbent": sealed,
                                      "candidate": candidate, "summary": result, "audit": audit}))
        print(json.dumps(audit, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
