# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen screenshot stratum, admission decisions and source-crop commitments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_image_small_boundary_control_stress import PASTES, audit_images, encoded, geometry, require, sha  # noqa: E402


def check_box(proposal, size, score) -> None:
    x, y, right, bottom = proposal
    width, height = size
    require(0 <= x < right <= width and 0 <= y < bottom <= height and math.isfinite(score) and score >= 0,
            "Invalid recorded box or score")


def seal_stratum(frozen, seal):
    require(seal["freeze_sha256"] == sha(encoded(frozen)), "The incumbent seal belongs to a different freeze")
    plan = frozen["screenshot_plan"]["groups"]
    captures = frozen["screenshot_captures"]["groups"]
    lineage = {row["id"]: row.get("lineage", row["id"]) for row in plan}
    initial = frozen["reserve_receipt"]
    original_names = set(initial["initial_capture_filenames"]) if initial else set()
    first_by_pixels, duplicates = {}, {}
    for filename in sorted(frozen["capture_rgb_sha256"], key=lambda name: (name not in original_names, name)):
        digest = frozen["capture_rgb_sha256"][filename]
        first = first_by_pixels.setdefault(digest, filename)
        if first != filename:
            duplicates[filename] = first
    require(duplicates == frozen["excluded_duplicate_views"], "Exact duplicate exclusion differs")
    expected = {(group["id"], view["view"]): (slot, group["family"], view)
                for slot, group in enumerate(captures) for view in group["views"] if view["filename"] not in duplicates}
    require({row["id"] for row in plan} == {row["id"] for row in captures}, "The capture roster differs")
    by_family = defaultdict(Counter)
    for group in plan:
        by_family[group["family"]]["planned_layouts"] += 1
        by_family[group["family"]]["planned_views"] += len(frozen["screenshot_plan"]["variants"])
    for family, _ in {(row["family"], lineage[row["id"]]) for row in plan}:
        by_family[family]["planned_groups"] += 1
    admitted = {}
    for row in seal["all_views"]:
        key = (row["group"], row["view"])
        require(key in expected, "Unexpected incumbent screenshot")
        slot, family, view = expected.pop(key)
        require((row["slot"], row["family"], row["lineage"], row["filename"], row["size"]) ==
                (slot, family, lineage[row["group"]], view["filename"], [view["width"], view["height"]]), "Incumbent screenshot binding differs")
        check_box(row["proposal"], row["size"], row["score"])
        by_family[family]["distinct_views"] += 1
        if row["score"] <= frozen["fixed_point"]:
            by_family[family]["eligible_views"] += 1
            admitted[key] = row
        else:
            by_family[family]["incumbent_flags"] += 1
    require(not expected and seal["stratum"] == [list(key) for key in admitted], "Incumbent-only stratum differs")
    for group in captures:
        by_family[group["family"]]["captured_layouts"] += bool(group["views"])
        by_family[group["family"]]["captured_views"] += len(group["views"])
    for family, _ in {(row["family"], lineage[row["id"]]) for row in captures if row["views"]}:
        by_family[family]["captured_groups"] += 1
    for family, _ in {(row["family"], row["lineage"]) for row in admitted.values()}:
        by_family[family]["eligible_groups"] += 1
    if initial:
        original = initial["initial_seal"]
        require(sha(encoded(original)) == initial["initial_seal_sha256"]
                and original["freeze_sha256"] == initial["initial_freeze_sha256"], "Initial incumbent seal differs")
        by_filename = {row["filename"]: row for row in seal["all_views"]}
        for row in original["all_views"]:
            require(all(by_filename[row["filename"]][key] == value for key, value in row.items() if key != "slot"),
                    "A sealed initial screenshot changed")
    return admitted, by_family


def selected(frozen, row):
    fixed, flexible = row["fixed"], row["combined"]
    for arm in (fixed, flexible):
        check_box(arm["proposal"], row["size"], arm["score"])
    old = fixed["proposal"] if fixed["score"] > frozen["fixed_point"] else None
    x, y, right, bottom = flexible["proposal"]
    width, height = row["size"]
    w, h = right - x, bottom - y
    active = (flexible["score"] > frozen["fallback_point"] and min(w, h) >= frozen["minimum_side"]
              and w <= width * frozen["maximum_axis_fraction"] and h <= height * frozen["maximum_axis_fraction"])
    broad = old or (flexible["proposal"] if active else None)
    fraction = row["flat_block_share"]
    require(math.isfinite(fraction) and 0 <= fraction <= 1, "Invalid flat-block share")
    guarded = old if fraction >= frozen["policy"]["flat_share_skip_at_or_above"] else broad
    observations = (fixed["selected"], flexible["selected"], row["admitted"])
    flags = [item is not None for item in observations]
    useful = [geometry(row, item, box) for item, box in zip(observations, (old, broad, guarded), strict=True)]
    require(not flags[0] or observations[0] == observations[1] == observations[2], "An incumbent crop changed")
    require(not flags[2] or observations[2] == observations[1], "Admission changed an existing crop")
    return flags, useful


def summarize(frozen, seal, candidate):
    require(candidate["seal_sha256"] == sha(encoded(seal)), "The candidate belongs to a different seal")
    admitted, screenshots = seal_stratum(frozen, seal)
    by_role = defaultdict(Counter)
    broad_screens, gated_screens, broad_families = set(), set(), set()
    gated_pastes, gated_photo_controls = set(), set()
    photo_slots = len(frozen["roster"]["photos"]) // 3
    photo_roles = ("original", "global_tone", "feathered_blur", "resize_roundtrip", "self_paste", "other_paste")
    photo_cases = {(slot, role) for slot in range(photo_slots) for role in photo_roles}
    for row in candidate["rows"]:
        role = row["role"]
        if row["cohort"] == "screenshot":
            group, view = row["group"], row["view"]
            require((group, view) in admitted, "The candidate saw a screenshot outside the incumbent seal")
            before = admitted.pop((group, view))
            require((row["slot"], row["family"], row["lineage"], row["size"], row["fixed"]["score"], row["fixed"]["proposal"]) ==
                    (before["slot"], before["family"], before["lineage"], before["size"], before["score"], before["proposal"]),
                    "Sealed incumbent measurement differs")
            old, broad, guarded = selected(frozen, row)[0]
            require(not old and row["target"] is None and row["target_pixels"] == 0, "Unexpected screenshot target")
            screenshots[row["family"]].update({"broad_added": broad, "gated_added": guarded})
            if broad:
                broad_screens.add(row["lineage"])
                broad_families.add(row["family"])
            if guarded:
                gated_screens.add(row["lineage"])
            continue
        require(row["cohort"] == "photo" and (row["slot"], role) in photo_cases, "Unexpected or duplicate photo case")
        photo_cases.remove((row["slot"], role))
        has_target = role in PASTES
        require(has_target == (row["target"] is not None and row["target_pixels"] > 0), "Invalid photo target")
        flags, useful = selected(frozen, row)
        cells = by_role[role]
        cells["views"] += 1
        for arm, flag, localized in zip(("fixed", "broad", "gated"), flags, useful, strict=True):
            cells.update({f"{arm}_flags": flag, f"{arm}_useful": localized,
                          f"{arm}_added_flags": flag and not flags[0], f"{arm}_added_useful": localized and not flags[0]})
        if useful[2] and not flags[0]:
            gated_pastes.add(row["slot"])
        if not has_target and flags[2] and not flags[0]:
            gated_photo_controls.add(row["slot"])
    require(not admitted and not photo_cases, "A required measured view is missing")
    photo_pastes = sum((by_role[role] for role in PASTES), Counter())
    photo_controls = sum((by_role[role] for role in by_role if role not in PASTES), Counter())
    totals = sum(screenshots.values(), Counter())
    policy = frozen["policy"]
    gate = {
        "screenshot_stratum": (totals["eligible_groups"] >= policy["minimum_screen_groups"]
                               and sum(bool(row["eligible_groups"]) for row in screenshots.values()) >= policy["minimum_screen_families"]),
        "challenged_screenshots": (len(broad_screens) >= policy["minimum_broad_added_screen_groups"]
                                  and len(broad_families) >= policy["minimum_broad_added_screen_families"]),
        "reduce_screenshot_flags": (len(gated_screens) <= policy["maximum_remaining_screen_groups"]
                                    and totals["gated_added"] <= (1 - policy["minimum_screen_flag_reduction_fraction"]) * totals["broad_added"]),
        "useful_pastes": (photo_pastes["gated_added_useful"] >= policy["minimum_new_useful"]
                          and len(gated_pastes) >= policy["minimum_new_useful_groups"]
                          and all(by_role[role]["gated_added_useful"] >= policy["minimum_new_useful_per_paste_type"] for role in PASTES)),
        "preserve_useful_pastes": photo_pastes["broad_useful"] - photo_pastes["gated_useful"] <= policy["maximum_lost_useful"],
        "wrong_new_pastes": photo_pastes["gated_added_flags"] - photo_pastes["gated_added_useful"] <= policy["maximum_new_wrong_pastes"],
        "new_photo_controls": (len(gated_photo_controls) <= policy["maximum_new_photo_control_groups"]
                               and photo_controls["gated_added_flags"] <= policy["maximum_new_photo_control_views"]),
        "incumbent_crops_preserved": True,
    }
    return {"screenshots_by_family": screenshots, "screenshots": totals, "photo_roles": by_role, "photo_pastes": photo_pastes,
            "photo_controls": photo_controls, "broad_added_screenshot_groups": len(broad_screens),
            "gated_added_screenshot_groups": len(gated_screens), "broad_added_screenshot_families": len(broad_families),
            "gated_added_photo_control_groups": len(gated_photo_controls), "gated_added_paste_groups": len(gated_pastes),
            "gate": gate, "qualifies": all(gate.values())}


def constant_blocks(raw, width, height):
    nrows, ncols = height // 8, width // 8
    stride = width * 3
    count = 0
    for y in range(nrows):
        for x in range(ncols):
            first = 8 * (y * stride + x * 3)
            uniform = raw[first:first + 3] * 8
            count += all(raw[first + i * stride:first + i * stride + 24] == uniform for i in range(8))
    return count / (nrows * ncols)


def native_audit(run, frozen, seal, candidate):
    from PIL import Image

    stats = audit_images(run, candidate["rows"])
    captures = {(group["id"], view["view"]): view for group in frozen["screenshot_captures"]["groups"] for view in group["views"]}
    for view in captures.values():
        with Image.open(run / "captures" / view["filename"]) as img:
            require(sha((run / "captures" / view["filename"]).read_bytes()) == view["sha256"]
                    and sha(img.convert("RGB").tobytes()) == frozen["capture_rgb_sha256"][view["filename"]], "Captured source differs")
    for row in candidate["rows"]:
        path = run / "inputs" / f"{row['cohort']}-{row['slot']:02d}-{row['role']}.png"
        with Image.open(path) as image:
            raw = image.tobytes()
            require(constant_blocks(raw, *image.size) == row["flat_block_share"], "Independent flat-block measurement differs")
        if row["cohort"] == "screenshot":
            view = captures[row["group"], row["view"]]
            require(sha(raw) == frozen["capture_rgb_sha256"][view["filename"]], "Scored screenshot pixels differ from the native capture")
    return stats | {"original_captures_verified": len(captures), "independent_flat_block_scores": len(candidate["rows"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["candidate"])) == package["audit"]["candidate_sha256"], "Published candidate commitment differs")
        summary = summarize(package["freeze"], package["incumbent"], package["candidate"])
        require(summary == package["summary"], "Published summary differs")
    else:
        freeze, incumbent, candidate = (json.loads((args.source / name).read_bytes())
                                         for name in ("freeze.json", "incumbent.json", "candidate.json"))
        summary = summarize(freeze, incumbent, candidate)
        audit = native_audit(args.source, freeze, incumbent, candidate) | {"candidate_sha256": sha(encoded(candidate))}
        if args.publish:
            with args.publish.open("xb") as stream:
                stream.write(encoded({"schema": freeze["schema"], "freeze": freeze, "incumbent": incumbent,
                                      "candidate": candidate, "summary": summary, "audit": audit}))
        print(json.dumps(audit, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
