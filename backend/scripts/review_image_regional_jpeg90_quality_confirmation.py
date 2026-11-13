# SPDX-License-Identifier: Apache-2.0
"""Recompute grouped JPEG90 agreement on the source-disjoint confirmation set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require
    from .review_image_regional_jpeg90_quality_agreement import ARMS
else:
    from review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require
    from review_image_regional_jpeg90_quality_agreement import ARMS

SCHEMA = "chipmunk.image-regional-jpeg90-quality-confirmation/1"
DESIGN = {
    "sources": "All qualifying CC BY 2.0 photos from the archived 12-group first F5 panel. Exclude previously used quality-agreement groups by individual photo, credited photographer and original bytes. One archived source falls below the 512-pixel short-side scope.",
    "delivery": "Eleven different photos and eight variants per photo. Final JPEG75 and direct JPEG90 share exact pre-encode RGB and truth. Run the unchanged quality-75 projection and agreement rule from the two-bank trial.",
    "criterion": "Preserve at least two-thirds of earlier useful files if there are three or more, require at least two useful files across at least two source photos, and add at most two ordinary flags and one new flagged source group versus the corresponding native JPEG75 controls.",
    "confidence_weight": 0,
}
ROLES = {"original", "local_blur", "local_tone", "local_copy", "global_resize", "second_jpeg", "sharp", "feathered"}


def summary(planned, rows):
    require(len(planned) == len(rows) == 88, "Need all 88 planned JPEG comparisons")
    totals = {arm: Counter() for arm in ARMS}
    by_family = {arm: Counter() for arm in ARMS}
    flag_groups = {arm: set() for arm in ARMS}
    useful_groups = {arm: set() for arm in ARMS}
    roles, changes = {}, Counter()
    for key, row in zip(planned, rows, strict=True):
        group, family, role = (key[name] for name in ("group", "family", "role"))
        seen = roles.setdefault((group, family), set())
        require(role not in seen, "Duplicate treatment within a source group")
        seen.add(role)
        h, w = key["geometry"]
        require(max(h, w) == 768 and min(h, w) >= 512 and key["q75_sha256"] != key["q90_sha256"], "Unexpected JPEG geometry")
        require(set(row["measures"]) == set(row["mask_sha256"]) == set(ARMS) and all(HASH.fullmatch(value)
                for value in (key["q75_sha256"], key["q90_sha256"], key["truth_sha256"], row["derived_sha256"],
                              *row["mask_sha256"].values())), "Missing source, repeated view or mask commitments")
        edited = role in EDITED
        data = row["measures"]
        require(len({value["reference_positive_pixels"] for value in data.values()}) == 1, "Edit truth differs across arms")
        result = {arm: measure(data[arm], h * w, edited) for arm in ARMS}
        overlap = row["native_components"]
        require(len(overlap) <= 2 and all(type(size) is int and type(shared) is int and 0 <= shared <= size and size > 0
                                        for size, shared in overlap), "Invalid repeated-view overlap")
        require(bool(overlap) == result["direct_q90"][0] and any(shared * 2 >= size for size, shared in overlap) == result["agreement"][0]
                and data["agreement"]["predicted_positive_pixels"] <= data["direct_q90"]["predicted_positive_pixels"]
                and data["agreement"]["intersection"] <= data["direct_q90"]["intersection"], "Agreement alters native components")
        for arm, (flag, useful) in result.items():
            if edited:
                totals[arm].update(edited=1, useful=useful, wrong=flag and not useful)
                by_family[arm][family] += useful
                if useful:
                    useful_groups[arm].add(group)
            else:
                totals[arm].update(ordinary=1, flags=flag)
                if flag:
                    flag_groups[arm].add(group)
        old, direct, agreed = (result[arm] for arm in ("native_q75", "direct_q90", "agreement"))
        if edited:
            changes.update(kept=old[1] and agreed[1], lost=old[1] and not agreed[1], gained=not old[1] and agreed[1],
                           direct_locations_lost=direct[1] and not agreed[1])
        else:
            changes.update(added_flags=agreed[0] and not old[0], removed_flags=old[0] and not agreed[0],
                           direct_flags_removed=direct[0] and not agreed[0])
    require({group for group, _ in roles} == set(range(1, 12)) and Counter(family for _, family in roles) == {"lama": 6, "sd15": 5}
            and all(found == ROLES for found in roles.values()), "Source-group coverage differs")
    changes["added_groups"] = len(flag_groups["agreement"] - flag_groups["native_q75"])
    earlier = totals["native_q75"]["useful"]
    passed = ((earlier < 3 or 3 * changes["kept"] >= 2 * earlier) and totals["agreement"]["useful"] >= 2
              and len(useful_groups["agreement"]) >= 2 and changes["added_flags"] <= 2 and changes["added_groups"] <= 1)
    return {"totals": totals, "useful_by_family": by_family, "paired": changes,
            "ordinary_groups": {arm: len(found) for arm, found in flag_groups.items()},
            "useful_groups": {arm: len(found) for arm, found in useful_groups.items()}, "passes": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    freeze = receipt["freeze"]
    require(freeze["schema"] == SCHEMA and freeze["design"] == DESIGN and all(HASH.fullmatch(value) for value in (
        freeze["checkpoint_sha256"], *freeze["code_sha256"].values(), *freeze["source_sha256"].values(),
    )), "Invalid frozen comparison")
    result = summary(freeze["rows"], receipt["rows"])
    require(receipt["summary"] == result, "Recorded source accounting differs")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
