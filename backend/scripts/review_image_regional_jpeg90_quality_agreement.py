# SPDX-License-Identifier: Apache-2.0
"""Recompute JPEG quality-agreement decisions without the images or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require
else:
    from review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require

SCHEMA = "chipmunk.image-regional-jpeg90-quality-agreement/1"
ARMS = ("native_q75", "direct_q90", "projected_q75", "agreement")
DESIGN = {
    "source": "Use the complete 200-delivery native JPEG90 panel and its exact direct-source byte, selected mask and truth commitments.",
    "repeated_view": "Decode the submitted baseline 4:2:0 JPEG90, then save at baseline IJG JPEG75 4:2:0. No crop or resize.",
    "arms": ["Earlier native JPEG75", "Direct JPEG90", "JPEG90-to-JPEG75 view", "Native JPEG90 components supported in the repeated view"],
    "selection": "Use the delivered JPEG90 reliability conditions and existing component order in both views. Retain each original component if at least half its grid cells are selected in the repeat. No replacement of native geometry.",
    "adoption": "Keep at least eight of the 11 useful earlier locations with both previously detected families, add at most five ordinary flags from at most two new source groups, and add at most four wrong edited highlights versus the existing native JPEG75 baseline. Require at least one useful transfer in the separate second bank.",
    "confidence_weight": 0,
}


def summary(rows, source):
    require(len(rows) == len(source) == 200, "Two complete source banks are required")
    groups, reverse = {}, {}
    totals = {arm: Counter() for arm in ARMS}
    panels = {key: {arm: Counter() for arm in ARMS} for key in ("retained_b", "external")}
    families = {arm: Counter() for arm in ARMS}
    ordinary_groups = {arm: set() for arm in ARMS}
    earlier_families, retained_families = set(), set()
    changes = Counter()
    for index, (row, baseline) in enumerate(zip(rows, source, strict=True)):
        group, panel, role, family = (baseline[key] for key in ("group", "panel", "role", "family"))
        alias = row["group_index"]
        require(row["source_index"] == index and groups.setdefault(group, alias) == alias
                and reverse.setdefault(alias, group) == group, "Source index or group binding differs")
        require(all(HASH.fullmatch(digest) for digest in (row["derived_sha256"], *row["mask_sha256"].values()))
                and set(row["mask_sha256"]) == set(ARMS[2:]), "Missing view or mask commitments")
        require(row["derived_sha256"] != baseline["q90_sha256"], "Repeated view reused original bytes")
        sizes = row["native_components"]
        require(len(sizes) <= 2 and all(type(count) is int and type(shared) is int and 0 <= shared <= count and count > 0
                                      for count, shared in sizes), "Impossible component overlap")
        data = {"native_q75": baseline["q75"], "direct_q90": baseline["q90"]} | row["measures"]
        h, w = baseline["shape"]
        edited = role in EDITED
        for arm in ARMS:
            require(data[arm]["reference_positive_pixels"] == baseline["q75"]["reference_positive_pixels"],
                    "Changed-pixel truth differs between arms")
        scores = {arm: measure(data[arm], h * w, edited) for arm in ARMS}
        require(bool(sizes) == scores["direct_q90"][0]
                and any(2 * shared >= count for count, shared in sizes) == scores["agreement"][0]
                and data["agreement"]["predicted_positive_pixels"] <= data["direct_q90"]["predicted_positive_pixels"]
                and data["agreement"]["intersection"] <= data["direct_q90"]["intersection"],
                "Agreement changed native region support or selection")
        for arm, (flag, useful) in scores.items():
            increments = (dict(edited=1, useful=useful, wrong=flag and not useful) if edited
                          else dict(ordinary=1, flags=flag))
            totals[arm].update(increments)
            panels[panel][arm].update(increments)
            if edited:
                families[arm][family] += useful
            elif flag:
                ordinary_groups[arm].add(group)
        old, direct, agreed = (scores[arm] for arm in ("native_q75", "direct_q90", "agreement"))
        if edited:
            changes.update(kept=old[1] and agreed[1], gained=not old[1] and agreed[1], lost=old[1] and not agreed[1],
                           lost_from_direct=direct[1] and not agreed[1], gained_from_direct=agreed[1] and not direct[1])
            if old[1]:
                earlier_families.add(family)
                if agreed[1]:
                    retained_families.add(family)
        else:
            changes.update(added_flags=agreed[0] and not old[0], removed_flags=old[0] and not agreed[0],
                           removed_from_direct=direct[0] and not agreed[0])
    require(len(groups) == 22 and totals["native_q75"] == {"edited": 46, "useful": 11, "wrong": 4, "ordinary": 154, "flags": 13}
            and totals["direct_q90"] == {"edited": 46, "useful": 11, "wrong": 9, "ordinary": 154, "flags": 30},
            "Baseline or panel accounting differs")
    changes["new_ordinary_groups"] = len(ordinary_groups["agreement"] - ordinary_groups["native_q75"])
    passed = (changes["kept"] >= 8 and earlier_families == retained_families and changes["added_flags"] <= 5
              and changes["new_ordinary_groups"] <= 2 and totals["agreement"]["wrong"] <= totals["native_q75"]["wrong"] + 4
              and panels["external"]["agreement"]["useful"] >= 1)
    return {"totals": totals, "panels": panels, "useful_by_family": families, "paired": changes,
            "ordinary_groups": {arm: len(value) for arm, value in ordinary_groups.items()}, "passes": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("direct_prior", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    direct = args.direct_prior.read_bytes()
    lock = receipt["freeze"]
    require(lock["schema"] == SCHEMA and lock["design"] == DESIGN
            and lock["direct_receipt_sha256"] == hashlib.sha256(direct).hexdigest(), "Frozen design or direct baseline differs")
    result = summary(receipt["rows"], json.loads(direct)["rows"])
    require(result == receipt["summary"], "Recorded comparison differs")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
