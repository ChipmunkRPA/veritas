# SPDX-License-Identifier: Apache-2.0
"""Recompute the published native-boundary study from its scalar case ledger."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("fixed", "flexible", "distributed")
KNOWN = {"edited", "feathered_edit", "small_sharp", "small_feathered"}
ORDINARY = {"original", "local_processing", "global_processing", "blur", "feathered_processing", "local_contrast",
            "small_processing", "small_blur"}
GROUPS = {"earlier": 12, "later": 12, "newest": 7, "commons": 13}
PROSPECTIVE_ROLES = {"original", "contrast", "blur", "global_processing", "sharp", "feathered"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(areas):
    area, target, overlap = areas
    return target > 0 and overlap / target >= .50 and overlap / (target + area - overlap) >= .30


def review_fallback(prospective, point):
    grouped = defaultdict(set)
    counts = defaultdict(Counter)
    extra_untouched = set()
    for cohort, slot, role, dimensions, box, score, fixed, combined in prospective["ledger"]:
        group = (cohort, slot)
        require(cohort in ("commons", "flux") and type(slot) is int and slot >= 0
                and role in PROSPECTIVE_ROLES and role not in grouped[group], "Invalid or duplicate fallback case")
        grouped[group].add(role)
        width, height = dimensions
        x, y, r, b = box
        w, h = r - x, b - y
        require(512 <= width <= 768 and 512 <= height <= 768 and 0 <= x < r <= width and 0 <= y < b <= height
                and type(score) in (float, int) and math.isfinite(score), "Invalid proposed fallback geometry")
        expected = w * h if score > point and 56 <= w <= .18 * width and 56 <= h <= .18 * height else 0
        require(fixed[1] == combined[1] and bool(fixed[1]) == (role in {"sharp", "feathered"})
                and all(0 <= overlap <= min(area, target) and area + target - overlap <= width * height
                        for area, target, overlap in (fixed, combined))
                and (fixed == combined if fixed[0] else combined[0] == expected), "Incorrect fixed or fallback decision")
        extra = bool(combined[0] and not fixed[0])
        counts[f"{cohort}/{role}"].update({"views": 1, "old_flags": bool(fixed[0]), "new_flags": bool(combined[0]),
                                         "old_useful": useful(fixed), "new_useful": useful(combined),
                                         "new_locations": useful(combined) and not useful(fixed), "extra_flags": extra})
        if role == "original" and extra:
            extra_untouched.add(group)
    require(Counter(cohort for cohort, _ in grouped) == {"commons": 25, "flux": 6}
            and all(roles == PROSPECTIVE_ROLES for roles in grouped.values()), "Incomplete fallback source groups")
    extra = sum(values["extra_flags"] for key, values in counts.items() if key.split("/")[1] not in {"sharp", "feathered"})
    advances = (counts["commons/sharp"]["new_locations"] >= 12 and counts["flux/sharp"]["new_locations"] >= 2
                and extra <= 4 and len(extra_untouched) <= 2)
    result = prospective["result"]
    require((counts, extra, len(extra_untouched), advances) == (result["counts"], result["extra_ordinary"],
             result["extra_untouched_groups"], result["advances"]), "Published fallback totals differ")
    return {"verified_views": len(prospective["ledger"]), "source_groups": len(grouped), "advances": advances}


def review(document):
    require(document["schema"] == "chipmunk.image-boundary-extent-transfer/1", "Unexpected public receipt version")
    points, rows = document["points"], document["ledger"]
    require(document["arms"] == list(ARMS) and len(rows) == 385, "Incomplete public study")
    thresholds = {"fixed": .8292734622955322}
    for arm in ARMS[1:]:
        values = points[arm]["source_maxima"]
        require(len(values) == 8 and all(type(value) in (float, int) and math.isfinite(value) for value in values)
                and sorted(values)[6] == points[arm]["point"], "Calibration point does not match its source maxima")
        thresholds[arm] = points[arm]["point"]

    grouped = defaultdict(set)
    counts = {arm: defaultdict(Counter) for arm in ARMS}
    decisions = {arm: Counter() for arm in ARMS[1:]}
    gain_sources = {arm: set() for arm in ARMS[1:]}
    for cohort, slot, role, candidates in rows:
        group = (cohort, slot)
        allowed = {"commons_original"} if cohort == "commons" else KNOWN | ORDINARY
        require(cohort in GROUPS and type(slot) is int and slot >= 0 and role in allowed and role not in grouped[group],
                "Unexpected cohort, role or duplicate case")
        grouped[group].add(role)
        require(len(candidates) == 3 and len({entry[2] for entry in candidates}) == 1, "Arm target areas differ")
        baseline = candidates[0][1:]
        for arm, entry in zip(ARMS, candidates, strict=True):
            response, area, target, overlap = entry
            require(type(response) in (float, int) and math.isfinite(response) and response >= 0
                    and all(type(n) is int for n in (area, target, overlap)) and 0 <= overlap <= min(target, area)
                    and bool(target) == (role in KNOWN) and bool(area) == (response > thresholds[arm]),
                    "Inconsistent regional decision or pixel counts")
            hit, before = useful(entry[1:]), useful(baseline)
            counts[arm][role].update({"views": 1, "flags": bool(area), "useful": hit})
            if arm == "fixed":
                continue
            if role == "small_sharp" and hit and not before:
                decisions[arm]["gains"] += 1
                gain_sources[arm].add(group)
            if role == "edited" and before:
                decisions[arm]["incumbent_sharp"] += 1
                decisions[arm]["preserved_sharp"] += hit
            if role in ORDINARY:
                decisions[arm]["net_ordinary"] += int(bool(area)) - int(bool(baseline[0]))
            if role == "commons_original":
                decisions[arm]["added_commons"] += bool(area and not baseline[0])

    require(Counter(cohort for cohort, _ in grouped) == GROUPS, "Incomplete source groups")
    require(all(roles == ({"commons_original"} if cohort == "commons" else KNOWN | ORDINARY)
                for (cohort, _), roles in grouped.items()), "A source is missing planned variants")
    for arm, totals in decisions.items():
        sources = len(gain_sources[arm])
        advances = (totals["gains"] >= 5 and sources >= 3
                    and 20 * totals["preserved_sharp"] >= 17 * totals["incumbent_sharp"]
                    and totals["net_ordinary"] <= 4 and totals["added_commons"] <= 1)
        decisions[arm] = dict(totals) | {"distinct_gains": sources, "advances": advances}
    require(counts == document["result"]["counts"] and decisions == document["result"]["decision"],
            "Published totals or advancement decision differ")
    report = {"verified_views": len(rows), "source_groups": len(grouped), "region_decisions": 3 * len(rows),
              "decisions": decisions}
    if "prospective" in document:
        report["prospective"] = review_fallback(document["prospective"], thresholds["distributed"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
