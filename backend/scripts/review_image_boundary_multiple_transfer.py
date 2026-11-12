# SPDX-License-Identifier: Apache-2.0
"""Recompute second-boundary decisions from published geometry and counts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-boundary-multiple-transfer/1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def apart(a, b, pixels):
    return any(left[axis+2] + pixels <= right[axis] for left, right in ((a, b), (b, a)) for axis in (0, 1))


def area(box):
    x, y, right, bottom = box
    return max(0, right-x) * max(0, bottom-y)


def intersect(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def useful(truth, proposal, index):
    overlap, target = truth["overlaps"][index], truth["pixels"]
    return 2*overlap >= target and 10*overlap >= 3*(area(proposal["box"]) + target - overlap)


def score(row, arm, freeze):
    proposal = row["second"]
    if proposal is None or (arm == "frame_excluded" and proposal["rim_sides"] >= freeze["rim"]["side_minimum"]):
        return 0.
    return proposal["score"]


def points(rows, freeze):
    out = {}
    for arm in freeze["arms"]:
        maxima = defaultdict(float)
        for row in rows:
            label = f'{row["domain"]}/{row["slot"]}'
            maxima[label] = max(maxima[label], score(row, arm, freeze))
        out[arm] = {"point": max(freeze["fixed_point"], sorted(maxima.values())[-2]), "maxima": dict(maxima)}
    return out


def summarize(rows, freeze, operating_points):
    rules, roles = freeze["rules"], freeze["roles"]
    result = {}
    for arm in freeze["arms"]:
        counts = defaultdict(Counter)
        nuisance = defaultdict(set)
        for row in rows:
            role, domain, slot = row["role"], row["domain"], row["slot"]
            cell = counts[role]
            truths = row["truths"]
            primary = row["first"]["score"] > freeze["fixed_point"]
            added = score(row, arm, freeze) > operating_points[arm]["point"]
            first = {i for i, truth in enumerate(truths) if primary and useful(truth, row["first"], 0)}
            second = {i for i, truth in enumerate(truths) if added and useful(truth, row["second"], 1)}
            full_pair = bool(first and second - first and len(first | second) == 2)
            cell.update({"views": 1, "first_flags": primary, "added_flags": added, "first_useful": bool(first),
                         "second_useful": bool(second), "any_useful": bool(first | second), "useful_pairs": full_pair,
                         "wrong_added": added and not second and bool(truths),
                         "rim_rejected_above_point": bool(row["second"] and row["second"]["score"] > operating_points[arm]["point"]
                                                          and row["second"]["rim_sides"] >= freeze["rim"]["side_minimum"])})
            if added and (not truths or (role in roles["single"] and not second)):
                nuisance["single" if truths else domain].add(slot)
        photo_views = sum(counts[role]["added_flags"] for role in roles["photo_control"])
        screen_views = sum(counts[role]["added_flags"] for role in roles["screen_control"])
        result[arm] = {"by_role": {key: dict(value) for key, value in counts.items()}, "added_photo_views": photo_views,
                       "added_screen_views": screen_views, "added_photo_groups": len(nuisance["photo"]),
                       "added_screen_groups": len(nuisance["screen"]), "wrong_single_groups": len(nuisance["single"]),
                       "advances": (counts["double_sharp"]["useful_pairs"] >= rules["minimum_sharp_pairs"]
                                    and photo_views <= rules["maximum_added_photo_views"]
                                    and screen_views <= rules["maximum_added_screen_views"]
                                    and len(nuisance["photo"]) <= rules["maximum_added_photo_groups"]
                                    and len(nuisance["screen"]) <= rules["maximum_added_screen_groups"]
                                    and len(nuisance["single"]) <= rules["maximum_wrong_single_groups"])}
    return result


def check_box(box, size):
    require(len(box) == 4 and all(type(value) is int for value in box) and 0 <= box[0] < box[2] <= size[0]
            and 0 <= box[1] < box[3] <= size[1], "Invalid native bounds")


def check_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")


def verify(record):
    freeze = record["freeze"]
    require(record["schema"] == freeze["schema"] == SCHEMA, "Schema differs")
    freeze_data = (json.dumps(freeze, sort_keys=True, indent=2) + "\n").encode()
    require(hashlib.sha256(freeze_data).hexdigest() == record["freeze_sha256"], "Frozen input commitment differs")
    sources = [item["sha256"] for source_group in freeze["roster"].values() for item in source_group]
    require(len(sources) == len(set(sources)) == 36, "Fit, held or screenshot photos overlap")
    capture = {(row["slot"], row["variant"]): row for row in freeze["captures"]["rows"]}
    roles = freeze["roles"]
    rows = []
    for cohort in ("fit", "held"):
        views = record[cohort]
        photos = roles["photo_control"] + roles["single"] + (roles["positive"] if cohort == "held" else [])
        expected = {(domain, slot, role) for domain, slots, categories in (
            ("photo", range(len(freeze["roster"][cohort])), photos),
            ("screen", range(4) if cohort == "fit" else range(4, 16), roles["screen_control"]))
                    for slot in slots for role in categories}
        for row in views:
            domain, slot, role, size = (row[key] for key in ("domain", "slot", "role", "size"))
            require((domain, slot, role) in expected, "Unexpected case or duplicate")
            expected.remove((domain, slot, role))
            check_hash(row["input_sha256"])
            check_hash(row["rgb_sha256"])
            if domain == "screen":
                frame = capture[slot, role]
                group = freeze["capture_plan"]["groups"][slot]
                require(row["input_sha256"] == frame["png_sha256"] and size == [group["width"], group["height"]]
                        and row["rgb_sha256"] == freeze["capture_rgb_sha256"][frame["filename"]],
                        "Chromium PNG commitment differs")
            first, second = row["first"], row["second"]
            for proposal in filter(None, (first, second)):
                box = proposal["box"]
                check_box(box, size)
                check_hash(proposal["crop_rgb_sha256"])
                require(math.isfinite(proposal["score"]) and 0 <= proposal["score"] <= 3.00001,
                        "Invalid score outside native float32 precision")
                require(all(box[index+2]-box[index] in {round(size[index]*part) for part in (1/6, 1/4, 1/3)}
                            for index in range(2)), "Proposal dimensions changed")
            if second:
                require(first["score"] > freeze["fixed_point"] and second["score"] <= first["score"]
                        and apart(first["box"], second["box"], freeze["separation_pixels"])
                        and second["rim_sides"] in range(5), "Invalid separated proposal")
            truths = row["truths"]
            require(len(truths) == (2 if role in roles["positive"] else 1 if role in roles["single"] else 0), "Wrong truth count")
            for truth in truths:
                box = truth["box"]
                check_hash(truth["sha256"])
                check_box(box, size)
                require(0 < truth["pixels"] <= area(box), "Impossible truth area")
                require(len(truth["overlaps"]) == 2, "Wrong overlap count")
                for overlap, proposal in zip(truth["overlaps"], (first, second), strict=True):
                    require(type(overlap) is int and 0 <= overlap <= min(truth["pixels"], intersect(box, proposal["box"]) if proposal else 0),
                            "Impossible truth overlap")
            if len(truths) == 2:
                require(apart(truths[0]["box"], truths[1]["box"], freeze["separation_pixels"]), "Truth regions intersect")
        require(not expected, "Missing case")
        rows.extend(views)
    require(len({row["input_sha256"] for row in rows}) == len(rows), "Duplicate input bytes")
    require(points(record["fit"], freeze) == record["points"], "Operating points differ")
    require(summarize(record["held"], freeze, record["points"]) == record["summary"], "Held counts differ")
    diagnostic_points = {arm: {"point": freeze["fixed_point"]} for arm in freeze["arms"]}
    return {"views": len(rows), "held": len(record["held"]), "fit": len(record["fit"]), "points": record["points"],
            "summary": record["summary"], "diagnostic_at_first_region_point": summarize(record["held"], freeze, diagnostic_points)}


if __name__ == "__main__":
    result = verify(json.loads(Path(sys.argv[1]).read_bytes()))
    print(json.dumps(result, sort_keys=True, indent=2))
