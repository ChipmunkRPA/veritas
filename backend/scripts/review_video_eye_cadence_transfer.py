# SPDX-License-Identifier: MIT
"""Recount the matched 25 and 30 fps eye-region comparison without media or weights."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from statistics import median

from scripts.review_video_eye_shortlist_context import COLLAR_POINT, load, localized, valid_digest

SCHEMA = "chipmunk.video-eye-cadence-transfer/1"
RATES = (25, 30)
TEXTURE_POINT = .115
EDITS = {"short_hold", "long_hold", "long_hold_tandem"}
WHOLE = {"whole_hold", "picture_jump", "hard_cut"}
CONTROLS = WHOLE | {"original", "original_tandem", "eye_blur", "eye_gain"}
ARMS = ("raw", "top_only", "shortlist")
FIELDS = (
    "time_base", "assessed", "truth", "raw", "top_collar_gap_hex", "shortlist", "shortlist_gap_hex",
    "source_witnesses", "elapsed_us", "delivery_sha256",
)
GATE = {"minimum_25fps_located": 8, "minimum_30fps_located": 8, "minimum_groups_each": 4,
        "max_ordinary_each": 2, "max_whole_picture_alerts": 0, "max_paired_30fps_loss": 4}


def valid_box(box) -> bool:
    x, y, width, height = box
    return all(type(value) is int for value in box) and 0 <= x < x + width <= 480 and 0 <= y < y + height <= 360


def validate_groups(group) -> None:
    clock, pts = group["parent_clock"]
    step = Fraction(clock)
    if (len(pts) != 60 or pts[0] != 0 or any(type(point) is not int for point in pts)
            or any((b - a) * step not in (Fraction(33, 1_000), Fraction(34, 1_000)) for a, b in pairwise(pts))):
        raise ValueError("The preserved camera timestamps are incomplete")
    sampled = [min(range(60), key=lambda i: (abs(pts[i] * step - Fraction(n, 25)), i)) for n in range(50)]
    if group["index_maps"] != {"25": sampled, "30": list(range(60))} or len(set(sampled)) != 50:
        raise ValueError("A published delivery does not map to the planned camera frames")
    if (len(group["eye_boxes"]) != 2 or not all(map(valid_box, group["eye_boxes"]))
            or set(group["staged_regions"]) != {"short_hold", "long_hold"}
            or {row[0] for row in group["cases"]} != EDITS | CONTROLS or len(group["cases"]) != 10):
        raise ValueError("Missing source region or matched case")
    blink = group["blink_camera_pictures"]
    if blink != sorted(set(blink)) or any(type(point) is not int or point not in range(0, 60, 2) for point in blink):
        raise ValueError("Invalid observed blink picture list")


def expected_truth(group, role, rate):
    if role in CONTROLS:
        return None
    short = role == "short_hold"
    side, start, stop = group["staged_regions"][role.removesuffix("_tandem")]
    if (type(side) is not int or side not in (0, 1) or type(start) is not int or type(stop) is not int
            or start not in (range(12, 20) if short else range(28, 36))
            or stop - start not in (range(10, 14) if short else range(14, 19))):
        raise ValueError("An authored native interval differs from the locked design")
    indices = group["index_maps"][str(rate)]
    return [side, [bisect_left(indices, start), bisect_left(indices, stop)], group["eye_boxes"][side]]


def check_reading(row, rate, truth):
    if (row["truth"] != truth or not valid_digest(row["delivery_sha256"]) or type(row["assessed"]) is not bool
            or Fraction(row["time_base"]) <= 0 or not 0 < row["elapsed_us"] <= 2_000_000):
        raise ValueError("A measured delivery differs from the predeclared recipe")
    raw, shortlist = row["raw"], row["shortlist"]
    for proposal in (raw, shortlist):
        if proposal is None:
            continue
        side, (start, stop), box = proposal
        if (not row["assessed"] or type(side) is not int or side not in (0, 1) or not valid_box(box)
                or not all(type(point) is int for point in (start, stop)) or not 0 < start < stop < rate * 2
                or stop - start not in (range(8, 19) if rate == 25 else range(10, 23))):
            raise ValueError("A reported eye region is outside its measured pictures")

    gap = row["top_collar_gap_hex"]
    first = None if gap is None else float.fromhex(gap)
    if first is not None and (raw is None or not math.isfinite(first) or not -2 <= first <= 2):
        raise ValueError("The unguarded proposal carries invalid nearby motion support")
    top = raw if first is not None and first >= COLLAR_POINT else None
    if (top is not None and shortlist != top or top is None and shortlist is not None and shortlist == raw
            or shortlist is not None and raw is None):
        raise ValueError("The shortlist disagrees with the first qualifying proposal")
    anchors, shortlist_gap = row["source_witnesses"], row["shortlist_gap_hex"]
    if shortlist is None:
        if anchors is not None or shortlist_gap is not None:
            raise ValueError("An empty shortlist result has source evidence")
        return raw, top, None
    if not COLLAR_POINT <= float.fromhex(shortlist_gap) <= 2 or len(anchors) != 3:
        raise ValueError("Invalid nearby motion or source witness")
    start, stop = shortlist[1]
    for expected, (index, tick, frame, crop) in zip((start - 1, (start + stop) // 2, stop), anchors, strict=True):
        if (type(index) is not int or index != expected or type(tick) is not int
                or tick * Fraction(row["time_base"]) != Fraction(index, rate)
                or not valid_digest(frame) or not valid_digest(crop)):
            raise ValueError("A frame clock or crop commitment does not match")
    return raw, top, shortlist


def count_rate(rows, rate) -> tuple[dict, set]:
    counts = {name: Counter() for name in ARMS}
    groups = {name: set() for name in ARMS}
    blink_alerts = {name: 0 for name in ARMS}
    assessed, witnesses, blink_assessed = Counter(), 0, 0
    ordinary_alerts, elapsed, localized_cases = [], [], set()
    for group, source, role, delivery in rows:
        row = dict(zip(FIELDS, delivery, strict=True))
        truth = expected_truth(source, role, rate)
        observed = check_reading(row, rate, truth)
        blink_control = source["blink_camera_pictures"] and role in {"original", "original_tandem"}
        if blink_control:
            blink_assessed += row["assessed"]
        assessed["edited" if truth else "ordinary"] += row["assessed"]
        assessed["hard_cut"] += role == "hard_cut" and row["assessed"]
        witnesses += len(row["source_witnesses"] or ())
        elapsed.append(row["elapsed_us"])
        for name, match in zip(ARMS, observed, strict=True):
            if match is None:
                continue
            if blink_control:
                blink_alerts[name] += 1
            if truth is None:
                counts[name]["ordinary_alerts"] += 1
                counts[name]["whole_alerts"] += role in WHOLE
                if name == "shortlist":
                    ordinary_alerts.append({"group": group, "role": role, "frames": match[1],
                                            "source_range_s": [index / rate for index in match[1]]})
            elif localized(match, truth):
                counts[name]["located"] += 1
                counts[name]["exact_endpoints"] += truth[1] == match[1]
                groups[name].add(group)
                if name == "shortlist":
                    localized_cases.add((group, role))
            else:
                counts[name]["misplaced"] += 1
    fields = ("located", "misplaced", "ordinary_alerts", "whole_alerts", "exact_endpoints")
    return {"arms": {name: {**{field: counts[name][field] for field in fields}, "located_groups": len(groups[name])}
                     for name in ARMS}, "assessed": dict(assessed), "verified_frame_and_crop_pairs": witnesses,
            "ordinary_alerts": ordinary_alerts, "blink_controls_assessed": blink_assessed, "blink_alerts": blink_alerts,
            "median_measurement_s": median(elapsed) / 1_000_000}, localized_cases


def review(path: Path) -> dict:
    value, _ = load(path)
    freeze = value["frozen"]
    if (value["schema"] != SCHEMA or value["rates"] != list(RATES) or value["geometry"] != [480, 360]
            or value["profile"] != "h264-yuv420p" or tuple(value["fields"]) != FIELDS
            or freeze["rates"] != list(RATES) or freeze["gate"] != GATE or freeze["shortlist"] != 8
            or freeze["short_limits_frames_on_parent"] != [10, 13] or freeze["long_limits_frames_on_parent"] != [14, 18]
            or freeze["collar_point_hex"] != COLLAR_POINT.hex() or freeze["texture_point_hex"] != TEXTURE_POINT.hex()
            or len(value["groups"]) != 12 or sum(bool(group["blink_camera_pictures"]) for group in value["groups"]) != 5
            or not all(valid_digest(value[key]) for key in ("native_receipt_sha256", "source_lock_sha256",
                       "edited_lock_sha256", "ordinary_lock_sha256", "natural_lock_sha256",
                       "algorithm_sha256", "prior_algorithm_sha256"))):
        raise ValueError("Unexpected cadence experiment schema or locked rules")
    by_rate = {rate: [] for rate in RATES}
    for group, source in enumerate(value["groups"], 1):
        validate_groups(source)
        for role, *deliveries in source["cases"]:
            for rate, reading in zip(RATES, deliveries, strict=True):
                by_rate[rate].append((group, source, role, reading))
    if (sum(map(len, by_rate.values())) != value["clock_compared_count"]
            or any(sum(role in EDITS for _, _, role, _ in rows) != value["matched_edited_pairs"]
                   or sum(role in CONTROLS for _, _, role, _ in rows) != value["matched_ordinary_pairs"]
                   for rows in by_rate.values())):
        raise ValueError("The receipt omits clock evidence or planned pairs")
    outcomes = {rate: count_rate(by_rate[rate], rate) for rate in RATES}
    result = {rate: pair[0] for rate, pair in outcomes.items()}
    at_25, at_30 = outcomes[25][1], outcomes[30][1]
    paired = {"both": len(at_25 & at_30), "only_25": len(at_25 - at_30), "only_30": len(at_30 - at_25),
              "neither": value["matched_edited_pairs"] - len(at_25 | at_30)}
    subchecks = {}
    for rate in RATES:
        arm = result[rate]["arms"]["shortlist"]
        subchecks[rate] = {"locations": arm["located"] >= GATE[f"minimum_{rate}fps_located"],
                           "groups": arm["located_groups"] >= GATE["minimum_groups_each"],
                           "ordinary": arm["ordinary_alerts"] <= GATE["max_ordinary_each"],
                           "whole_picture": arm["whole_alerts"] <= GATE["max_whole_picture_alerts"]}
    paired_passed = paired["only_25"] <= GATE["max_paired_30fps_loss"]
    return {"rates": result, "paired_edited_locations": paired, "frozen_rate_subchecks": subchecks,
            "frozen_paired_loss_check": paired_passed,
            "frozen_joint_gate_passed": paired_passed and all(all(checks.values()) for checks in subchecks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(parser.parse_args().receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
