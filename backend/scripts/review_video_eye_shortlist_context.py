# SPDX-License-Identifier: MIT
"""Recount native eye-region shortlist evidence without source media or weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

SCHEMA = "chipmunk.video-eye-shortlist-context-evaluation/1"
COLLAR_POINT = .100
FIELDS = (
    "group", "role", "clock", "assessed", "existing_hold", "truth", "unguarded_top", "top_collar_passes",
    "guarded_shortlist", "selected_collar_gap_hex", "source_witnesses", "measurement_us", "delivery_sha256",
)
PANELS = {"historical_1": (30, 60), "historical_2": (27, 58), "fresh": (27, 68)}
EDIT_ROLES = {"left_hold": (0, [16, 27]), "right_hold": (1, [28, 41]), "right_hold_tandem": (1, [28, 41])}
WHOLE_ROLES = {"whole_hold", "picture_jump", "hard_cut"}
ORDINARY_ROLES = WHOLE_ROLES | {"original", "original_tandem", "eye_blur", "eye_gain"}
ARM_NAMES = ("unguarded_top", "top_only_collar", "guarded_shortlist")
FRESH_GATE = {
    "minimum_new_useful": 3, "minimum_new_groups": 2, "maximum_added_ordinary": 2,
    "maximum_new_global": 0, "maximum_extra_misplaced": 1,
}
RETROSPECTIVE_GATE = {"minimum_new_useful": 1, "maximum_added_ordinary": 2, "maximum_new_global": 0}


def valid_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def valid_range(value) -> bool:
    eye, (start, stop), (x, y, width, height) = value
    return (type(eye) is int and eye in (0, 1) and all(type(n) is int for n in (start, stop, x, y, width, height))
            and 0 < start < stop < 50 and 0 <= x < x + width <= 480 and 0 <= y < y + height <= 360)


def localized(reading, truth) -> bool:
    if reading is None or truth is None or reading[0] != truth[0]:
        return False
    _, (start, stop), (x, y, w, h) = reading
    _, (first, end), (tx, ty, tw, th) = truth
    overlap = max(0, min(stop, end) - max(start, first))
    pixels = max(0, min(x + w, tx + tw) - max(x, tx)) * max(0, min(y + h, ty + th) - max(y, ty))
    return 2 * overlap >= max(stop, end) - min(start, first) and 4 * pixels >= 3 * w * h


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    result = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return json.loads(result), hashlib.sha256(raw).hexdigest()


def normalize(reading):
    return [reading["eye"], reading["frames"], reading["box"]] if reading is not None else None


def historical_matches(row, previous) -> bool:
    context = previous["context"]
    passes = context is not None and context["collar_gap"] >= COLLAR_POINT
    return (previous["role"] == row["role"] and previous["sha256"] == row["delivery_sha256"]
            and normalize(previous["truth"]) == row["truth"] and normalize(previous["texture"]) == row["unguarded_top"]
            and passes == row["top_collar_passes"] and normalize(previous["candidate"]) == (
                row["unguarded_top"] if passes else None))


def readings(row) -> tuple:
    raw, passing, chosen = (row[key] for key in ("unguarded_top", "top_collar_passes", "guarded_shortlist"))
    expected = EDIT_ROLES.get(row["role"])
    truth = row["truth"]
    if (row["role"] not in EDIT_ROLES and row["role"] not in ORDINARY_ROLES
            or (truth is not None) != (expected is not None)
            or expected is not None and (not valid_range(truth) or (truth[0], truth[1]) != expected)
            or any(type(row[key]) is not bool for key in ("assessed", "existing_hold", "top_collar_passes"))
            or not valid_digest(row["delivery_sha256"])
            or raw is not None and (not valid_range(raw) or not row["assessed"])
            or chosen is not None and (not valid_range(chosen) or raw is None)
            or passing != (chosen is not None and chosen == raw)):
        raise ValueError("Unexpected source role, landmark coverage or proposal order")
    gap = row["selected_collar_gap_hex"]
    anchors = row["source_witnesses"]
    if chosen is None:
        if gap is not None or anchors is not None:
            raise ValueError("Unexpected evidence for an empty shortlist result")
    else:
        value = float.fromhex(gap)
        start, stop = chosen[1]
        if not math.isfinite(value) or not COLLAR_POINT <= value <= 2 or len(anchors) != 3:
            raise ValueError("The adjacent motion or source witness does not qualify")
        for expected_index, (index, pts, frame_hash, crop_hash) in zip((start - 1, (start + stop) // 2, stop),
                                                                     anchors, strict=True):
            if (type(index) is not int or index != expected_index or type(pts) is not int
                    or pts * Fraction(row["clock"]) != Fraction(index, 25)
                    or not valid_digest(frame_hash) or not valid_digest(crop_hash)):
                raise ValueError("Source frame, clock or crop commitment does not match")
    return raw, raw if passing else None, chosen


def count_panel(panel, historical=None) -> dict:
    key = panel["key"]
    arm_counts = {name: Counter() for name in ARM_NAMES}
    useful_groups = {name: set() for name in ARM_NAMES}
    counts, added, ordinary_motion = Counter(), Counter(), Counter()
    groups, added_groups, delivered = set(), set(), set()
    times, ordinary_roles = [], []
    reviewed = set(panel["ordinary_motion_review_groups"])
    if historical is not None:
        prior, digest = historical
        if digest != panel["held_prior_sha256"] or len(prior["rows"]) != len(panel["rows"]):
            raise ValueError("The original historical panel does not match the receipt")
    for index, values in enumerate(panel["rows"]):
        row = dict(zip(FIELDS, values, strict=True))
        group, role, clock, assessed, existing = (row[field] for field in FIELDS[:5])
        truth = row["truth"]
        if (type(group) is not int or group < 1 or (group, role) in delivered or not Fraction(clock) > 0
                or not 0 < row["measurement_us"] <= 2_000_000
                or role == "hard_cut" and key != "fresh"
                or historical is not None and not historical_matches(row, prior["rows"][index])):
            raise ValueError("Unexpected source group, native cursor, or historical row")
        groups.add(group)
        delivered.add((group, role))
        times.append(row["measurement_us"])
        counts["edits" if truth else "ordinary"] += 1
        counts["edits_assessed" if truth else "ordinary_assessed"] += assessed
        if not truth:
            counts["existing_ordinary_alerts"] += existing

        raw, previous, chosen = readings(row)
        for name, mark in zip(ARM_NAMES, (raw, previous, chosen), strict=True):
            if mark is None:
                continue
            tally = arm_counts[name]
            if truth is None:
                tally["ordinary_alerts"] += 1
                tally["whole_picture_alerts"] += role in WHOLE_ROLES
            elif localized(mark, truth):
                tally["useful"] += 1
                tally["exact_endpoints"] += mark[1] == truth[1]
                useful_groups[name].add(group)
            else:
                tally["misplaced"] += 1

        if chosen:
            counts["verified_frame_and_crop_pairs"] += 3
        if chosen and previous is None:
            if truth is None:
                added["ordinary"] += 1
                added["whole_picture"] += role in WHOLE_ROLES
                ordinary_roles.append(role)
            elif localized(chosen, truth):
                added["useful"] += 1
                added_groups.add(group)
            else:
                added["misplaced"] += 1
        if group in reviewed and role in ("original", "original_tandem"):
            ordinary_motion["controls"] += 1
            ordinary_motion["assessed"] += assessed
            ordinary_motion["shortlist_alerts"] += chosen is not None

    if (groups != set(range(1, 11)) or (counts["edits"], counts["ordinary"]) != PANELS[key]
            or reviewed and (key != "fresh" or len(reviewed) != 6 or ordinary_motion["controls"] != 12)
            or not valid_digest(panel["native_outcomes_sha256"])):
        raise ValueError("The source groups or outcome counts do not match the fixed panel")
    fields = ("useful", "ordinary_alerts", "whole_picture_alerts", "misplaced", "exact_endpoints")
    return {"counts": dict(counts), "arms": {name: {**{field: n[field] for field in fields},
                                                      "useful_groups": len(useful_groups[name])}
                                             for name, n in arm_counts.items()},
            "added": {**{key: added[key] for key in ("useful", "ordinary", "whole_picture", "misplaced")},
                      "groups": len(added_groups)},
            "added_ordinary_roles": ordinary_roles, "ordinary_motion": dict(ordinary_motion),
            "median_measurement_seconds": median(times) / 1_000_000}


def review(path: Path, histories: list[Path] | tuple[Path, ...] = ()) -> dict:
    value, _ = load(path)
    if (value["schema"] != SCHEMA or tuple(value["fields"]) != FIELDS
            or [panel["key"] for panel in value["panels"]] != list(PANELS)
            or value["profile"] != {"width": 480, "height": 360, "pictures": 50, "fps": 25, "codec": "h264-yuv420p"}
            or value["fresh_gate"] != FRESH_GATE or value["retrospective_gate"] != RETROSPECTIVE_GATE
            or value["method"]["shortlist_size"] != 8
            or value["method"]["minimum_time_intersection_over_union"] != [1, 2]
            or value["method"]["minimum_crop_inside_truth"] != [3, 4]
            or value["method"]["minimum_adjacent_motion_gap_hex"] != COLLAR_POINT.hex()
            or not all(valid_digest(digest) for name, digest in value["method"].items() if name.endswith("_sha256"))
            or len(histories) not in (0, 2)):
        raise ValueError("Unexpected eye shortlist receipt or frozen procedure")
    originals = [load(history) for history in histories]
    panels = {panel["key"]: count_panel(panel, originals[index] if index < len(originals) else None)
              for index, panel in enumerate(value["panels"])}
    old = Counter()
    for name in ("historical_1", "historical_2"):
        old.update(panels[name]["added"])
    fresh = panels["fresh"]["added"]
    passed_retrospective = (old["useful"] >= RETROSPECTIVE_GATE["minimum_new_useful"]
                            and old["ordinary"] <= RETROSPECTIVE_GATE["maximum_added_ordinary"]
                            and old["whole_picture"] <= RETROSPECTIVE_GATE["maximum_new_global"])
    passed_fresh = (fresh["useful"] >= FRESH_GATE["minimum_new_useful"]
                    and fresh["groups"] >= FRESH_GATE["minimum_new_groups"]
                    and fresh["ordinary"] <= FRESH_GATE["maximum_added_ordinary"]
                    and fresh["whole_picture"] <= FRESH_GATE["maximum_new_global"]
                    and fresh["misplaced"] <= FRESH_GATE["maximum_extra_misplaced"])
    return {"panels": panels, "historical_references_verified": len(originals),
            "retrospective_gate_passed": passed_retrospective, "fresh_gate_passed": passed_fresh}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--historical", type=Path, nargs=2, default=())
    args = parser.parse_args()
    print(json.dumps(review(args.receipt, args.historical), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
