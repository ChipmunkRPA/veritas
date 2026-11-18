# SPDX-License-Identifier: MIT
"""Recount the six-second eye-region experiment from anonymous scalar records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median

from scripts.review_video_eye_cadence_transfer import valid_box
from scripts.review_video_eye_shortlist_context import load, valid_digest

SCHEMA = "chipmunk.video-eye-long-windows-portable/1"
STARTS = tuple(range(0, 121, 24))
EDITS = {"early": [17, 31], "straddling": [52, 68], "late": [146, 164]}
CONTROL_CHANGES = {"original": None, "tandem": None, "whole_picture_hold": [104, 120],
                   "same_camera_jump": [76, 93], "cross_camera_cut": [92, 112], "local_eye_blur": [52, 68]}
GATE = {"at_least_distinct_sources": 8, "at_least_measured_original_sources": 6, "at_least_later_useful": 8,
        "at_least_later_useful_groups": 4, "at_most_ordinary_alerts": 2, "max_whole_picture_alerts": 0,
        "all_first_seen_exhibits_native": True}
STATUSES = {"matched", "no_match", "no_stable_face"}


def require(valid: bool, message: str) -> None:
    if not valid:
        raise ValueError(message)


def shared(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap(a, b) -> int:
    return (shared((a[0], a[0] + a[2]), (b[0], b[0] + b[2]))
            * shared((a[1], a[1] + a[3]), (b[1], b[1] + b[3])))


def area(box) -> int:
    return box[2] * box[3]


def useful(proposal, truth) -> bool:
    if truth is None or proposal["eye"] != truth["eye"]:
        return False
    a, b = proposal["frames"], truth["frames"]
    return (2 * shared(a, b) >= max(a[1], b[1]) - min(a[0], b[0])
            and 4 * overlap(proposal["box"], truth["box"]) >= 3 * area(proposal["box"]))


def linked(proposal, other) -> bool:
    a, b = proposal["frames"], other["frames"]
    first, second = proposal["box"], other["box"]
    pixels = overlap(first, second)
    return (proposal["eye"] == other["eye"] and 4 * shared(a, b) >= max(a[1], b[1]) - min(a[0], b[0])
            and 5 * pixels >= area(first) + area(second))


def first_seen(proposals) -> list:
    indexed = {row["window"]: row for row in proposals}
    remaining = set(indexed)
    representatives = []
    while remaining:
        opening = min(remaining)
        edges = {opening}
        representatives.append(indexed[opening])
        while neighbors := {key for key in remaining - edges
                            if any(linked(indexed[key], indexed[member]) for member in edges)}:
            edges.update(neighbors)
        remaining.difference_update(edges)
    return representatives


def review_file(row) -> tuple[list, int]:
    statuses = row["window_states"]
    require(len(statuses) == len(STARTS) and set(statuses) <= STATUSES, "Unexpected window states")
    proposals = row["proposals"]
    require([item["window"] for item in proposals] == [window for window, status in zip(STARTS, statuses, strict=True)
                                                      if status == "matched"], "A selected window is missing or duplicated")
    for proposal in proposals:
        lo, hi = proposal["frames"]
        start = proposal["window"]
        require(all(type(x) is int for x in (lo, hi, proposal["eye"])) and proposal["eye"] in (0, 1)
                and start + 5 <= lo < hi <= start + 55 and 10 <= hi - lo <= 22
                and valid_box(proposal["box"]) and min(proposal["box"][2:]) >= 8, "Window-relative eye crop or range differs")
    first = first_seen(proposals)
    for proposal in proposals:
        lo, hi = proposal["frames"]
        expected = [[index, index * 1000] for index in (lo - 1, (lo + hi) // 2, hi)] if proposal in first else None
        require(proposal["clock_anchors"] == expected, "First-seen evidence differs from the native transport clock")
    return first, sum(state != "no_stable_face" for state in statuses)


def recount(value) -> dict:
    require(value["schema"] == SCHEMA and value["format"] == [180, 480, 360, "h264", "yuv420p", "mp4", "1/30000"]
            and value["window_starts"] == list(STARTS) and value["gate"] == GATE
            and value["intake"] == {"recordings": 16, "eligible": 8, "source_clock_or_crop_exclusions": 6,
                                    "not_downloaded": 2}
            and all(valid_digest(value[key]) for key in ("frozen_sha256", "native_receipt_sha256", "detector_sha256")),
            "Unexpected source or protocol commitment")
    require(all(any(window + 5 <= start and start + size <= window + 55 for window in STARTS)
                for start in range(5, 175) for size in range(10, 23) if start + size <= 175),
            "The schedule leaves an interior supported interval unmeasured")
    rows = value["cases"]
    require([(row["group"], row["role"]) for row in rows] == [(group, role) for group in range(1, 9)
            for role in (*EDITS, *CONTROL_CHANGES)], "A source or ordinary control is missing")
    totals = Counter(files=len(rows))
    roles = {}
    useful_groups, ordinary_groups, measured_original_groups = set(), set(), set()
    ordinary_records = []
    work_times, replay_times = [], []
    for row in rows:
        group, role, truth = row["group"], row["role"], row["truth"]
        require((truth is not None) == (role in EDITS), "A staged edit and ordinary control were interchanged")
        if truth is not None:
            require(truth["frames"] == EDITS[role] and type(truth["eye"]) is int and truth["eye"] in (0, 1)
                    and valid_box(truth["box"]), "The planted eye or interval differs")
        authored = EDITS[role] if role in EDITS else CONTROL_CHANGES[role]
        require(row["changed_pictures"] == (authored[1] - authored[0] if authored else 0), "Unexpected changed-picture count")
        donor = group % 8 + 1 if role == "cross_camera_cut" else None
        require(row["donor"] == donor, "The ordinary cut no longer uses its own distinct camera control")
        first, measured = review_file(row)
        opening = next((p for p in row["proposals"] if p["window"] == 0), None)
        counters = roles.setdefault(role, Counter())
        counters.update(files=1, measured_files=int(measured > 0), measured_windows=measured,
                        opening_alerts=int(opening is not None), overlap_alerts=int(bool(first)))
        totals.update(window_proposals=len(row["proposals"]), grouped_proposals=len(first),
                      windows=len(STARTS), measured_windows=measured, first_seen_anchors=3 * len(first))
        if truth is not None:
            located = [p for p in first if useful(p, truth)]
            counters.update(opening_useful=int(opening is not None and useful(opening, truth)),
                            overlap_useful=int(bool(located)), exact_ranges=int(any(p["frames"] == truth["frames"] for p in located)))
            if role != "early" and located:
                totals["later_useful"] += 1
                useful_groups.add(group)
        else:
            totals.update(ordinary_files=1, ordinary_measured=int(measured > 0),
                          opening_ordinary_alerts=int(opening is not None), overlap_ordinary_alerts=int(bool(first)))
            if role == "original" and measured:
                measured_original_groups.add(group)
            if first:
                ordinary_groups.add(group)
                ordinary_records.extend({"group": group, "role": role, "window": p["window"], "frames": p["frames"],
                                         "overlap_with_authored_control": shared(p["frames"], authored) if authored else None}
                                        for p in first)
        timings = row["score_us"], row["total_us"]
        require(all(type(t) is int for t in timings) and 0 < timings[0] <= timings[1], "Invalid measured run time")
        work_times.append(timings[0])
        replay_times.append(timings[1])
    totals.update(later_useful_groups=len(useful_groups), ordinary_affected_groups=len(ordinary_groups),
                  originals_measured=len(measured_original_groups), source_groups=8, serving_eligible=0)
    checks = {"source_groups": totals["source_groups"] >= GATE["at_least_distinct_sources"],
              "ordinary_assessed": totals["originals_measured"] >= GATE["at_least_measured_original_sources"],
              "later_useful": totals["later_useful"] >= GATE["at_least_later_useful"],
              "later_useful_groups": totals["later_useful_groups"] >= GATE["at_least_later_useful_groups"],
              "ordinary_alerts": totals["overlap_ordinary_alerts"] <= GATE["at_most_ordinary_alerts"],
              "whole_picture": roles["whole_picture_hold"]["overlap_alerts"] <= GATE["max_whole_picture_alerts"],
              "native_anchors": value["verified_native_anchors"] == totals["first_seen_anchors"]}
    return {"counts": totals, "by_role": roles, "ordinary_ranges": ordinary_records, "gate_checks": checks,
            "passed": all(checks.values()), "timing_us": {"score_median": median(work_times),
            "score_p95": sorted(work_times)[ceil(.95 * len(work_times)) - 1], "score_max": max(work_times),
            "total_max": max(replay_times)}}


def review(path: Path) -> dict:
    value, _ = load(path)
    return recount(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(parser.parse_args().receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
