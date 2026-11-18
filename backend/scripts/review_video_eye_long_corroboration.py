# SPDX-License-Identifier: MIT
"""Recount the held and exploratory eye-region corroboration panels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_eye_cadence_transfer import valid_box
from scripts.review_video_eye_long_windows import (
    CONTROL_CHANGES,
    EDITS,
    STARTS,
    STATUSES,
    area,
    first_seen,
    overlap,
    require,
    useful,
)
from scripts.review_video_eye_shortlist_context import load, valid_digest

SCHEMA = "chipmunk.video-eye-long-corroboration-portable/1"
PROFILE = [180, 480, 360, "h264", "yuv420p", "mp4", "1/30000"]
METHOD = {"starts": list(STARTS), "minimum_shift_frames": 8, "max_endpoint_change": 3, "spatial_iou_minimum": [1, 3]}
GATE = {"minimum_sources": 8, "minimum_originals_assessed": 8, "minimum_later_useful": 6,
        "minimum_later_useful_groups": 4, "retained_fraction": [2, 3], "maximum_ordinary_files": 2,
        "maximum_ordinary_groups": 1, "maximum_whole_hold_files": 0}
PANELS = {"held_vcd2": 11, "exploratory_wikitongues": 8}


def alternate_for(row) -> int | None:
    start, stop = row["frames"]
    initial = row["window"]
    choices = [offset for offset in range(121) if offset + 5 <= start and stop <= offset + 55 and abs(offset - initial) >= 8]
    return min(choices, key=lambda offset: (-abs(offset - initial), offset), default=None)


def validate_region(row, window) -> None:
    lo, hi = row["frames"]
    require(row["window"] == window and all(type(x) is int for x in (lo, hi, row["eye"])) and row["eye"] in (0, 1)
            and window + 5 <= lo < hi <= window + 55 and 10 <= hi - lo <= 22
            and valid_box(row["box"]) and min(row["box"][2:]) >= 8, "Eye crop or window-relative range changed")


def consistent(first, second) -> bool:
    if second is None or first["eye"] != second["eye"]:
        return False
    a, b = first["box"], second["box"]
    return (4 * overlap(a, b) >= area(a) + area(b)
            and all(abs(i - j) <= 3 for i, j in zip(first["frames"], second["frames"], strict=True)))


def clock(row) -> list:
    lo, hi = row["frames"]
    return [[index, index * 1000] for index in (lo - 1, (lo + hi) // 2, hi)]


def check_file(row) -> tuple[list, list, int, Counter]:
    states = dict(zip(STARTS, row["window_states"], strict=True))
    require(set(states.values()) <= STATUSES, "Unexpected base window state")
    base = {region["window"]: region for region in row["base"]}
    require(list(base) == [start for start in STARTS if states[start] == "matched"] and len(base) == len(row["base"]),
            "Base window proposals differ")
    for start, region in base.items():
        validate_region(region, start)
    representatives = first_seen(row["base"])
    require([p["window"] for p in representatives] == [p["base_window"] for p in row["comparisons"]],
            "First-seen windows omit an observed region or duplicate a cluster")
    accepted, alternate_starts, counts = [], set(), Counter()
    for first, check in zip(representatives, row["comparisons"], strict=True):
        shifted = alternate_for(first)
        second, status = check["observation"], check["status"]
        require(shifted == check["alternate"] and (second is not None) == (status == "matched"),
                "Corroboration differs from the fixed lawful alternate")
        if shifted is None:
            require(status == "insufficient_displacement", "A boundary-limited proposal has no alternate")
        else:
            require(status in STATUSES, "Unknown shifted-window state")
            alternate_starts.add(shifted)
            if shifted in states:
                require(states[shifted] == status and base.get(shifted) == second, "Cached observation differs from its base window")
            if second is not None:
                validate_region(second, shifted)
        pass_pair = consistent(first, second)
        require(check["clock_triples"] == ([clock(first), clock(second)] if pass_pair else None),
                "The paired evidence anchors differ from the transport clock")
        if pass_pair:
            accepted.append(first)
            counts.update(accepted_pairs=1, anchored_pictures=6)
        else:
            counts["no_alternate" if shifted is None else "different_proposal" if second is not None else "no_match"] += 1
    require(row["measurements"] == len(set(STARTS) | alternate_starts) <= 12, "Unexpected count of distinct native windows")
    return representatives, accepted, sum(state != "no_stable_face" for state in states.values()), counts


def check_panel(panel) -> dict:
    key = panel["key"]
    group_count = PANELS[key]
    held = key == "held_vcd2"
    expected = [(group, role) for group in range(1, group_count + 1) for role in (*EDITS, *CONTROL_CHANGES)
                if not (held and group == 9 and role == "early")]
    rows = panel["cases"]
    require(list(map(lambda row: (row["group"], row["role"]), rows)) == expected
            and all(valid_digest(panel[name]) for name in ("frozen_input_sha256", "native_result_sha256")),
            "The panel omits a frozen camera recording or test role")
    totals = Counter(files=len(rows), source_groups=group_count)
    roles = {}
    later_groups, ordinary_groups = set(), set()
    for row in rows:
        group, role, truth = row["group"], row["role"], row["truth"]
        require((truth is not None) == (role in EDITS), "Staged hold and ordinary control interchanged")
        if truth is not None:
            require(truth["frames"] == EDITS[role] and type(truth["eye"]) is int and truth["eye"] in (0, 1)
                    and valid_box(truth["box"]), "The planted eye or time interval differs")
        primary, accepted, measured, readings = check_file(row)
        opening = next((pick for pick in row["base"] if pick["window"] == 0), None)
        totals.update(readings)
        totals.update(assessed_windows=measured, measured_files=int(measured > 0), unique_windows=row["measurements"],
                      native_replays=int(held or bool(primary)))
        role_counts = roles.setdefault(role, Counter())
        role_counts.update(files=1, measured=int(measured > 0), baseline_alerts=int(bool(primary)), guarded_alerts=int(bool(accepted)))
        if truth is None:
            totals.update(ordinary_files=1, ordinary_assessed=int(measured > 0), opening_ordinary=int(opening is not None),
                          baseline_ordinary=int(bool(primary)), guarded_ordinary=int(bool(accepted)))
            if accepted:
                ordinary_groups.add(group)
            continue
        before, after = (any(useful(p, truth) for p in proposals) for proposals in (primary, accepted))
        role_counts.update(opening_useful=int(opening is not None and useful(opening, truth)),
                           baseline_useful=int(before), guarded_useful=int(after))
        totals.update(staged_files=1, staged_assessed=int(measured > 0))
        if role != "early":
            totals.update(baseline_later=int(before), guarded_later=int(after))
            if after:
                later_groups.add(group)
    totals.update(guarded_later_groups=len(later_groups), guarded_ordinary_groups=len(ordinary_groups))
    return {"totals": totals, "roles": roles}


def review(path: Path) -> dict:
    value, _ = load(path)
    require(value["schema"] == SCHEMA and value["profile"] == PROFILE and value["method"] == METHOD and value["gate"] == GATE
            and [panel["key"] for panel in value["panels"]] == list(PANELS) and valid_digest(value["protocol_sha256"]),
            "The frozen corroboration protocol or panel order differs")
    results = {panel["key"]: check_panel(panel) for panel in value["panels"]}
    held = results["held_vcd2"]
    counts, roles = held["totals"], held["roles"]
    checks = {"source_groups": counts["source_groups"] >= GATE["minimum_sources"],
              "assessed_originals": roles["original"]["measured"] >= GATE["minimum_originals_assessed"],
              "later_count": counts["guarded_later"] >= GATE["minimum_later_useful"],
              "later_groups": counts["guarded_later_groups"] >= GATE["minimum_later_useful_groups"],
              "retention": 3 * counts["guarded_later"] >= 2 * counts["baseline_later"] > 0,
              "ordinary_count": counts["guarded_ordinary"] <= GATE["maximum_ordinary_files"],
              "ordinary_groups": counts["guarded_ordinary_groups"] <= GATE["maximum_ordinary_groups"],
              "whole_picture": roles["whole_picture_hold"]["guarded_alerts"] <= GATE["maximum_whole_hold_files"]}
    return {"panels": results, "held_gate_checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(parser.parse_args().receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
