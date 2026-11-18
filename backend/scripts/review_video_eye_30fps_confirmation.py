# SPDX-License-Identifier: MIT
"""Recount the independent 30 fps eye-region confirmation without media or weights."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from scripts.review_video_eye_cadence_transfer import (
    CONTROLS,
    EDITS,
    FIELDS,
    TEXTURE_POINT,
    count_rate,
    expected_truth,
    valid_box,
)
from scripts.review_video_eye_shortlist_context import COLLAR_POINT, load, valid_digest

SCHEMA = "chipmunk.video-eye-30fps-confirmation-portable/1"
PANELS = {
    "actor_confirmation": (16, 7, {"minimum_30fps_located": 8, "minimum_groups": 4, "max_ordinary": 2,
                                    "max_whole_picture_alerts": 0}),
    "cross_corpus_support": (5, 4, {"minimum_30fps_located": 3, "minimum_groups": 2, "max_ordinary": None,
                                    "max_whole_picture_alerts": 0}),
}
SHA_FIELDS = ("source_sha256", "edited_roles_sha256", "ordinary_roles_sha256", "visual_lock_sha256",
              "native_receipt_sha256")


def measure_group(source, number: int, total: int) -> tuple[list, int]:
    ticks = source["camera_relative_pts"]
    time_base = Fraction(source["camera_time_base"])
    if (len(ticks) != 60 or ticks[0] != 0 or any(type(tick) is not int for tick in ticks)
            or any((b - a) * time_base not in (Fraction(33, 1_000), Fraction(34, 1_000))
                   for a, b in pairwise(ticks))):
        raise ValueError("The source window is missing recorded camera pictures or exact timestamps")
    group = source | {"index_maps": {"30": list(range(60))}, "blink_camera_pictures": source["observed_motion"]}
    if (len(source["eye_boxes"]) != 2 or not all(map(valid_box, source["eye_boxes"]))
            or set(source["staged_regions"]) != {"short_hold", "long_hold"}
            or len(source["cases"]) != 10 or {row[0] for row in source["cases"]} != EDITS | CONTROLS):
        raise ValueError("A capture differs from the frozen 30 fps matched roles")
    motion = source["observed_motion"]
    if motion != sorted(set(motion)) or any(type(frame) is not int or frame not in range(0, 60, 2) for frame in motion):
        raise ValueError("The native picture numbers used in the unedited motion review do not match")
    edits = source["changed_pixel_counts"]
    if set(edits) != {"short_hold", "long_hold"}:
        raise ValueError("Missing actual pixel changes for a staged source patch")
    changed = 0
    for role, differences in edits.items():
        _, (start, stop), (_, _, width, height) = expected_truth(group, role, 30)
        if len(differences) != stop - start or any(type(n) is not int or not 0 < n <= width * height for n in differences):
            raise ValueError("The pre-encode pixels do not match the authored eye region")
        changed += len(differences)
    rows = []
    for role, donor, observation in source["cases"]:
        if donor != (number % total + 1 if role == "hard_cut" else None):
            raise ValueError("The hard cut differs from its distinct-source role")
        rows.append((number, group, role, observation))
    return rows, changed


def count_panel(panel) -> dict:
    people, marked, gate = PANELS[panel["key"]]
    if (panel["gate"] != gate or len(panel["groups"]) != people
            or sum(bool(group["observed_motion"]) for group in panel["groups"]) != marked
            or not all(valid_digest(panel[field]) for field in SHA_FIELDS)):
        raise ValueError("The source bank or advancement checks differ from the freeze")
    rows, changed = [], 0
    for group, source in enumerate(panel["groups"], 1):
        captures, changed_count = measure_group(source, group, people)
        rows.extend(captures)
        changed += changed_count
    if len(rows) != panel["cases"] or changed != panel["physical_changed_pictures"]:
        raise ValueError("The source bank omits a frozen role or an authored picture")
    outcome, _ = count_rate(rows, 30)
    shortlist = outcome["arms"]["shortlist"]
    checks = {"locations": shortlist["located"] >= gate["minimum_30fps_located"],
              "groups": shortlist["located_groups"] >= gate["minimum_groups"],
              "whole_picture": shortlist["whole_alerts"] <= gate["max_whole_picture_alerts"]}
    if gate["max_ordinary"] is not None:
        checks["ordinary"] = shortlist["ordinary_alerts"] <= gate["max_ordinary"]
    natural = {"source_groups": marked, "assessed": outcome.pop("blink_controls_assessed"),
               "alerts": outcome.pop("blink_alerts")}
    return outcome | {"source_groups": people, "staged_edits": people * 3, "ordinary_controls": people * 7,
                      "reviewed_natural_motion": natural, "actual_changed_pictures": changed,
                      "frozen_checks": checks, "passed": all(checks.values())}


def review(path: Path) -> dict:
    value, _ = load(path)
    if (value["schema"] != SCHEMA or (value["cadence"], value["decoded_pictures"]) != (30, 60)
            or value["geometry"] != [480, 360] or value["profile"] != "h264-yuv420p" or value["shortlist_size"] != 8
            or value["collar_point_hex"] != COLLAR_POINT.hex() or value["texture_point_hex"] != TEXTURE_POINT.hex()
            or not valid_digest(value["historic_plan_sha256"]) or tuple(value["fields"]) != FIELDS
            or [panel["key"] for panel in value["panels"]] != list(PANELS)):
        raise ValueError("Unexpected confirmation corpus, method or source profile")
    panels = {panel["key"]: count_panel(panel) for panel in value["panels"]}
    return {"panels": panels, "both_predeclared_checks_passed": all(result["passed"] for result in panels.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(parser.parse_args().receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
