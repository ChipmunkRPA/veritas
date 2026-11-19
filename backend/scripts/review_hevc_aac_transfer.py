# SPDX-License-Identifier: Apache-2.0
"""Recount the frozen HEVC face comparison across AAC soundtrack layouts."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

LAYOUTS = {"mono44": (44100, 1), "mono48": (48000, 1), "stereo48": (48000, 2)}
ROLES = {
    "compound_3_4_a": ("localized", (77, 81)),
    "compound_3_4_b": ("localized", (97, 100)),
    "no_finding_clean_swap": ("unassessed", None),
    "ordinary_blur_highlight": ("localized", (73, 76)),
    "ordinary_blur_comparable": ("no_match", None),
    "untouched_camera": ("unassessed", None),
    "ordinary_authored_cut": ("unassessed", None),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def review(receipt):
    require(receipt["schema"] == "chipmunk.hevc-aac-transport/1", "Unexpected transport receipt")
    require(receipt["video"] == {"width": 1280, "height": 720, "frames": 120, "timescale": 90000,
                                 "ticks_per_frame": 3600, "encoding": "hvc1"}, "Unexpected video clock")
    require(set(receipt["commitments"]) == {"design", "native_readback", "observations"}
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in receipt["commitments"].values()),
            "Invalid transport study commitments")
    roster = set()
    summary = Counter()
    marker_max = 0
    for case in receipt["cases"]:
        mode, role = case["layout"], case["role"]
        require(mode in LAYOUTS and role in ROLES and (mode, role) not in roster,
                "Duplicate or unknown soundtrack case")
        roster.add((mode, role))
        rate, channels = LAYOUTS[mode]
        require(case["soundtrack"] == {"rate": rate, "channels": channels, "start_pts": 0,
                                        "end_pts": rate * 24 // 5, "priming_samples": 1024},
                "Soundtrack metadata differs from the tested layout")
        require(all(case[key] is True for key in ("same_pictures", "same_compressed_packets", "same_face_record",
                                                 "same_coverage", "same_incumbent", "same_shots")),
                "A paired transport result differs from the source")
        offsets = case["marker_offsets_at_48k"]
        require(len(offsets) == 3 and all(type(offset) is int and abs(offset) <= 48 for offset in offsets),
                "Audio marker offset exceeds the frozen allowance")
        marker_max = max(marker_max, *(abs(offset) for offset in offsets))
        status, expected = ROLES[role]
        require(case["status"] == status, "Unexpected coverage for the original source role")
        require(case["shot_frames"] == ([67] if role == "ordinary_authored_cut" else []),
                "Shot boundaries differ from the source")
        if expected is None:
            require(case["range"] is None and case["pts"] is None and not case["display_frames"],
                    "An unselected source has an unexpected face range")
            continue
        start, end = expected
        require(case["range"] == [start, end] and case["pts"] == [start * 3600, end * 3600]
                and case["display_frames"] == [start - 1, start + (end - start) // 2, end],
                "Exact source location differs from the silent carrier")
        summary["ordinary_blur_highlights" if role == "ordinary_blur_highlight" else "known_swap_locations"] += 1
    require(roster == {(mode, role) for mode in LAYOUTS for role in ROLES}, "Incomplete transport roster")

    rejected = receipt["out_of_scope"]
    excluded = {case["role"]: case for case in rejected}
    require(len(rejected) == 3 and set(excluded) == {"offset", "short", "dual"}, "Missing ordinary carrier controls")
    for role, case in excluded.items():
        require(case["same_compressed_packets"] is True and case["same_pictures"] is True,
                "An excluded control changed the video")
        require(case["streams"] == (2 if role == "dual" else 1), "Unverified audio track count")
        require(type(case["start_pts"]) is int and type(case["duration_pts"]) is int, "Invalid audio source clock")
        require(case["start_pts"] > 0 if role == "offset" else case["start_pts"] == 0,
                "Incorrect offset control")
        require(0 < case["duration_pts"] < 230400 if role == "short" else case["duration_pts"] >= 230400,
                "Incorrect audio presentation length")

    return {"qualified_muxes": len(roster), "paired_source_roles": len(ROLES), "audio_layouts": len(LAYOUTS),
            "exact_rgb_pts_and_packets": (len(roster) + len(rejected)) * 120,
            "maximum_marker_offset_at_48k": marker_max, "out_of_scope_controls": len(rejected), **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    arg = parser.parse_args()
    print(json.dumps(review(json.loads(arg.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
