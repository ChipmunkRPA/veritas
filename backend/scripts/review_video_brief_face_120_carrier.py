# SPDX-License-Identifier: Apache-2.0
"""Recompute brief face-proposal additions and native carrier evidence checks."""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-brief-face-120-carrier/1"
SHA = re.compile(r"[0-9a-f]{64}")
CUTS = {"cross_recording_cuts", "person_switch", "tandem_person_switch"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def overlap(first, second):
    return 2 * (min(first[1], second[1]) - max(first[0], second[0])) >= max(first[1], second[1]) - min(first[0], second[0])


def expected_tick(index, row, carrier):
    if carrier == "vp9":
        return row["first_pts"] + (100 * index + 1) // 3
    ticks = Fraction(index, 30) / Fraction(row["time_base"])
    require(ticks.denominator == 1, "Invalid AVC picture lattice")
    return row["first_pts"] + int(ticks)


def review(rows, carrier, design):
    require(len(rows) == 56 and [row["index"] for row in rows] == list(range(56)), "Missing held files")
    totals = Counter()
    groups = set()
    for row in rows:
        require(row["bank"] in ("edits", "cuts") and type(row["group"]) is int and type(row["first_pts"]) is int
                and all(SHA.fullmatch(row[key]) for key in ("source_sha256", "delivery_sha256"))
                and (carrier != "vp9" or Fraction(row["time_base"]) == Fraction(1, 1000)), "Invalid case binding")
        measured = row["measurement"]
        require(type(measured["available"]) is bool and 0 <= measured["face_frames"] <= measured["scheduled"] <= 84
                and SHA.fullmatch(measured["native_pts_sha256"]), "Invalid native coverage")
        ticks = tuple(expected_tick(i, row, carrier) for i in range(120))
        require(hashlib.sha256(json.dumps(ticks).encode()).hexdigest() == measured["native_pts_sha256"], "Incomplete native clock")
        totals["measured"] += measured["available"]
        totals["routed_without_previous_return"] += not row["retained_short"]
        truth = row["truth"]
        targets = [span for span in truth if span[1] - span[0] == 6]
        require(len(targets) == (1 if truth else 0) and all(0 <= s < e <= 120 for s, e in truth + row["prior"]),
                "Invalid known or previous range")
        before = [span for span in targets if any(overlap(span, found) for found in row["prior"])]
        totals["ordinary"] += not truth
        totals["target_six_picture_ranges"] += len(targets)
        totals["incumbent_six_picture_locations"] += len(before)
        proposal = row["proposal"]
        if not proposal:
            continue
        start, end = proposal["frames"]
        require(not row["retained_short"] and measured["available"] and proposal["response"] > design["point"]
                and 6 <= start < end <= 114 and end - start in (5, 6, 7)
                and proposal["pts"] == [ticks[start], ticks[end]] and not any(overlap((start, end), p) for p in row["prior"]),
                "Invalid additional range or source ticks")
        a, b, c = proposal["anchors"]
        require(a["frame_index"] == start - 1 and start <= b["frame_index"] < end and c["frame_index"] == end,
                "Invalid source evidence order")
        for anchor in proposal["anchors"]:
            require(anchor["pts"] == ticks[anchor["frame_index"]] and all(
                SHA.fullmatch(anchor[key]) for key in ("decoded_rgb_sha256", "aligned_rgb_sha256")), "Invalid source evidence")
        hits = [span for span in targets if span not in before and overlap((start, end), span)]
        totals["added_six_picture_locations"] += len(hits)
        totals["additional_ordinary_flags"] += not truth
        totals["additional_cut_flags"] += row["condition"] in CUTS
        totals["additional_wrong_edited_ranges"] += bool(truth) and not any(overlap((start, end), span) for span in truth)
        if hits:
            groups.add(row["group"])
            totals["exact_additions"] += proposal["frames"] in hits
    totals["gained_groups"] = len(groups)
    require(totals["ordinary"] == 40 and totals["target_six_picture_ranges"] == 16, "Incorrect evaluation mix")
    totals["final_six_picture_locations"] = totals["incumbent_six_picture_locations"] + totals["added_six_picture_locations"]
    valid = (totals["measured"] >= design["min_measured_files"]
             and totals["added_six_picture_locations"] >= design["minimum_added_six_picture_ranges"]
             and totals["gained_groups"] >= design["minimum_distinct_gained_groups"]
             and totals["exact_additions"] >= design["minimum_exact_additions"]
             and totals["additional_ordinary_flags"] <= design["maximum_new_ordinary_flags"]
             and totals["additional_cut_flags"] <= design["maximum_new_speaker_or_cross_recording_cut_flags"]
             and totals["additional_wrong_edited_ranges"] <= design["maximum_wrong_edited_ranges"])
    return dict(totals) | {"passes": valid}


def native(path, row, carrier):
    with path.open("rb") as handle:
        require(hashlib.file_digest(handle, "sha256").hexdigest() == row["delivery_sha256"], "Delivery bytes differ")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "rgb24",
        "-threads", "1", "-f", "framehash", "-hash", "sha256", "-"], text=True, timeout=30).splitlines()
    require(f"#tb 0: {row['time_base']}" in raw, "Independent decode time base differs")
    frames = [line.replace(" ", "").split(",") for line in raw if not line.startswith("#")]
    require(len(frames) == 120 and all(int(frame[2]) == expected_tick(i, row, carrier) for i, frame in enumerate(frames)),
            "Independent native frame times differ")
    for anchor in (row["proposal"] or {}).get("anchors", []):
        actual = frames[anchor["frame_index"]]
        require(int(actual[2]) == anchor["pts"] and actual[5] == anchor["decoded_rgb_sha256"], "Independent evidence image differs")
    return len((row["proposal"] or {}).get("anchors", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    document = json.loads(args.receipt.read_bytes())
    require(document["freeze"]["schema"] == SCHEMA, "Unknown face proposal transfer")
    design = document["freeze"]["design"]
    held = document["held"]
    require(set(held) == {"avc", "vp9"}, "Both carriers are required")
    for avc, vp9 in zip(held["avc"], held["vp9"], strict=True):
        keys = ("source_sha256", "bank", "group", "condition", "truth")
        require(all(avc[key] == vp9[key] for key in keys), "Paired source commitments differ")
    result = {carrier: review(rows, carrier, design) for carrier, rows in held.items()}
    for carrier, rows in held.items():
        expected = {"clocks": 120 * len(rows), "exact_frames": sum(len((r["proposal"] or {}).get("anchors", ())) for r in rows)}
        require(document["independent_native"][carrier] == expected, "Independent native replay counts differ")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
