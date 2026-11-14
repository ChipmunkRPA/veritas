# SPDX-License-Identifier: Apache-2.0
"""Recompute four-second AVC face gains and replay native evidence frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, expected_tick, overlap, require

SCHEMA = "chipmunk.video-brief-face-avc-confirmation/1"
ROLES = ("original", "tandem_original", "swap", "tandem_swap", "lighting", "same_recording_jump", "cross_recording_cut")
POSITIVES = {"swap", "tandem_swap"}
PLACEMENTS = (63, 72, 81, 90, 99, 108)
GATES = {
    "minimum_useful_additions": 4,
    "minimum_distinct_gained_groups": 3,
    "minimum_exact_additions": 2,
    "maximum_new_ordinary_flags": 1,
    "maximum_new_ordinary_cut_flags": 0,
    "maximum_new_wrong_edited_ranges": 1,
    "minimum_candidate_coverage": 0.75,
}


def summarize(rows, *, placements=PLACEMENTS):
    ordered = [(group, condition) for group in range(1, len(placements) + 1) for condition in ROLES]
    require([(row["group"], row["condition"]) for row in rows] == ordered, "Held source-role roster is incomplete")
    require(len({row["sha256"] for row in rows}) == len(rows), "Repeated delivery commitment")
    keys = ("files", "baseline_flags", "baseline_useful", "eligible", "measured", "additional_ranges", "additional_useful",
            "additional_exact", "additional_ordinary", "additional_cut", "additional_wrong", "final_useful")
    totals = Counter(dict.fromkeys(keys, 0))
    strata = {condition: Counter(dict.fromkeys(keys, 0)) for condition in ROLES}
    gained = set()
    for row in rows:
        group, condition, truth, prior, proposal = (row[key] for key in ("group", "condition", "truth", "prior", "proposal"))
        start = placements[group - 1]
        require(truth == ([[start, start + 6]] if condition in POSITIVES else []), "Known edit differs from frozen construction")
        require(type(row["first_pts"]) is int and SHA.fullmatch(row["sha256"]), "Invalid source commitment")
        require(type(row["eligible"]) is bool and isinstance(row["baseline_status"], str)
                and all(0 <= start < stop < 120 for start, stop in prior), "Invalid incumbent outcome")
        ticks = tuple(expected_tick(i, row, "avc") for i in range(120))
        measurement = row["measurement"]
        require(row["eligible"] == (measurement is not None), "Missing eligible measurements")
        if measurement:
            require(type(measurement["available"]) is bool and 0 <= measurement["face_frames"] <= measurement["scheduled"] <= 84
                    and measurement["native_pts_sha256"] == hashlib.sha256(json.dumps(ticks).encode()).hexdigest(),
                    "Invalid candidate coverage or native time commitment")
        before = bool(truth and any(overlap(span, truth[0]) for span in prior))
        after = exact = False
        if proposal:
            left, right = proposal["frames"]
            require(row["eligible"] and measurement["available"] and 6 <= left < right <= 114 and right - left in (5, 6, 7)
                    and 0.10 < proposal["response"] <= 2.000000000001
                    and proposal["pts"] == [ticks[left], ticks[right]] and not any(overlap((left, right), r) for r in prior),
                    "Invalid additional range")
            a, b, c = (anchor["frame_index"] for anchor in proposal["anchors"])
            require(a == left - 1 and left <= b < right and c == right, "Invalid before, during and after indices")
            for anchor in proposal["anchors"]:
                require(anchor["pts"] == ticks[anchor["frame_index"]] and all(
                    SHA.fullmatch(anchor[key]) for key in ("decoded_rgb_sha256", "aligned_rgb_sha256")), "Invalid evidence hash or tick")
            after = bool(truth and overlap((left, right), truth[0]))
            exact = [left, right] in truth
        if after and not before:
            gained.add(group)
        changes = {
            "files": 1, "baseline_flags": bool(prior), "baseline_useful": before, "eligible": row["eligible"],
            "measured": bool(measurement and measurement["available"]), "additional_ranges": bool(proposal),
            "additional_useful": after and not before, "additional_exact": exact and not before,
            "additional_ordinary": bool(proposal and not truth),
            "additional_cut": bool(proposal and condition in {"same_recording_jump", "cross_recording_cut"}),
            "additional_wrong": bool(proposal and truth and not after), "final_useful": before or after,
        }
        totals.update(changes)
        strata[condition].update(changes)
    totals["gained_groups"] = len(gained)
    coverage = Fraction(totals["measured"], totals["eligible"]) if totals["eligible"] else Fraction()
    tests = {
        "useful_additions": totals["additional_useful"] >= GATES["minimum_useful_additions"],
        "distinct_gained_groups": totals["gained_groups"] >= GATES["minimum_distinct_gained_groups"],
        "exact_additions": totals["additional_exact"] >= GATES["minimum_exact_additions"],
        "ordinary_flags": totals["additional_ordinary"] <= GATES["maximum_new_ordinary_flags"],
        "ordinary_cut_flags": totals["additional_cut"] <= GATES["maximum_new_ordinary_cut_flags"],
        "wrong_edited_ranges": totals["additional_wrong"] <= GATES["maximum_new_wrong_edited_ranges"],
        "candidate_coverage": coverage >= Fraction(str(GATES["minimum_candidate_coverage"])),
    }
    return {"totals": dict(totals), "strata": {key: dict(values) for key, values in strata.items()},
            "checks": tests, "passes": all(tests.values())}


def replay_native(path, row, expected_rgb):
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"], "Encoded delivery differs")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "rgb24",
        "-threads", "1", "-f", "framehash", "-hash", "sha256", "-"], text=True, timeout=30).splitlines()
    require(f"#tb 0: {row['time_base']}" in raw, "Independent native picture time base differs")
    frames = [line.replace(" ", "").split(",") for line in raw if not line.startswith("#")]
    require(len(frames) == 120 and all(int(frame[2]) == expected_tick(i, row, "avc") for i, frame in enumerate(frames)),
            "Independent picture clock differs")
    require([frame[5] for frame in frames] == expected_rgb, "Independent native RGB sequence differs")
    anchors = (row["proposal"] or {}).get("anchors", [])
    for anchor in anchors:
        index = anchor["frame_index"]
        require(frames[index][5] == anchor["decoded_rgb_sha256"], "Source evidence frame differs")
    return len(anchors)


def compare_authoring(first, second):
    first_freeze = (first / "freeze.json").read_bytes()
    require(first_freeze == (second / "freeze.json").read_bytes(), "Repeated authoring froze a different procedure")
    first_media = json.loads((first / "media.json").read_bytes())
    second_media = json.loads((second / "media.json").read_bytes())
    require(first_media == second_media, "Repeated media commitments differ")
    for row in second_media["rows"]:
        with (second / "media" / row["file"]).open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"], "Repeated delivery bytes differ")
    return len(second_media["rows"])


def assemble(folder, repeated):
    freeze = json.loads((folder / "freeze.json").read_bytes())
    media = json.loads((folder / "media.json").read_bytes())
    rows = list(map(json.loads, (folder / "scored.jsonl").read_text().splitlines()))
    require(freeze["gates"] == GATES and freeze["point"] == .10 and freeze["schema"] == SCHEMA, "Different prospective freeze")
    require(hashlib.sha256((folder / "freeze.json").read_bytes()).hexdigest() == media["freeze_sha256"], "Private media roster differs")
    result = summarize(rows)
    evidence = 0
    for row, recorded in zip(rows, media["rows"], strict=True):
        require(all(row[key] == recorded[key] for key in ("group", "condition", "truth", "sha256")), "Private delivery differs")
        evidence += replay_native(folder / "media" / recorded["file"], row, recorded["rgb_sha256"])
    kept = ("prior_sha256", "qualified_sha256", "method_sha256", "face_weights", "flags", "opencv", "binaries", "gates", "point")
    return {"schema": SCHEMA, "protocol": {key: freeze[key] for key in kept}, "private_freeze_sha256": media["freeze_sha256"],
            "rows": rows, "independent_native": {"clock_ticks": 120 * len(rows), "rgb_frames": 120 * len(rows), "anchors": evidence,
                                                "repeated_authoring_files": compare_authoring(folder, repeated)},
            "result": result}


def verify(document):
    require(document["schema"] == SCHEMA and document["protocol"]["point"] == .10 and document["protocol"]["gates"] == GATES
            and SHA.fullmatch(document["private_freeze_sha256"]), "Incorrect protocol commitment")
    rows = document["rows"]
    result = summarize(rows)
    frames = 120 * len(rows)
    expected = {"clock_ticks": frames, "rgb_frames": frames,
                "anchors": sum(len(row["proposal"]["anchors"]) for row in rows if row["proposal"]),
                "repeated_authoring_files": len(rows)}
    require(document["independent_native"] == expected and document["result"] == result, "Reported counts differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--authoring-repeat", type=Path)
    args = parser.parse_args()
    if args.assemble:
        if args.authoring_repeat is None:
            parser.error("--assemble requires --authoring-repeat")
        value = assemble(args.assemble, args.authoring_repeat)
        with args.receipt.open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
    else:
        value = json.loads(args.receipt.read_bytes())
    print(json.dumps(verify(value), sort_keys=True))


if __name__ == "__main__":
    main()
