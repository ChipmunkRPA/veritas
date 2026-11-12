# SPDX-License-Identifier: Apache-2.0
"""Audit cut-guard routing and reconstruct every selected transition from source pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import _features
from scripts.review_video_short_return_margin import case_status, clock, digest, interval_iou, read, require

NATIVE = [Fraction(i, 25) for i in range(150)]
DURATIONS = ("3/8", "1/2", "3/4")
CURRENT_POINT = .11243900750879454


def span(duration):
    return [Fraction(math.ceil((3 + sign * Fraction(duration) / 2) * 25), 25) for sign in (-1, 1)]


def source_row(pin, receipts, media, condition):
    receipt = receipts / f"group-{pin['index']:03d}-complete.json"
    require(digest(receipt) == pin["receipt_sha256"], "RTFS source receipt differs")
    case = next(row for row in read(receipt)["cases"] if row["condition"] == condition)
    require(clock(case["frame_table"]) == NATIVE, "RTFS source clock is not exact 25 Hz")
    path = media / f"group-{pin['index']:03d}" / case["file"]
    return path, case["sha256"], case["expected_rgb_sha256"]


def audit(folder, archive, receipts):
    journal = folder / "scored.jsonl"
    freeze = folder / "freeze.json"
    lock, run, result = (read(folder / name) for name in ("freeze.json", "run.json", "aggregate.json"))
    require(digest(freeze) == result["freeze_sha256"] and digest(journal) == run["journal_sha256"]
            and run == result["runtime"] and freeze.stat().st_mtime_ns <= journal.stat().st_mtime_ns,
            "Freeze, journal or runtime commitment differs")
    require((lock["incumbent_point"], lock["content_point"], lock["histogram_point"], lock["end_slop_s"],
             lock["minimum_cut_overlap"], tuple(lock["spans_s"])) == (CURRENT_POINT, 27., .35, "1/4", "1/2", DURATIONS),
            "Frozen operating point or durations differ")
    groups = [json.loads(line) for line in journal.read_text().splitlines()]
    require([g["group"] for g in groups] == [*(f"rtfs_{i}" for i in range(1, 7)), "cut_1", "cut_2", "cut_3", "camera_1"]
            and sum(g["verified_frames"] for g in groups) == run["verified_frames"], "Group roster differs")
    camera_pin, = lock["camera"]
    camera_path = archive / camera_pin["qualification"]
    camera_doc = read(camera_path)
    require(digest(camera_path) == camera_pin["qualification_sha256"], "Camera source receipt differs")
    camera_times = clock(camera_doc)
    media = archive / "video-native-inserts-v95"
    source_cache, pixel_cache = {}, {}

    def feature(path, sha, expected, i, times):
        if path not in source_cache:
            require(digest(path) == sha, "Replay source bytes differ from the qualification")
            table = probe_frames(path, timeout_s=30)
            require(not table.rotation and not table.truncated and not table.interpolated
                    and [(p - table.pts[0]) * table.tb for p in table.pts] == times, "Replay source clock differs")
            source_cache[path] = table
        if (path, i) not in pixel_cache:
            picture, = decode_frames(path, source_cache[path], [i], timeout_s=45)
            require(picture.pts_verified and picture.index == i and hashlib.sha256(picture.rgb).hexdigest() == expected[i],
                    "Replay picture differs from exact source receipt")
            pixel_cache[path, i] = _features(picture.rgb)
        return pixel_cache[path, i]

    def measure(group, position, row, first_index, last_index, times):
        family, kind = group["family"], row["condition"]
        source_index = int(group["group"].split("_")[-1]) - 1

        def frame(i):
            if family == "camera":
                case = camera_doc["cases"][kind]
                return feature(camera_path.parent / case["file"], case["sha256"], case["rgb_sha256"], i, times)
            if family == "rtfs":
                edit = span(row["duration"]) if row["duration"] else None
                active = kind if edit and edit[0] <= times[i] < edit[1] else "original"
                pin = lock["rtfs"][source_index]
            else:
                pair = lock["cut_pairs"][source_index]
                main, insert = pair if position < 4 else pair[::-1]
                if kind == "two_cut":
                    left, right = map(Fraction, row["cut_span"])
                    shifted = left <= times[i] < right
                else:
                    shifted = times[i] >= Fraction(row["cut_at"])
                pin, active = lock["rtfs"][insert if shifted else main], "original"
            path, sha, expected = source_row(pin, receipts, media, active)
            return feature(path, sha, expected, i, times)

        before, first, last, after = (frame(i) for i in (first_index - 1, first_index, last_index - 1, last_index))
        result = {}
        for name, (a, b) in {"entry": (before, first), "exit": (last, after), "flank": (before, after), "middle": (first, last)}.items():
            result[name] = float(np.abs(a[0] - b[0]).mean()), 1. - float(np.minimum(a[1], b[1]).sum())
        return result

    totals, durations = defaultdict(Counter), defaultdict(Counter)
    condition_flags = defaultdict(Counter)
    exact_cut_diagnostics = Counter()
    sources = defaultdict(set)
    reviewed = replayed = 0
    for group in groups:
        family = group["family"]
        rows = group["rows"]
        if family == "rtfs":
            conditions = {"swap_a", "swap_b", "ordinary_light", "ordinary_blur", "ordinary_time_step"}
            expected = {(kind, duration) for kind in conditions for duration in DURATIONS} | {("original", None)}
            require(len(rows) == 16 and {(row["condition"], row["duration"]) for row in rows} == expected, "RTFS roster differs")
        elif family == "camera":
            require(len(rows) == 3 and {row["condition"] for row in rows} == set(camera_doc["cases"]), "Camera roster differs")
        else:
            expected = [*(('two_cut', duration) for duration in DURATIONS), ('single_cut', None)]
            require([(row["condition"], row["duration"]) for row in rows] == expected * 2, "Source-cut roster differs")
        times = camera_times if family == "camera" else NATIVE
        for i, row in enumerate(rows):
            reviewed += 1
            incumbent, selected, truth = (row[key] for key in ("incumbent", "cut_support", "truth"))
            kind = row["condition"]
            require((truth is not None) == (kind in {"swap_a", "swap_b"}), "Swap truth was assigned to an ordinary input")
            if truth:
                require(list(map(Fraction, truth)) == span(row["duration"]), "Swap clock differs from the frozen edit")
            if kind == "two_cut":
                require(list(map(Fraction, row["cut_span"])) == span(row["duration"]), "Cut clock differs from the frozen edit")
            if kind == "single_cut":
                require(Fraction(row["cut_at"]) == 3, "Single-cut clock differs from the frozen edit")
            status = "unassessed" if incumbent is None else "measured_no_flag"
            if incumbent is not None:
                a, b = (Fraction(incumbent[key]) for key in ("start", "end"))
                require(a in times and b in times and a < b and math.isfinite(incumbent["score"]), "Incumbent clock differs")
                if incumbent["score"] > CURRENT_POINT:
                    status = case_status(incumbent, truth)[0]
            cohort = "swap" if truth else kind if family == "cut" else family
            totals[cohort][f"current_{status}"] += 1
            if truth is None:
                condition_flags[kind]["views"] += 1
                condition_flags[kind]["current_flags"] += status == "false_flag"
                condition_flags[kind]["suppressed_flags"] += status == "false_flag" and selected is not None
            if kind == "two_cut" and status == "false_flag":
                c, d = map(Fraction, row["cut_span"])
                measures = measure(group, i, row, times.index(c), times.index(d), times)
                checks = {
                    "endpoints_within_quarter_second": max(abs(c - a), abs(d - b)) <= Fraction(1, 4),
                    "overlap_at_least_half": interval_iou(incumbent, row["cut_span"]) >= Fraction(1, 2),
                    "both_authored_edges_meet_app_cut_points": all(measures[key][0] >= 27 and measures[key][1] >= .35
                                                                   for key in ("entry", "exit")),
                    "authored_flank_and_middle_scene_return": all(measures[key][0] < 27 and measures[key][1] < .35
                                                                 for key in ("flank", "middle")),
                }
                exact_cut_diagnostics["incumbent_paired_cut_flags"] += 1
                exact_cut_diagnostics.update({key: int(value) for key, value in checks.items()})
                exact_cut_diagnostics["all_qualification_checks"] += all(checks.values())
            if selected is not None:
                require(status in {"localized", "wrong_interval", "false_flag"}, "A nonflagged interval was suppressed")
                first, last = selected["indices"]
                require(1 <= first <= last - 2 < last < len(times), "Selected scene range has invalid native indices")
                c, d = (times[k] for k in (first, last))
                require([c, d] == list(map(Fraction, selected["span"])) and max(abs(c - a), abs(d - b)) <= Fraction(1, 4)
                        and Fraction(selected["iou"]) == interval_iou(incumbent, selected["span"]) >= Fraction(1, 2),
                        "Selected cut interval differs from its source clock or incumbent range")
                require(all(selected[key][0] >= 27 and selected[key][1] >= .35 for key in ("entry", "exit"))
                        and all(selected[key][0] < 27 and selected[key][1] < .35 for key in ("flank", "middle")),
                        "Scene thresholds do not support the recorded decision")
                totals[cohort][f"suppressed_{status}"] += 1
                sources[cohort].add(group["group"])
                if kind == "two_cut" and selected["span"] == row["cut_span"]:
                    totals[cohort]["suppressed_with_exact_genuine_cut_boundaries"] += 1
                measures = measure(group, i, row, first, last, times)
                require(all(abs(value - stored) <= 1e-7 for key, pair in measures.items()
                            for value, stored in zip(pair, selected[key], strict=True)), "Stored shot measurement differs from source replay")
                replayed += 1
            else:
                totals[cohort][f"retained_{status}"] += 1
            if row["duration"]:
                key = f"{cohort}@{row['duration']}"
                durations[key][f"current_{status}"] += 1
                if selected:
                    durations[key][f"suppressed_{status}"] += 1
    cuts, swaps = totals["two_cut"], totals["swap"]
    passed = (cuts["current_false_flag"] >= 6 and cuts["suppressed_false_flag"] >= 4
              and 2 * cuts["suppressed_false_flag"] >= cuts["current_false_flag"] and len(sources["two_cut"]) >= 2
              and not swaps["suppressed_localized"] and not totals["single_cut"]["suppressed_false_flag"])
    require(reviewed == 123 and totals == result["totals"] and durations == result["by_duration"]
            and {key: len(v) for key, v in sources.items()} == result["suppressed_source_groups"]
            and passed == result["pilot_advances"], "Aggregate differs from independent decisions")
    return {"reviewed_rows": reviewed, "selected_transitions_replayed": replayed, "distinct_replayed_source_pictures": len(pixel_cache),
            "ordinary_condition_flags": condition_flags,
            "authored_cut_diagnostics": exact_cut_diagnostics,
            "aggregate_sha256": digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.archive, args.receipts), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
