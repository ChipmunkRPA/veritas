# SPDX-License-Identifier: Apache-2.0
"""Check resolution trial routing, native clocks and source crops without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

import cv2

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.review_video_short_return_margin import case_status, clock, digest, interval_iou, read, require

NATIVE = [Fraction(i, 25) for i in range(150)]
SIZES = {"720": (1280, 720), "360": (640, 360)}
RANK_POINT = .10
INCUMBENT_POINT = .11243900750879454
DURATIONS = ("3/8", "1/2")
SWAPS = {"swap_a", "swap_b"}
ORDINARY = {"ordinary_light", "ordinary_blur", "ordinary_time_step"}


def span(duration):
    half = Fraction(duration) / 2
    return [Fraction(math.ceil((3 - half) * 25), 25), Fraction(math.ceil((3 + half) * 25), 25)]


def receipt_cases(plan, receipts):
    sources = {}
    for pin in plan["rtfs"]:
        i = pin["index"]
        path = receipts / f"group-{i:03d}-complete.json"
        review_path = receipts / f"group-{i:03d}-command-review-v96/result.json"
        require(digest(path) == pin["receipt_sha256"] and digest(review_path) == pin["review_sha256"],
                "Source qualification differs from the committed roster")
        cases = {case["condition"]: case for case in read(path)["cases"]}
        require(set(cases) == {"original"} | SWAPS | ORDINARY
                and all(clock(case["frame_table"]) == NATIVE and
                        (case["frame_table"]["width"], case["frame_table"]["height"]) == SIZES["720"]
                        for case in cases.values()), "A source group lacks an exact matched clock or condition")
        sources[i] = cases
    return sources


def add_anchors(batch, height, selected, truth, group, row, plan):
    left, right = (Fraction(selected[edge]) for edge in ("start", "end"))
    anchors = selected["anchors"]
    require(set(anchors) == {"before", "during", "after"}
            and left <= Fraction(anchors["during"]["time"]) < right
            and Fraction(anchors["before"]["time"]) < left
            and Fraction(anchors["after"]["time"]) >= right
            and len(selected["scene_correlations"]) == 2
            and all(.55 <= value <= 1.000001 for value in selected["scene_correlations"]),
            "Selected anchors or scene correlations differ from the range")
    index = group["group"] - 1
    if group["family"] == "matched":
        source = plan["rtfs"][index]["index"]
        insertion = span(row["duration"]) if row["duration"] else None
        for anchor in anchors.values():
            time = Fraction(anchor["time"])
            inside = insertion is not None and insertion[0] <= time < insertion[1]
            require(anchor["source_group"] == source and anchor["condition"] == (row["condition"] if inside else "original"),
                    "Evidence role came from the wrong matched source picture")
    else:
        pair = {plan["rtfs"][slot]["index"] for slot in plan["cut_pairs"][index]}
        require(all(anchor["source_group"] in pair and anchor["condition"] == "original"
                    for anchor in anchors.values()) and truth is None, "Cut did not use the paired ordinary pictures")

    for anchor in anchors.values():
        frame = anchor["frame"]
        require(0 <= frame < len(NATIVE) and NATIVE[frame] == Fraction(anchor["time"]),
                "Anchor frame and exact clock disagree")
        key = anchor["source_group"], anchor["condition"]
        batch[key][frame].append((height, anchor))


def verify_pixels(batch, sources, archive):
    counts = Counter()
    for (group, condition), frames in sorted(batch.items()):
        case = sources[group][condition]
        path = archive / "video-native-inserts-v95" / f"group-{group:03d}" / case["file"]
        require(digest(path) == case["sha256"], "Anchor media digest differs")
        table = probe_frames(path, timeout_s=30)
        require([(value - table.pts[0]) * table.tb for value in table.pts] == NATIVE
                and (table.width, table.height) == SIZES["720"] and not (table.rotation or table.interpolated or table.truncated),
                "Reopened anchor source has a different native clock")
        seen = set()
        for decoded in decode_frames(path, table, sorted(frames), timeout_s=110):
            index, picture = decoded.index, decoded.rgb
            require(decoded.pts_verified and index not in seen and index in frames
                    and hashlib.sha256(picture).hexdigest() == case["expected_rgb_sha256"][index],
                    "Reopened native RGB differs from qualification")
            seen.add(index)
            pictures = {"720": picture, "360": cv2.resize(picture, SIZES["360"], interpolation=cv2.INTER_AREA)}
            for height, anchor in frames[index]:
                x, y, w, h = anchor["box"]
                columns, rows = SIZES[height]
                require(0 <= x < x + w <= columns and 0 <= y < y + h <= rows, "Evidence crop exceeds its declared view")
                crop = pictures[height][y:y + h, x:x + w]
                require(anchor["source_rgb_sha256"] == case["expected_rgb_sha256"][index]
                        and anchor["analysis_rgb_sha256"] == hashlib.sha256(pictures[height]).hexdigest()
                        and anchor["crop_rgb_sha256"] == hashlib.sha256(crop.tobytes()).hexdigest(),
                        "Reopened view or evidence crop differs")
                counts[height] += 1
        require(seen == set(frames), "Reopened source is missing an anchor")
    return counts


def audit(folder, archive, receipts):
    lock, journal, aggregate = (folder / name for name in ("freeze.json", "scored.jsonl", "aggregate.json"))
    plan, result = read(lock), read(aggregate)
    runtime = read(folder / "run.json")
    require(result["freeze_sha256"] == digest(lock) and runtime["journal_sha256"] == digest(journal)
            and runtime == result["runtime"] and lock.stat().st_mtime_ns <= journal.stat().st_mtime_ns,
            "Run is not bound to the prospective freeze")
    require((plan["point"], plan["incumbent_point"], plan["maximum_length_exclusive_s"], plan["dedupe_iou"],
             plan["correct_iou"], plan["durations"], plan["cut_pairs"], plan["center"], plan["cadence_hz"]) ==
            (RANK_POINT, INCUMBENT_POINT, "1/2", "1/2", "1/2", list(DURATIONS), [[0, 1], [2, 3], [4, 5]], "3", 8),
            "Threshold, interval, cadence or cut roster differs")
    sources = receipt_cases(plan, receipts)
    require(len(sources) == len(plan["rtfs"]) == 6, "Source roster has duplicates")
    groups = [json.loads(line) for line in journal.read_text().splitlines()]
    require([(group["family"], group["group"]) for group in groups] ==
            [(family, i) for family, n in (("matched", 6), ("cut", 3)) for i in range(1, n + 1)],
            "Missing or duplicate groups")
    batch = defaultdict(lambda: defaultdict(list))
    panels = {}
    for height in SIZES:
        totals, gains = defaultdict(Counter), set()
        windows, errors, overlaps = 0, [], []
        for group in groups:
            family = group["family"]
            if family == "matched":
                expected = Counter({("original", None): 1})
                expected.update((kind, duration) for kind in SWAPS | ORDINARY for duration in DURATIONS)
            else:
                expected = Counter({("source_cut", duration): 2 for duration in DURATIONS})
            require(Counter((row["condition"], row["duration"]) for row in group["rows"]) == expected,
                    "A group does not match its frozen case roster")
            for row in group["rows"]:
                truth = row["truth"]
                require((list(map(Fraction, truth)) == span(row["duration"]) if truth else row["condition"] not in SWAPS)
                        and (truth is None or row["condition"] in SWAPS), "Labeled interval differs")
                current, rank, selected = (row["arms"][height][key] for key in ("current", "ranked", "selected"))
                for arm in (item for item in (current, rank) if item is not None):
                    start, end = (Fraction(arm[edge]) for edge in ("start", "end"))
                    require(math.isfinite(arm["score"]) and start in NATIVE and end in NATIVE and start < end,
                            "A scored range does not lie on the native clock")
                    windows += 1
                view = row["arms"][height]
                require(not (view["multiple_faces"] and rank is not None), "Multiple-face observation was assessed")
                base_flag = current is not None and current["score"] > INCUMBENT_POINT
                baseline = case_status(current, truth)[0] if base_flag else (
                    "unassessed" if current is None else "measured_no_flag")
                keep = rank is not None and rank["score"] > RANK_POINT and (
                    Fraction(rank["end"]) - Fraction(rank["start"]) < Fraction(1, 2)) and (
                    not base_flag or interval_iou(rank, (current["start"], current["end"])) < Fraction(1, 2))
                require(keep == (selected is not None) and (not keep or all(rank[key] == selected[key] for key in rank)),
                        "Selected range differs from independent routing")
                added = case_status(rank, truth)[0] if keep else None
                cohort = f"swap_{row['duration']}" if truth else family
                count = totals[cohort]
                count["views"] += 1
                count[f"current_{baseline}"] += 1
                count["rank_assessed"] += rank is not None
                count["multiple_face_skips"] += view["multiple_faces"]
                count["dense_single_face_samples"] += view["single_face_samples"]
                count["dense_samples"] += view["dense_samples"]
                count["final_localized"] += "localized" in (baseline, added)
                count["final_false_flag"] += "false_flag" in (baseline, added)
                if keep:
                    count[f"secondary_{added}"] += 1
                    add_anchors(batch, height, selected, truth, group, row, plan)
                if added == "localized" and baseline != "localized":
                    count["additional_correct"] += 1
                    overlaps.append(interval_iou(rank, truth))
                    errors.extend(abs(Fraction(rank[key]) - Fraction(label))
                                  for key, label in zip(("start", "end"), truth))
                    if row["duration"] == "3/8":
                        gains.add(group["group"])
        actual = result["by_height"][height]
        require(totals == actual["totals"] and len(gains) == actual["shortest_gain_source_groups"],
                "Independent outcome counts differ")
        coverage = {name: Fraction(sum(totals[key]["rank_assessed"] for key in cohorts),
                                   sum(totals[key]["views"] for key in cohorts))
                    for name, cohorts in (("swap", ("swap_3/8", "swap_1/2")), ("control", ("matched", "cut")))}
        advances = (totals["swap_3/8"]["additional_correct"] >= 6 and len(gains) >= 3
                    and min(coverage.values()) >= Fraction(4, 5)
                    and totals["matched"]["secondary_false_flag"] <= 2 and not totals["cut"]["secondary_false_flag"])
        require({k: str(v) for k, v in coverage.items()} == actual["assessed_coverage"]
                and advances == actual["pilot_advances"], "Independent advancement gate differs")
        panels[height] = {"verified_windows": windows, "median_added_correct_iou": float(median(overlaps)) if overlaps else None,
                          "median_added_endpoint_error_ms": float(1000 * median(errors)) if errors else None,
                          "max_added_endpoint_error_ms": float(1000 * max(errors)) if errors else None}
    verified = verify_pixels(batch, sources, archive)
    for height, panel in panels.items():
        panel["verified_before_during_after_anchors"] = verified[height]
    return {"aggregate_sha256": digest(aggregate), "views_per_size": 78, "by_height": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(audit(args.folder, args.archive, args.receipts), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
