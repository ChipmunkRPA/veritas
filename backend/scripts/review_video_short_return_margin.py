# SPDX-License-Identifier: Apache-2.0
"""Recompute the short-return trial decisions and check native clocks without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(valid, detail):
    if not valid:
        raise ValueError(detail)


def clock(document):
    pts = document.get("pts", document.get("relative_pts"))
    return [(p - pts[0]) * Fraction(document["time_base"]) for p in pts]


def native_span(center, duration):
    return [Fraction(math.ceil((center + side * duration / 2) * 30), 30) for side in (-1, 1)]


def interval_iou(record, comparison):
    left, right = map(Fraction, comparison)
    start, stop = (Fraction(record[key]) for key in ("start", "end"))
    intersection = max(Fraction(0), min(right, stop) - max(left, start))
    return intersection / (max(right, stop) - min(left, start))


def case_status(record, truth):
    if truth is None:
        return "false_flag", None
    iou = interval_iou(record, truth)
    return "localized" if iou >= Fraction(1, 2) else "wrong_interval", iou


def samples_for(family, position, lock, archive, receipts):
    if family == "cut":
        return [Fraction(i, 30) for i in range(180)]
    if family == "camera":
        pin = lock["camera"][position]
        path = archive / pin["qualification"]
        expected = pin["qualification_sha256"]
        document = read(path)
    else:
        pin = lock["rtfs"][position]
        path = receipts / f"group-{pin['index']:03d}-complete.json"
        expected = pin["receipt_sha256"]
        document = next(case["frame_table"] for case in read(path)["cases"] if case["condition"] == "original")
    require(digest(path) == expected, "Native clock receipt differs from the sealed roster")
    times = clock(document)
    require(times == sorted(set(times)) and times[0] == 0, "Native clock is not strictly increasing from zero")
    if family == "rtfs":
        require(times == [Fraction(i, 30) for i in range(180)], "RTFS receipt has a different frame clock")
    return times


def roster(group, phases):
    if group["family"] == "rtfs":
        conditions = {"ordinary_light", "ordinary_blur", "ordinary_time_step", "swap_a", "swap_b"}
        expected = {(kind, center, duration) for kind in conditions for center, duration in phases}
        observed = [(row["condition"], row["center"], row["duration"]) for row in group["rows"]]
        require(len(observed) == 26 and set(observed) == expected | {("original", None, None)}, "RTFS roster differs")
    elif group["family"] == "camera":
        observed = [row["condition"] for row in group["rows"]]
        require(len(observed) == 3 and set(observed) == {"unchanged", "chroma_conversion_only", "recompression_only"},
                "Camera delivery roster differs")
    else:
        require(Counter((row["condition"], row["duration"]) for row in group["rows"])
                == {("ordinary_source_cut", "1/2"): 2, ("ordinary_source_cut", "3/4"): 2}, "Genuine cut roster differs")


def audit(folder, archive, receipts):
    freeze = folder / "freeze.json"
    journal = folder / "scored.jsonl"
    lock, runtime, summary = (read(folder / name) for name in ("freeze.json", "run.json", "aggregate.json"))
    require(digest(freeze) == summary["freeze_sha256"] and digest(journal) == runtime["journal_sha256"]
            and summary["runtime"] == runtime and freeze.stat().st_mtime_ns <= journal.stat().st_mtime_ns,
            "Freeze, journal or result commitment differs")
    require(lock["point"] == summary["point"] == .10 and lock["maximum_length_exclusive_s"] == "1/2"
            and lock["incumbent_point"] == .11243900750879454, "Operating point differs")
    require(len(lock["rtfs"]) == len({pin["index"] for pin in lock["rtfs"]}) == 6
            and len(lock["camera"]) == len({read(archive / pin["qualification"])["source_sha256"]
                                            for pin in lock["camera"]}) == 4, "Held source roster contains a duplicate")
    groups = [json.loads(line) for line in journal.read_text().splitlines()]
    planned_groups = [(family, f"{family}_{index}") for family, count in (("rtfs", 6), ("camera", 4), ("cut", 3))
                      for index in range(1, count + 1)]
    require([(group["family"], group["group"]) for group in groups] == planned_groups
            and sum(group["verified_frames"] for group in groups) == runtime["verified_frames"], "Input groups differ")
    counts = defaultdict(lambda: defaultdict(Counter))
    by_phase, gain_groups = defaultdict(Counter), defaultdict(set)
    adds, false_groups = Counter(), set()
    direct, direct_false_groups, dedup_false_groups = Counter(), set(), set()
    added_ious, endpoint_errors = [], []
    old_flags = rows = native_windows = anchors = 0
    for group in groups:
        family, label = group["family"], group["group"]
        roster(group, lock["phases"])
        times = samples_for(family, int(label.split("_")[-1]) - 1, lock, archive, receipts)
        native = set(times)
        for row in group["rows"]:
            rows += 1
            truth, arms = row["truth"], row["arms"]
            if row["condition"] in {"swap_a", "swap_b"}:
                require(truth is not None and list(map(Fraction, truth)) == native_span(Fraction(row["center"]), Fraction(row["duration"])),
                        "Swap truth differs from the predeclared exact native span")
            else:
                require(truth is None, "An ordinary condition has swap truth")
            if family == "cut":
                require(list(map(Fraction, row["ordinary_cut_span"])) == native_span(Fraction(3), Fraction(row["duration"])),
                        "Ordinary cut differs from its predeclared native span")
            for result in arms.values():
                if result is None:
                    continue
                start, stop = (Fraction(result[key]) for key in ("start", "end"))
                require(math.isfinite(result["score"]) and start in native and stop in native and start < stop,
                        "Scored interval lacks increasing exact native endpoints")
                native_windows += 1
            current, rank = arms["current"], arms["rank"]
            cohort = "swap" if truth else family
            baseline = "unassessed" if current is None else "measured_no_flag"
            merged = "unassessed" if current is None and rank is None else "measured_no_flag"
            phase = f"{row.get('center')}@{row['duration']}"
            if current is not None and current["score"] > lock["incumbent_point"]:
                baseline = merged = case_status(current, truth)[0]
                old_flags += 1
            elif rank is not None and rank["score"] > .10 and Fraction(rank["end"]) - Fraction(rank["start"]) < Fraction(1, 2):
                merged, overlap = case_status(rank, truth)
                adds[cohort] += 1
                left, right = (Fraction(rank[key]) for key in ("start", "end"))
                require(len(rank["scene_correlations"]) == 2
                        and all(math.isfinite(x) and x >= .55 for x in rank["scene_correlations"]), "Selected scene guard differs")
                selected = rank["anchors"]
                require(len(selected) == 3 and all(0 <= x["index"] < len(times) and times[x["index"]] == Fraction(x["time"])
                                                  for x in selected),
                        "Selected source picture anchor differs")
                before, during, after = (Fraction(anchor["time"]) for anchor in selected)
                require(before < left <= during < right <= after, "Before/during/after anchors do not surround the selected range")
                anchors += 3
                if merged == "localized":
                    by_phase[phase]["added_correct"] += 1
                    gain_groups[phase].add(label)
                    added_ious.append(overlap)
                    a, b = map(Fraction, truth)
                    endpoint_errors += [abs(left - a), abs(right - b)]
                if merged == "false_flag":
                    false_groups.add(label)
            counts[cohort]["current"][baseline] += 1
            counts[cohort]["additive"][merged] += 1
            if truth:
                by_phase[phase][f"current_{baseline}"] += 1
                by_phase[phase][f"additive_{merged}"] += 1
            # Descriptive follow-up only. The frozen combined route preserves an incumbent flag without a second interval.
            if rank is not None and rank["score"] > .10 and Fraction(rank["end"]) - Fraction(rank["start"]) < Fraction(1, 2):
                rank_status, _ = case_status(rank, truth)
                current_flag = current is not None and current["score"] > lock["incumbent_point"]
                duplicate = current_flag and interval_iou(rank, [current["start"], current["end"]]) >= Fraction(1, 2)
                if truth and row["duration"] == "3/8" and rank_status == "localized":
                    direct["correct_shortest"] += 1
                    direct[f"correct_shortest_with_incumbent_{baseline}"] += 1
                    direct[f"correct_shortest_{'duplicate' if duplicate else 'distinct'}"] += 1
                    if not duplicate:
                        direct[f"distinct_correct_shortest_with_incumbent_{baseline}"] += 1
                elif not truth:
                    direct[f"{family}_false_views"] += 1
                    direct[f"{family}_false_views_with_{'incumbent_flag' if current_flag else 'no_incumbent_flag'}"] += 1
                    direct_false_groups.add(label)
                    if not duplicate:
                        direct[f"{family}_false_views_after_dedup"] += 1
                        dedup_false_groups.add(label)
    shortest = [f"{center}@3/8" for center in ("3", "35/12", "73/24")]
    passes = (all(by_phase[key]["added_correct"] >= 6 and len(gain_groups[key]) >= 3 for key in shortest)
              and adds["rtfs"] + adds["camera"] <= 2 and adds["cut"] == 0)
    require(rows == 180 and counts == summary["totals"] and by_phase == summary["swap_outcomes_by_center_and_duration"]
            and {key: len(values) for key, values in gain_groups.items() if values} == summary["gain_group_counts_by_center_and_duration"]
            and adds == summary["added_flags_by_cohort"] and old_flags == summary["incumbent_flags_preserved"]
            and passes == summary["pilot_advances"], "Independently recomputed results differ from the published readout")
    for family in ("rtfs", "camera", "cut"):
        for count in ("false_views", "false_views_with_incumbent_flag", "false_views_with_no_incumbent_flag",
                      "false_views_after_dedup"):
            direct.setdefault(f"{family}_{count}", 0)
    return {"rows": rows, "native_windows_verified": native_windows, "new_frame_anchors_verified": anchors,
            "incumbent_flags_preserved": old_flags, "groups_with_additional_false_flags": len(false_groups),
            "posthoc_rank_regardless_of_incumbent": {**direct, "false_source_groups": len(direct_false_groups),
                "false_source_groups_after_dedup": len(dedup_false_groups), "duplicate_interval_min_iou": .50},
            "minimum_added_correct_iou": float(min(added_ious)) if added_ious else None,
            "median_added_correct_iou": float(median(added_ious)) if added_ious else None,
            "median_added_correct_endpoint_error_ms": float(1000 * median(endpoint_errors)) if endpoint_errors else None,
            "maximum_added_correct_endpoint_error_ms": float(1000 * max(endpoint_errors)) if endpoint_errors else None,
            "aggregate_sha256": digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.archive, args.receipts), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
