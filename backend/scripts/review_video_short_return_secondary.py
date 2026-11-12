# SPDX-License-Identifier: Apache-2.0
"""Independently verify the prospective secondary range decisions and clocks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from scripts.review_video_short_return_margin import (
    case_status,
    clock,
    digest,
    interval_iou,
    read,
    require,
    roster,
)

NATIVE = [Fraction(i, 25) for i in range(150)]
INCUMBENT_POINT = .11243900750879454
RANK_POINT = .10


def native_span(center, duration):
    return [Fraction(math.ceil((center + sign * duration / 2) * 25), 25) for sign in (-1, 1)]


def observed_clock(group, freeze, archive, receipts):
    family = group["family"]
    position = int(group["group"].split("_")[-1]) - 1
    if family == "cut":
        return NATIVE
    if family == "camera":
        pin = freeze[family][position]
        path = archive / pin["qualification"]
        require(digest(path) == pin["qualification_sha256"], "Camera receipt changed")
        times = clock(read(path))
        require(times == sorted(set(times)) and times[0] == 0, "Invalid camera receipt clock")
        return times
    pin = freeze[family][position]
    path = receipts / f"group-{pin['index']:03d}-complete.json"
    require(digest(path) == pin["receipt_sha256"], "RTFS receipt changed")
    original = next(row for row in read(path)["cases"] if row["condition"] == "original")
    require(clock(original["frame_table"]) == NATIVE, "RTFS receipt does not bind a native 25 Hz clock")
    return NATIVE


def audit(folder, archive, receipts):
    plan, journal = folder / "freeze.json", folder / "scored.jsonl"
    freeze, run, result = read(plan), read(folder / "run.json"), read(folder / "aggregate.json")
    require(digest(plan) == result["freeze_sha256"] and digest(journal) == run["journal_sha256"]
            and result["runtime"] == run and plan.stat().st_mtime_ns <= journal.stat().st_mtime_ns, "Run commitment differs")
    require((freeze["point"], freeze["incumbent_point"], freeze["dedupe_iou"], freeze["maximum_length_exclusive_s"])
            == (RANK_POINT, INCUMBENT_POINT, "1/2", "1/2"), "Frozen operating points differ")
    require(len(freeze["rtfs"]) == len({pin["index"] for pin in freeze["rtfs"]}) == 6
            and len(freeze["camera"]) == len({read(archive / pin["qualification"])["source_sha256"]
                                              for pin in freeze["camera"]}) == 4, "Duplicate source group")
    groups = [json.loads(line) for line in journal.read_text().splitlines()]
    expected = [f"{kind}_{i}" for kind, n in (("rtfs", 6), ("camera", 4), ("cut", 3)) for i in range(1, n + 1)]
    require([group["group"] for group in groups] == expected
            and sum(group["verified_frames"] for group in groups) == run["verified_frames"], "Group coverage differs")
    totals, phases = defaultdict(Counter), defaultdict(Counter)
    gains, gain_groups, false_groups = Counter(), defaultdict(set), defaultdict(set)
    evidence_ious, edge_errors = [], []
    rows = preserved = verified_windows = anchors = 0
    for group in groups:
        name, family = group["group"], group["family"]
        require(name.startswith(f"{family}_"), "Unexpected family label")
        roster(group, freeze["phases"])
        times = observed_clock(group, freeze, archive, receipts)
        native = set(times)
        for row in group["rows"]:
            rows += 1
            truth, arms = row["truth"], row["arms"]
            if row["condition"] in {"swap_a", "swap_b"}:
                require(truth and list(map(Fraction, truth)) == native_span(Fraction(row["center"]), Fraction(row["duration"])),
                        "Expected swap does not match its exact input interval")
            else:
                require(truth is None, "Genuine control was marked as a swap")
            if family == "cut":
                require(list(map(Fraction, row["ordinary_cut_span"])) == native_span(Fraction(3), Fraction(row["duration"])),
                        "Ordinary source cut does not match its exact interval")
            for arm in (value for value in arms.values() if value is not None):
                left, right = (Fraction(arm[key]) for key in ("start", "end"))
                require(math.isfinite(arm["score"]) and left in native and right in native and left < right,
                        "Scored window does not use increasing native endpoints")
                verified_windows += 1
            current, rank = arms["current"], arms["rank"]
            has_current = current is not None and current["score"] > INCUMBENT_POINT
            baseline = case_status(current, truth)[0] if has_current else (
                "unassessed" if current is None else "measured_no_flag")
            preserved += has_current
            cohort = "swap" if truth else family
            totals[cohort][f"current_{baseline}"] += 1
            secondary = None
            if rank is not None and rank["score"] > RANK_POINT and Fraction(rank["end"]) - Fraction(rank["start"]) < Fraction(1, 2):
                if has_current and interval_iou(rank, [current["start"], current["end"]]) >= Fraction(1, 2):
                    totals[cohort]["suppressed_duplicate"] += 1
                else:
                    secondary, overlap = case_status(rank, truth)
                    totals[cohort][f"secondary_{secondary}"] += 1
                    totals[cohort][f"secondary_{secondary}_with_current_{baseline}"] += 1
                    if secondary == "false_flag":
                        false_groups[cohort].add(name)
                    left, right = (Fraction(rank[key]) for key in ("start", "end"))
                    evidence = rank["anchors"]
                    require(len(rank["scene_correlations"]) == 2 and all(.55 <= x <= 1.000001 for x in rank["scene_correlations"]),
                            "Scene guard is inconsistent with the selected interval")
                    require(len(evidence) == 3 and all(0 <= item["index"] < len(times)
                            and times[item["index"]] == Fraction(item["time"]) for item in evidence), "Selected picture clock differs")
                    a, b, c = (Fraction(item["time"]) for item in evidence)
                    require(a < left <= b < right <= c, "Picture anchors do not surround the selected interval")
                    anchors += 3
                    if secondary == "localized" and baseline != "localized":
                        evidence_ious.append(overlap)
                        start, stop = map(Fraction, truth)
                        edge_errors.extend((abs(left - start), abs(right - stop)))
            useful = baseline == "localized" or secondary == "localized"
            totals[cohort]["at_least_one_correct"] += useful
            if truth:
                phase = phases[f"{row['center']}@{row['duration']}"]
                phase[f"current_{baseline}"] += 1
                phase["at_least_one_correct"] += useful
                if secondary == "localized" and baseline != "localized":
                    phase["added_correct"] += 1
                    if row["duration"] == "3/8":
                        key = "wrong" if baseline == "wrong_interval" else "absent"
                        gains[key] += 1
                        gain_groups[key].add(name)
    passes = (sum(gains.values()) >= 18 and len(gain_groups["absent"] | gain_groups["wrong"]) >= 4
              and all(gains[key] >= 4 and len(gain_groups[key]) >= 2 for key in ("absent", "wrong"))
              and sum(totals[key]["secondary_false_flag"] for key in ("rtfs", "camera")) <= 2
              and totals["cut"]["secondary_false_flag"] == 0)
    require(rows == 180 and totals == result["totals"] and phases == result["swaps_by_center_and_duration"]
            and gains == result["shortest_gains_by_current_status"] and preserved == result["incumbent_flags_preserved"]
            and passes == result["pilot_advances"] and {key: len(value) for key, value in gain_groups.items()}
            == result["shortest_gain_source_groups_by_current_status"] and {key: len(value) for key, value in false_groups.items()}
            == result["additional_false_source_groups_by_cohort"], "Independent outcome arithmetic differs")
    return {"rows": rows, "native_window_endpoints_verified": verified_windows, "secondary_picture_anchors_verified": anchors,
            "incumbent_flags_preserved": preserved, "median_added_correct_iou": float(median(evidence_ious)) if evidence_ious else None,
            "median_added_correct_endpoint_error_ms": float(1000 * median(edge_errors)) if edge_errors else None,
            "maximum_added_correct_endpoint_error_ms": float(1000 * max(edge_errors)) if edge_errors else None,
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
