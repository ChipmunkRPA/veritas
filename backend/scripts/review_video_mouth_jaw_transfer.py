# SPDX-License-Identifier: Apache-2.0
"""Independently check mouth-motion decisions, source clocks, and measured crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from PIL import Image
from scripts import review_video_mouth_reconstruction_transfer as replay
from scripts import video_mouth_jaw_transfer as trial
from scripts.video_common_sources_v89 import read, require, sha


def overlap(left, right):
    a, b = map(Fraction, left)
    c, d = map(Fraction, right)
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def outcome(finding, arm, threshold, truth):
    if finding is None:
        return "unassessed"
    left, center, right = finding["medians"]
    expected = min(center-left, center-right)
    if arm == "jaw_coupling":
        expected = max(expected, min(left-center, right-center))
    require(finding["score"] == expected, "Saved score differs from native flank arithmetic")
    if expected <= threshold:
        return "measured_no_range"
    if truth is None:
        return "false_range"
    return "correct_range" if overlap(finding["source_clock_s"], truth) >= Fraction(1, 2) else "misplaced_range"


def records(folder, plan, controls):
    native = trial.native_groups(Path(plan["archive"]))
    roster = {}
    for index, component, cases in trial.grid_cases(plan, "held"):
        roster["grid", index] = (component, {
            row["condition"]: (path, {**row, "motion_compatible": row["condition"] == "liveportrait_source_time"
                                     and motion == trial.prior.SOURCE_CLOCK})
            for path, row, motion in cases
        })
    for pin in plan["native"]:
        index = pin["index"]
        rows = native[index]
        require(sha(trial.DATA / pin["journal"]) == pin["journal_sha256"] and [row["file_sha256"] for row in rows] == pin["files"],
                "The native confirmation source receipt differs from freeze")
        roster["native", index] = (index, {row["condition"]: (Path(plan["archive"]) / row["file"], row) for row in rows})
    for row in controls["roster"]:
        roster.setdefault(("delivery", row["index"]), (None, {}))[1][row["condition"]] = (
            folder / "controls" / row["file"], {**row, "file_sha256": row["sha256"]})
    return roster


def audit(folder, frozen_worker=None):
    freeze = folder / "freeze.json"
    if frozen_worker is None:
        plan = trial.frozen(folder)
    else:
        plan = read(freeze)
        worker_key = str(Path(trial.__file__).relative_to(trial.ROOT))
        require(plan["bindings"] == {**trial.bindings(), worker_key: sha(frozen_worker)}
                and plan["gate"] == trial.GATE and plan["spec"] == trial.SPEC
                and plan["prior_grid_sha256"] == sha(trial.PRIOR_GRID), "Historical frozen worker or methods differ")
    calibration, controls, runtime = (read(folder / filename) for filename in ("calibration.json", "controls.json", "run.json"))
    journal = folder / "scored.jsonl"
    require(sha(freeze) == calibration["freeze_sha256"] == controls["freeze_sha256"]
            and runtime["journal_sha256"] == sha(journal) and runtime["calibration_sha256"] == sha(folder / "calibration.json")
            and runtime["controls_sha256"] == sha(folder / "controls.json")
            and freeze.stat().st_mtime_ns <= (folder / "calibration.json").stat().st_mtime_ns <= journal.stat().st_mtime_ns,
            "Freeze, development or held commitment differs")
    for arm in trial.ARMS:
        values = sorted(row["arms"][arm]["score"] for row in calibration["ordinary_development"] if row["arms"][arm] is not None)
        require(calibration["points"][arm] == values[-3], "Development operating point was not fixed by the frozen rule")
    roster = records(folder, plan, controls)
    totals = defaultdict(lambda: {arm: Counter() for arm in trial.ARMS})
    subtypes = defaultdict(lambda: {arm: Counter() for arm in trial.ARMS})
    additional, compatible, groups = Counter(), Counter(), defaultdict(set)
    observed, qualified, cases, anchors = set(), set(), 0, 0
    cut_guard = Counter()
    selected = None
    for group in trial.prior.jsonl(journal):
        bank, index = group["bank"], group["index"]
        component, lookup = roster[bank, index]
        require(group["component"] == component, "Source assignment differs from freeze")
        local = {}
        for case in group["cases"]:
            condition = case["condition"]
            path, source = lookup[condition]
            key = (bank, index, condition)
            require(key not in observed and path == Path(case["file"]) and sha(path) == case["sha256"] == source["file_sha256"],
                    "Missing, duplicate or changed source input")
            observed.add(key)
            local[condition] = case["arms"]
            cases += 1
            expected_pts = [Fraction(ms, 1000) for ms in source["expected"]["picture_pts_ms"]]
            edited = source.get("known_ai_provenance") or source.get("neural_provenance_verified")
            truth = [expected_pts[i] for i in source["edit_frames_half_open"]] if edited else None
            require(case["truth"] == (list(map(str, truth)) if truth else None)
                    and case["source_time_compatible"] == source.get("motion_compatible", False),
                    "Ground truth or visible-motion qualification differs from freeze")
            cohort = condition if bank == "delivery" else bank
            results, findings = {}, []
            for arm in trial.ARMS:
                best = case["arms"][arm]
                result = outcome(best, arm, calibration["points"][arm], truth)
                results[arm] = result
                totals[cohort][arm][result] += 1
                subtypes[f"{bank}/{condition}"][arm][result] += 1
                if case["source_time_compatible"]:
                    compatible[f"{arm}/{result}"] += 1
                if result in {"correct_range", "misplaced_range", "false_range"}:
                    findings.append(best)
            baseline, candidate = (results[arm] for arm in trial.ARMS)
            best_new = case["arms"]["jaw_coupling"]
            if cohort == "hard_cut":
                cut_guard["controls_with_detected_boundary"] += case["cuts"] > 0
                cut_guard["false_ranges_on_missed_cuts"] += case["cuts"] == 0 and candidate == "false_range"
            if candidate == "correct_range" and baseline != "correct_range":
                groups[bank].add(component)
                additional[f"{bank}_correct"] += 1
                if case["source_time_compatible"]:
                    additional["motion_compatible_correct"] += 1
            if candidate == "false_range" and (baseline != "false_range" or overlap(best_new["source_clock_s"],
                      case["arms"]["direct"]["source_clock_s"]) < Fraction(1, 2)):
                additional[f"{cohort}_false"] += 1
            if truth is not None and candidate != "unassessed":
                additional[f"{bank}_positive_measured"] += 1
            if truth is None and candidate != "unassessed":
                additional[f"{bank}_control_measured"] += 1
            if findings:
                frames = replay.verify_anchors(path, source, findings)
                for finding in findings:
                    for anchor in finding["anchors"]:
                        x0, y0, x1, y1 = anchor["native_crop_xyxy"]
                        pixels = frames[anchor["frame"]][y0:y1, x0:x1]
                        require(hashlib.sha256(pixels.tobytes()).hexdigest() == anchor["crop_rgb_sha256"],
                                "The displayed crop differs from its measured native source pixels")
                        anchors += 1
                if truth and candidate in {"correct_range", "misplaced_range"} and (selected is None or
                          (selected[0] != "correct_range" and candidate == "correct_range")):
                    selected = candidate, best_new, truth, frames
            if edited:
                qualified.add(key)
        if "audio_delay_320ms" in local:
            require(local["audio_delay_320ms"] == local["original"], "The visual measurement responded to audio-only retiming")
            additional["identical_audio_only_pairs"] += 1
    expected = {(bank, index, condition) for (bank, index), (_, lookup) in roster.items() for condition in lookup}
    require(observed == expected and cases == 160 and len(qualified) == 52, "Held roster is incomplete")
    coverage = {"grid_neural": str(Fraction(additional["grid_positive_measured"], 32)),
                "grid_ordinary": str(Fraction(additional["grid_control_measured"], 56)),
                "native_neural": str(Fraction(additional["native_positive_measured"], 20)),
                "native_ordinary": str(Fraction(additional["native_control_measured"], 40))}
    advances = (additional["grid_correct"] >= 6 and len(groups["grid"]) >= 3 and additional["motion_compatible_correct"] >= 2
                and additional["grid_false"] + additional["reencode_false"] <= 3 and additional["hard_cut_false"] == 0
                and additional["native_correct"] >= 4 and len(groups["native"]) >= 3 and additional["native_false"] <= 2
                and min(map(Fraction, coverage.values())) >= Fraction(4, 5))
    report = {"schema": "chipmunk.video-mouth-jaw-transfer/1", "freeze_sha256": sha(freeze), "runtime": runtime, "gate": plan["gate"],
        "cases": cases, "sampled_native_pictures": sum(case["sample_count"] for group in trial.prior.jsonl(journal) for case in group["cases"]),
        "aggregate": totals, "condition_breakdown": subtypes, "motion_compatible_original_time_subset": compatible,
        "additional": additional, "groups_with_additions": {key: len(value) for key, value in groups.items()},
        "candidate_assessed_coverage": coverage, "exact_crop_and_anchor_replays": anchors,
        "sampled_shot_guard_on_hard_cut_controls": cut_guard, "pilot_advances": advances}
    if selected:
        status, finding, truth, frames = selected
        report["example"] = {"status": status, "range_s": list(map(float, map(Fraction, finding["source_clock_s"]))),
            "ground_truth_s": list(map(float, truth)), "iou": float(overlap(finding["source_clock_s"], truth)),
            "anchor_source_seconds": [float(Fraction(anchor["pts"]) * Fraction(finding["time_base"])) for anchor in finding["anchors"]]}
        exhibits = folder / "private-example"
        exhibits.mkdir(mode=0o700, exist_ok=True)
        for anchor in finding["anchors"]:
            pixels = frames[anchor["frame"]]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            for label, rgb in (("original", pixels), ("crop", pixels[y0:y1, x0:x1])):
                path = exhibits / f"{anchor['role']}-{label}.png"
                if path.exists():
                    with Image.open(path) as image:
                        require(image.convert("RGB").tobytes() == rgb.tobytes(), "Existing evidence picture differs from replay")
                else:
                    Image.fromarray(rgb).save(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--frozen-worker", type=Path, help="Exact historical worker snapshot for an archived run.")
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.frozen_worker), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
