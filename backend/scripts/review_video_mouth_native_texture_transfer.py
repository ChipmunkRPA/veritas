# SPDX-License-Identifier: Apache-2.0
"""Replay held decisions, picture clocks and native mouth crops independently."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_mouth_native_texture_transfer as trial
from scripts.video_common_sources_v89 import read, require, sha, write


def best_window(values, frames, shots):
    selected = None
    for length in (5, 7, 9, 11):
        for left in range(4, len(frames) - length - 3):
            right = left + length
            if shots[left - 3] != shots[right + 2]:
                continue
            parts = [[j for j in range(a, b) if values[j] is not None]
                     for a, b in ((left - 3, left), (left, right), (right, right + 3))]
            if min(len(parts[0]), len(parts[2])) < 2 or len(parts[1]) * 5 < 4 * length:
                continue
            middles = [float(median(values[j] for j in part)) for part in parts]
            a, middle, b = middles
            signal = min(middle - a, middle - b)
            if selected is None or signal > selected[0]:
                samples = [parts[0][-1], parts[1][len(parts[1]) // 2], parts[2][0]]
                selected = signal, [frames[left], frames[right]], middles, [frames[j] for j in samples]
    return selected


def overlap(left, right):
    a, b = map(Fraction, left)
    c, d = map(Fraction, right)
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def outcome(proposal, point, truth):
    if proposal is None:
        return "unassessed"
    if proposal["score"] <= point:
        return "measured_no_range"
    if not truth:
        return "false_range"
    return "correct_range" if overlap(proposal["source_clock_s"], truth) >= Fraction(1, 2) else "misplaced_range"


def check_crops(path, proposal, times):
    anchors = proposal["anchors"]
    table = probe_frames(path, timeout_s=12)
    require([pts * table.tb for pts in table.pts] == times and table.time_base == proposal["time_base"],
            "Native picture clock differs")
    native = {row["frame"]: row for row in anchors}
    recovered = set()
    for frame in decode_frames(path, table, sorted(native), timeout_s=15):
        bound = native[frame.index]
        x0, y0, x1, y1 = bound["native_crop_xyxy"]
        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        crop = hashlib.sha256(frame.rgb[y0:y1, x0:x1].tobytes()).hexdigest()
        require(frame.pts_verified and frame.pts == bound["pts"] and frame.pts * table.tb == times[frame.index]
                and digest == bound["rgb_sha256"] and crop == bound["crop_rgb_sha256"],
                "Retained picture or crop does not match its source")
        recovered.add(frame.index)
    require(set(native) == recovered, "Incomplete before/during/after pictures")
    return len(anchors)


def audit(folder, prior):
    freeze, calibration, runtime, aggregate = (read(folder / name) for name in
                                               ("freeze.json", "calibration.json", "run.json", "aggregate.json"))
    rows = trial.panel.prior.jsonl(folder / "held.jsonl")
    freeze_path, calibration_path, held_path = (folder / name for name in ("freeze.json", "calibration.json", "held.jsonl"))
    require(freeze == trial.seal(prior) and len(rows) == 160
            and calibration["freeze_sha256"] == aggregate["freeze_sha256"] == sha(freeze_path)
            and runtime["held_sha256"] == sha(held_path) and runtime["calibration_sha256"] == sha(calibration_path)
            and freeze_path.stat().st_mtime_ns <= calibration_path.stat().st_mtime_ns <= held_path.stat().st_mtime_ns,
            "Prior, freeze, calibration, or held commitment differs")
    scores = sorted(row["candidate"]["score"] for row in calibration["ordinary_development"] if row["candidate"])
    require(len(scores) >= 40 and calibration["point"] == aggregate["point"] == scores[-3],
            "Candidate calibration was not the third largest ordinary response")
    roster = trial.roster(prior, read(prior / "freeze.json"))
    reference = trial.baselines(prior)
    seen, counts, compatible = set(), Counter(), Counter()
    anchors = 0
    for row in rows:
        bank, index, condition = (row[name] for name in ("bank", "index", "condition"))
        cohort = condition if bank == "delivery" else bank
        key = bank, index, condition
        component, cases = roster[bank, index]
        path, source = cases[condition]
        previous = reference[key]
        require(key not in seen and row["component"] == component
                and row["source_sha256"] == sha(path) == source["file_sha256"] == previous["sha256"]
                and row["baseline"] == previous["arms"]["direct"], "Source or direct comparison binding differs")
        seen.add(key)
        times = [Fraction(ms, 1000) for ms in source["expected"]["picture_pts_ms"]]
        if source.get("known_ai_provenance") or source.get("neural_provenance_verified"):
            truth = [str(times[j]) for j in source["edit_frames_half_open"]]
        else:
            truth = None
        require(truth == row["truth"] == previous["truth"]
                and row["compatible"] == bool(source.get("motion_compatible", False)), "Truth or compatible case changed")
        picks = trial.panel.sample_indices(times)
        require(list(picks) == row["frames"] and len(picks) == len(row["shots"]) == len(row["scores"])
                and row["measured_centers"] == sum(score is not None for score in row["scores"]), "Exact sample roster differs")
        proposed = best_window(row["scores"], row["frames"], row["shots"])
        candidate = row["candidate"]
        if proposed is None:
            require(candidate is None, "Unexpected candidate")
        else:
            signal, frames, middles, sampled = proposed
            require(candidate["score"] == signal and candidate["frames"] == frames and candidate["medians"] == middles
                    and [anchor["frame"] for anchor in candidate["anchors"]] == sampled
                    and candidate["source_clock_s"] == [str(times[j]) for j in frames], "Selected range differs")
        for arm, record, point in (("direct", row["baseline"], calibration["prior_direct_point"]),
                                   ("candidate", candidate, calibration["point"])):
            state = outcome(record, point, truth)
            counts[cohort, arm, state] += 1
            if row["compatible"]:
                compatible[f"{arm}/{state}"] += 1
            if arm == "candidate" and state in {"correct_range", "misplaced_range", "false_range"}:
                anchors += check_crops(path, record, times)
    published = {(cohort, arm, state): n for cohort, arms in aggregate["outcomes"].items()
                 for arm, states in arms.items() for state, n in states.items()}
    require(counts == published and dict(compatible) == aggregate["compatible"]
            and seen == set(reference), "Public decision counts differ")
    return {"status": "verified", "exact_held_files": len(rows), "replayed_window_decisions": len(rows) * 2,
            "verified_before_during_after_crops": anchors, "report_sha256": sha(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=trial.PRIOR)
    args = parser.parse_args()
    result = audit(args.folder, args.prior)
    write(args.folder / "independent-audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
