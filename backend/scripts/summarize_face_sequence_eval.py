"""Match review rates on development clips, then compare held-out edit localization."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean, median

from prepare_video_upgrade_eval import sha
from summarize_face_interval_eval import load

from app.detectors.video.decode import FrameTable
from app.evidence.face_intervals import RULE, IntervalRule, bind_tracks, localize

MODES = ("current", "dense-fixed", "dense", "sequence")


def input_rows(manifest: Path, baseline: Path, candidate: Path, split: str) -> list[dict]:
    rows = load(manifest, baseline, split)
    with candidate.open() as stream:
        header = json.loads(next(stream))
        proposals = [json.loads(line) for line in stream]
    if (header["manifest_sha256"] != sha(manifest) or header["baseline_sha256"] != sha(baseline)
            or header["split"] != split or [r["id"] for r in rows] != [p["id"] for p in proposals]):
        raise ValueError("Candidate receipt differs from the baseline inputs")
    for row, proposal in zip(rows, proposals, strict=True):
        if row.get("sha256") != proposal.get("source_sha256"):
            raise ValueError("Candidate input bytes differ from the baseline")
        row["candidate"] = proposal
        if row.get("points"):
            table = FrameTable.from_json(row["table"])
            row["dense_tracks"] = bind_tracks(row["points"] + proposal.get("dense_points", []),
                                              [*row["exhibits"], proposal["extra_regions"]], table)
            row["current"] = first_interval(row.get("tracks", {}), RULE) or row.get("baseline_review_s")
    return rows


def first_interval(tracks: dict, rule: IntervalRule) -> list[float] | None:
    intervals = localize(tracks, rule)
    if not intervals:
        return None
    span = intervals[0].during
    return [span[0].time_s, span[-1].time_s]


def iou(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def predict(row: dict, mode: str, rule: IntervalRule, threshold: float | None) -> tuple[list[float] | None, bool]:
    if mode.startswith("dense"):
        proposed = first_interval(row.get("dense_tracks", {}), rule)
    elif mode == "sequence" and (windows := row["candidate"].get("sequence_windows")) and threshold is not None:
        first = windows[0]
        proposed = [first["start_s"], first["end_s"]] if first["response"] >= threshold else None
    else:
        proposed = None
    return proposed or row.get("current"), proposed is not None


def evaluate(rows: list[dict], mode: str, rule: IntervalRule = RULE, threshold: float | None = None) -> dict:
    by_case = []
    for row in rows:
        if row["status"] != "ready":
            continue
        predicted, used_new = predict(row, mode, rule, threshold)
        overlap = iou(predicted, row["edit_s"]) if predicted and row["label"] else 0.0
        by_case.append({"id": row["id"], "group": row["group"], "family": row["family"], "label": row["label"],
                        "edit_s": row["edit_s"], "first_review_s": predicted, "used_candidate": used_new,
                        "overlap": overlap, "localized": overlap >= .30})
    edits = [r for r in by_case if r["label"]]
    originals = [r for r in by_case if not r["label"]]
    hard = [r for r in originals if r["family"] != "original_videos"]
    return {"edits": len(edits), "originals": len(originals), "hard_controls": len(hard),
            "localized": sum(r["localized"] for r in edits), "mean_overlap": mean(r["overlap"] for r in edits),
            "original_flags": sum(bool(r["first_review_s"]) for r in originals),
            "hard_control_flags": sum(bool(r["first_review_s"]) for r in hard),
            "used_candidate": sum(r["used_candidate"] for r in by_case),
            "candidate_localized": sum(r["used_candidate"] and r["localized"] for r in edits), "cases": by_case}


def choose(rows: list[dict]) -> tuple[IntervalRule, float | None]:
    baseline = evaluate(rows, "current")

    def rank(result: dict) -> tuple:
        acceptable = (result["original_flags"] <= baseline["original_flags"] + 1
                      and result["hard_control_flags"] <= baseline["hard_control_flags"] + 1)
        return acceptable, result["localized"], -result["original_flags"], result["mean_overlap"]

    dense_rules = [RULE, *(IntervalRule(*values) for values in product((.60, .70, .75), (.15, .20, .25), (.35, .40, .45)))]
    dense = max(dense_rules, key=lambda rule: rank(evaluate(rows, "dense", rule)))
    values = {row["candidate"]["sequence_windows"][0]["response"] for row in rows if row["candidate"].get("sequence_windows")}
    thresholds = sorted((value for value in values if value > 0), reverse=True) or [None]
    sequence = max(thresholds, key=lambda threshold: rank(evaluate(rows, "sequence", threshold=threshold)))
    return dense, sequence


def summary(rows: list[dict], rule: IntervalRule, threshold: float | None) -> dict:
    out = {mode: evaluate(rows, mode, RULE if mode == "dense-fixed" else rule, threshold) for mode in MODES}
    current = out["current"]
    for mode in ("dense", "sequence"):
        comparison = out[mode]
        prior = {r["id"]: r for r in current["cases"]}
        changes = [(case["localized"], prior[case["id"]]["localized"]) for case in comparison["cases"] if case["label"]]
        comparison["newly_localized"] = sum(new and not old for new, old in changes)
        comparison["lost_localizations"] = sum(old and not new for new, old in changes)
        comparison["qualifies_for_adoption"] = (comparison["localized"] > current["localized"]
            and comparison["original_flags"] <= current["original_flags"] + 1
            and comparison["hard_control_flags"] <= current["hard_control_flags"] + 1)
    dense = [r["candidate"] for r in rows if r["candidate"].get("dense_points")]
    sequence = [r["candidate"] for r in rows if r["candidate"].get("sequence_windows")]
    out["coverage"] = {"dense_scored_clips": len(dense), "sequence_window_clips": len(sequence),
                       "median_added_frames": median([len(r["dense_points"]) for r in dense]) if dense else 0,
                       "median_dense_inference_s": median([r["dense_inference_s"] for r in dense]) if dense else None,
                       "median_sequence_inference_s": median([r["sequence_inference_s"] for r in sequence]) if sequence else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--test-output", type=Path)
    args = parser.parse_args()
    if args.test_output is None:
        rows = input_rows(args.manifest, args.baseline, args.candidate, "development")
        rule, threshold = choose(rows)
        result = {"manifest_sha256": sha(args.manifest), "baseline_sha256": sha(args.baseline),
                  "candidate_sha256": sha(args.candidate), "development_groups": sorted({r["group"] for r in rows}),
                  "dense_rule": asdict(rule), "sequence_threshold": threshold, "development": summary(rows, rule, threshold)}
        target = args.lock
        metrics = result["development"]
    else:
        lock = json.loads(args.lock.read_text())
        rows = input_rows(args.manifest, args.baseline, args.candidate, "test")
        if lock["manifest_sha256"] != sha(args.manifest) or set(lock["development_groups"]).intersection(r["group"] for r in rows):
            raise ValueError("The held-out split does not match the frozen experiment")
        rule, threshold = IntervalRule(**lock["dense_rule"]), lock["sequence_threshold"]
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("Invalid frozen threshold")
        metrics = summary(rows, rule, threshold)
        result = {"manifest_sha256": sha(args.manifest), "lock_sha256": sha(args.lock), "baseline_sha256": sha(args.baseline),
                  "candidate_sha256": sha(args.candidate), "test": metrics}
        target = args.test_output
    with target.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    for mode in MODES:
        print(mode, {k: v for k, v in metrics[mode].items() if k != "cases"})


if __name__ == "__main__":
    main()
