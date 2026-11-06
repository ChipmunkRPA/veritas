"""Freeze interval thresholds on development groups and replay on the held-out groups."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean

from prepare_video_upgrade_eval import sha

from app.detectors.video.decode import FrameTable
from app.evidence.face_intervals import IntervalRule, bind_tracks, localize


def load(manifest: Path, scores: Path, split: str) -> list[dict]:
    expected = [row for row in json.loads(manifest.read_text())["items"] if row["split"] == split]
    with scores.open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    if header["manifest_sha256"] != sha(manifest) or header["split"] != split or len(rows) != len(expected):
        raise ValueError("incomplete or mismatched inference receipt")
    for row, source in zip(rows, expected, strict=True):
        if any(row.get(key) != value for key, value in source.items()):
            raise ValueError("score row differs from frozen test input")
        points = row.get("points") or []
        if points:
            table = FrameTable.from_json(row["table"])
            points = sorted(points, key=lambda p: p["frame_index"])
            peak = max(range(len(points)), key=lambda i: points[i]["score"])
            if points[peak]["score"] >= .5:
                row["peak_s"] = table.seconds(points[peak]["frame_index"])
                row["baseline_review_s"] = [table.seconds(points[i]["frame_index"])
                                             for i in (max(0, peak - 1), min(len(points) - 1, peak + 1))]
            row["tracks"] = bind_tracks(points, row["exhibits"], table)
    return rows


def evaluate(rows: list[dict], rule: IntervalRule) -> dict:
    originals = edits = baseline_false = baseline_hit = candidate_false = candidate_hit = candidate_peak = 0
    baseline_iou, intervals_iou = [], []
    cases = []
    for row in rows:
        if row["status"] != "ready":
            continue
        base = row.get("peak_s")
        ranges = [(i.during[0].time_s, i.during[-1].time_s, i.peak.time_s) for i in localize(row.get("tracks", {}), rule)]
        case = {"id": row["id"], "group": row["group"], "family": row["family"], "edit_s": row["edit_s"],
                "baseline_peak_s": base, "baseline_review_s": row.get("baseline_review_s"), "candidate_ranges_and_peaks_s": ranges,
                "scored": bool(row.get("points")), "track_bound_samples": sum(len(t) for t in row.get("tracks", {}).values())}
        if row["label"]:
            edits += 1
            a, b = row["edit_s"]
            baseline_hit += base is not None and a <= base < b
            if context := row.get("baseline_review_s"):
                x, y = context
                baseline_iou.append(max(0, min(b, y) - max(a, x)) / (max(b, y) - min(a, x)))
            else:
                baseline_iou.append(0.0)
            # Use the highest-priority interval, as shown first in the ordinary result.
            if ranges:
                x, y, peak = ranges[0]
                iou = max(0, min(b, y) - max(a, x)) / (max(b, y) - min(a, x))
                candidate_peak += a <= peak < b
            else:
                iou = 0.0
            candidate_hit += iou >= .30
            intervals_iou.append(iou)
            case["first_interval_iou"] = iou
        else:
            originals += 1
            baseline_false += base is not None
            candidate_false += bool(ranges)
        cases.append(case)
    return {"originals": originals, "edits": edits, "excluded": len(rows) - originals - edits,
            "baseline_false_originals": baseline_false, "baseline_peak_in_edit": baseline_hit,
            "baseline_iou_at_least_030": sum(iou >= .30 for iou in baseline_iou), "baseline_mean_iou": mean(baseline_iou),
            "interval_false_originals": candidate_false, "interval_peak_in_edit": candidate_peak,
            "interval_iou_at_least_030": candidate_hit, "interval_mean_iou": mean(intervals_iou), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "development", "lock"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    development = load(args.manifest, args.development, "development")
    if args.test is None:
        grid = [IntervalRule(*values) for values in product((.50, .60, .70), (.15, .20, .25), (.40, .45, .50))]
        trials = [(rule, evaluate(development, rule)) for rule in grid]
        rule, result = max(trials, key=lambda trial: (trial[1]["interval_false_originals"] <= min(2, trial[1]["baseline_false_originals"]),
                           trial[1]["interval_iou_at_least_030"], -trial[1]["interval_false_originals"], trial[1]["interval_mean_iou"],
                           trial[0].minimum_lift, trial[0].minimum_sample_response, trial[0].minimum_response))
        with args.lock.open("x") as out:
            json.dump({"manifest_sha256": sha(args.manifest), "development_sha256": sha(args.development),
                       "rule": asdict(rule), "development": result}, out, indent=2)
        print(asdict(rule), {key: value for key, value in result.items() if key != "cases"})
    else:
        locked = json.loads(args.lock.read_text())
        if locked["manifest_sha256"] != sha(args.manifest) or locked["development_sha256"] != sha(args.development):
            raise ValueError("frozen rule does not match the reference data")
        test = load(args.manifest, args.test, "test")
        if {row["group"] for row in test}.intersection(row["group"] for row in development):
            raise ValueError("development and held-out source groups overlap")
        result = evaluate(test, IntervalRule(**locked["rule"]))
        with args.output.open("x") as out:
            json.dump({**locked, "rule_sha256": sha(args.lock), "test_sha256": sha(args.test), "test": result}, out, indent=2)
        print({key: value for key, value in result.items() if key != "cases"})


if __name__ == "__main__":
    main()
