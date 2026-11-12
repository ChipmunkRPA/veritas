# SPDX-License-Identifier: Apache-2.0
"""Reopen the GRID reserve comparison and independently verify source-clock evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import review_video_reference_h264_earlier_transfer as replay
from scripts import video_reference_h264_grid_reserve as study
from scripts.video_common_sources_v89 import read, require, sha, write


def review(run, plan, initial):
    locked, runtime, aggregate = (read(run / filename) for filename in ("freeze.json", "runtime.json", "aggregate.json"))
    require(locked["spec"] == study.SPEC and locked["binding"] == study.binding(plan, initial)
            and aggregate["runtime"] == runtime and sha(run / "freeze.json") == runtime["freeze_sha256"]
            and sha(run / "held.jsonl") == runtime["held_sha256"], "Scope or held commitment differs")
    chosen = study.inputs(plan, initial)
    rows = study.study.previous.rows(run, "held")
    require([(row["ordinal"], row["condition"]) for row in rows]
            == [(item["ordinal"], role) for item in chosen for role in study.ROLES], "Complete verified roster required")
    indexed = {(row["ordinal"], row["condition"]): row for row in rows}
    require(locked["roster"] == [{"ordinal": row["ordinal"], "component": row["component_index"],
            "input_sha256": row["input"]["source_sha256"], "unrelated_ordinal": study.donor_for(row, chosen)["ordinal"]}
            for row in chosen], "Source grouping or unrelated roster differs")
    counts = {role: Counter() for role in study.ROLES}
    qualified_groups = set()
    crops = checked_spatial = 0
    for item in chosen:
        ordinal = item["ordinal"]
        source = study.pictures(item)
        ref = run / "deliveries" / f"{ordinal:02d}-earlier.mp4"
        earlier = study.study.table(ref)
        study.expected_clock(earlier)
        for role in study.ROLES:
            row = indexed[ordinal, role]
            path = run / "deliveries" / f"{ordinal:02d}-{role}.mp4"
            later = study.study.table(path)
            study.expected_clock(later)
            require(row["earlier_sha256"] == sha(ref) and row["submitted_sha256"] == sha(path)
                    and row["known_edit"] == (list(study.RANGE) if role.startswith("local_") else None)
                    and row["component"] == item["component_index"], "Held source or authored interval differs")
            outcome = replay.approved(row, study.CLOCK)
            counts[role].update({"views": 1, **outcome})
            if role == "local_tone" and outcome["joint_correct"]:
                qualified_groups.add(row["component"])
            if not outcome["flag"]:
                continue
            best = row["best"]
            first, peak, last = best["samples"]
            start, end = best["frames"]
            require(first + 1 == start <= peak < end == last, "Wrong before/during/after native pictures")
            x, y, w, h = best["tile_xywh"]
            for clip, clock in ((ref, earlier), (path, later)):
                for rgb in replay.read_pictures(clip, clock, best["samples"]).values():
                    require(rgb[y:y+h, x:x+w].shape == (h, w, 3), "Native crop lies outside its source")
                    crops += 1
            if row["known_edit"]:
                authored = study.author(source, role)
                mask = np.any(authored[peak] != source[peak], axis=2)
                require(row["spatial"] == [int(mask[y:y+h, x:x+w].sum()), int(mask.sum()), w*h],
                        "Actual authored changed-pixel counts differ")
                checked_spatial += 1
    require(counts == aggregate["counts"] and len(qualified_groups) == aggregate["useful_tone_groups"],
            "Published paired outcomes differ")
    observed = {key: value for key, value in aggregate["criteria"].items()}
    expected = {"tone_locations": counts["local_tone"]["joint_correct"] >= 14, "tone_groups": len(qualified_groups) >= 6,
                "coverage": all(counts[role]["assessed"] * 5 >= counts[role]["views"] * 4
                                for role in study.ROLES if role != "unrelated"),
                "neutral": sum(counts[role]["flag"] for role in study.ROLES[:3]) <= 1,
                "tandem": sum(counts[role]["flag"] for role in study.ROLES[3:5]) == 0,
                "unrelated": counts["unrelated"]["flag"] == 0}
    require(observed == expected and aggregate["advances"] == all(expected.values()), "Prospective gate differs")
    return {"status": "verified", "matched_clock_pairs": len(rows), "native_originals": len(chosen),
            "before_during_after_crops": crops, "authored_spatial_counts": checked_spatial,
            "aggregate_sha256": sha(run / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plan", type=Path, default=study.PLAN)
    parser.add_argument("--initial", type=Path, default=study.INITIAL)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = review(args.run, args.plan, args.initial)
    write(args.run / "independent-audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
