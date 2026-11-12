# SPDX-License-Identifier: Apache-2.0
"""Build a compact public readout from the frozen whole-image evaluation journals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.core.canonical import canonical_json
from scripts import review_image_whole_fresh as review


def journal(directory, arm):
    status = review.read(directory / f"{arm}-execution.json")
    review.require(status["exit_code"] == 0 and status["guard"] is None, f"{arm} did not complete")
    rows = [json.loads(line) for line in (directory / f"{arm}.jsonl").read_text().splitlines()]
    keyed = {row["id"]: row for row in rows}
    review.require(len(rows) == len(keyed) == 68, f"{arm} does not cover all frozen deliveries")
    return keyed, status


def score(directory, source_path):
    source = review.verify_sources(source_path, directory)
    candidate, candidate_execution = journal(directory, "candidate")
    baseline, baseline_execution = journal(directory, "baseline")
    complete = review.read(directory / "candidate-complete.json")
    lock = review.read(directory / "baseline-lock.json")
    freeze_hash = source["frozen_input_record"]["sha256"]
    review.require(complete["input_freeze_sha256"] == lock["input_freeze_sha256"] == freeze_hash
                   and complete["candidate_journal_sha256"] == review.digest((directory / "candidate.jsonl").read_bytes()),
                   "Runtime journals are not bound to the published freeze")
    result_rows = []
    checks = Counter()
    by_case = {}
    for item in source["frozen_record"]["items"]:
        case = f"{item['source_index']:03d}-{item['format']}"
        novel, current = candidate[case], baseline[case]
        review.require(novel["status"] == "ok" and current["status"] == "done"
                       and novel["input_sha256"] == current["input_sha256"] == item["sha256"], "Case status or bytes differ")
        blob = (directory / "baseline-results" / f"{case}.json").read_bytes()
        packet = json.loads(blob)
        review.require(review.digest(blob) == current["retained_packet_sha256"]
                       and review.digest(canonical_json(packet["result"])) == current["canonical_result_sha256"],
                       "Retained Full result commitment changed")
        checks.update(row["status"] for row in current["checks"])
        direction = current["interpretation"]["customer_direction"]
        row = {"id": case, "input_sha256": item["sha256"], "crop_logits": novel["crop_logits"], "mean_logit": novel["mean_logit"],
               "candidate_flag": novel["flagged"], "baseline_direction": direction}
        result_rows.append(row)
        by_case[case] = {**row, "format": item["format"], "label": item["label"], "family": item["family"]}

    counts, added_families = {}, {}
    criteria = source["frozen_record"]["criteria"]
    passed = True
    for format_ in ("native", "jpeg75"):
        counts[format_] = {}
        for label in ("ordinary", "synthetic"):
            selected = [row for row in by_case.values() if row["format"] == format_ and row["label"] == label]
            added = [row for row in selected if row["candidate_flag"] and row["baseline_direction"] != "ai"]
            directions = Counter(row["baseline_direction"] for row in selected)
            flags = sum(row["candidate_flag"] for row in selected)
            counts[format_][label] = {"candidate_flags": flags, "candidate_misses": len(selected) - flags,
                                      "added_flags": len(added), "combined_flags": len(added) + directions["ai"],
                                      "added_original_conflicts": sum(row["baseline_direction"] == "original" for row in added),
                                      **{f"baseline_{name}": directions[name] for name in ("ai", "unavailable", "original", "inconclusive", "record")}}
            if label == "synthetic":
                families = sorted({row["family"] for row in added})
                added_families[format_] = families
                passed &= (len(added) >= criteria["format_minimum_added_synthetic"]
                           and len(families) >= criteria["format_minimum_represented_generator_families"])
            else:
                passed &= len(added) <= criteria["format_maximum_additional_ordinary_groups"]
    context = lock["baseline"]
    summary = {"candidate_complete": True, "baseline_complete": True, "counts": counts,
               "added_generator_families": added_families, "gate_passed": bool(passed)}
    return {"schema": "chipmunk.image-whole-fresh-comparison/1", "source_manifest_sha256": review.digest(source_path.read_bytes()),
            "candidate_execution": candidate_execution, "baseline_execution": baseline_execution,
            "baseline_commit": lock["source_sha256"]["commit"], "baseline_enabled_checks": len(context["detectors"]),
            "baseline_check_statuses": checks, "baseline_fusion_mode": context["fusion_selection"]["mode"],
            "baseline_settings_sha256": context["config_sha256"], "baseline_policy_sha256": context["policy"]["sha256"],
            "summary": summary, "cases": result_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=review.SOURCE)
    args = parser.parse_args()
    result = score(args.directory, args.source)
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        output.write("\n")
