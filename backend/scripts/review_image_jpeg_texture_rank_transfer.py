# SPDX-License-Identifier: Apache-2.0
"""Rebuild matched JPEG trial inputs and independently audit its native geometry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import image_jpeg_texture_rank_transfer as trial
from scripts.review_image_residual_encoder_transfer import overlap


def geometry_oracle(truth):
    """Check whether any permitted rectangle could achieve the overlap criterion."""
    if not truth.any():
        return False
    grid, _ = trial.old.grid(truth, 0)
    rows, cols = grid.shape[:2]
    target = int(truth.sum())
    integral = np.pad(truth.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cell = trial.old.STRIDE
    for h in trial.old.SIDES:
        for w in trial.old.SIDES:
            area = h * w * cell**2
            if max(h, w) > 2 * min(h, w) or area > .10 * truth.size or h > rows or w > cols:
                continue
            x, y = grid[:rows-h+1, :cols-w+1].transpose(2, 0, 1) - cell // 2
            inside = integral[y + h * cell, x + w * cell] + integral[y, x] - integral[y + h * cell, x] - integral[y, x + w * cell]
            if np.any((2 * inside >= target) & (10 * inside >= 3 * (target + area - inside))):
                return True
    return False


def audit_inputs(inputs, rows, points=None):
    seen = {(row["source"], row["role"], row["quality"]): row for row in rows}
    trial.old.require(len(seen) == len(rows), "Repeated JPEG trial view")
    measures, reachable = 0, defaultdict(Counter)
    for item, native, exact in inputs:
        ordinary, edits = trial.old.authored(native, exact, item["recipient"])
        for role, rgb in (ordinary | edits).items():
            truth = np.any(rgb != native["original"], axis=2) if role in trial.old.POSITIVE else np.zeros(rgb.shape[:2], bool)
            if role in trial.old.POSITIVE:
                reachable[f"{item['generator']}/{role}"].update(views=1, oracle_useful=geometry_oracle(truth))
            for quality in trial.old.QUALITIES:
                row = seen.pop((item["recipient"], role, quality))
                arms = set(trial.ARMS if points is not None else trial.ARMS[:2])
                trial.old.require(row["input_sha256"] == trial.old.sha(trial.old.encode(rgb, quality)) and
                                  row["truth_sha256"] == trial.old.sha(truth.tobytes()) and set(row["result"]) == arms and
                                  row["generator"] == item["generator"] and 0 <= row["valid_cells"] <= row["cells"],
                                  "JPEG, exact native mask or arm scope differs")
                for arm, result in row["result"].items():
                    metric = overlap(truth, result["box"])
                    useful = metric["iou"] is not None and metric["iou"] >= .30 and metric["recall"] >= .50
                    trial.old.require(result["overlap"] == metric and result["useful"] == useful, "Native region overlap changed")
                    if points is not None:
                        point = points[str(quality)].get(arm)
                        flagged = bool(result["box"]) and (arm == "current_luma" or
                                                           (point is not None and result["score"] > point))
                        trial.old.require(result["flagged"] == flagged, "Recorded flag differs from fixed point")
                    measures += 1
    trial.old.require(not seen, "Unexpected leftover JPEG trial views")
    return measures, reachable


def original_ranking(rows):
    original_boxes = {(r["source"], r["quality"]): r["result"] for r in rows if r["role"] == "original"}
    summary = defaultdict(Counter)
    for row in rows:
        if row["role"] not in trial.old.POSITIVE:
            continue
        previous = original_boxes[row["source"], row["quality"]]
        for arm, result in row["result"].items():
            section = summary[f"{row['quality']}/{row['generator']}/{arm}"]
            section.update(edited_views=1, same_nonempty_box_as_original=bool(result["box"]) and result["box"] == previous[arm]["box"],
                           touched_truth=result["overlap"]["overlap"] > 0, useful_before_point=result["useful"])
    return summary


def review(args):
    plan, _ = trial.source_plan(args)
    folder = args.output
    result = trial.old.read(folder / "result.json")
    trial.old.require(plan == trial.old.read(folder / "plan.json") and result["design"] == plan["design"] == trial.DESIGN and
                      all(trial.roster.sha(folder / name) == digest for name, digest in result["commitments"].items()),
                      "Frozen result or input commitment changed")
    profile = trial.old.read(args.encoder / "freeze.json")
    args.rosters = trial.roster.DATA
    cal = trial.old.read(folder / "calibration.json")
    held = trial.old.read(folder / "held.json")
    points = trial.old.read(folder / "points.json")
    trial.old.require(points == result["points"] == trial.thresholds(cal), "Ordinary-only point does not reproduce")
    calibration_count, _ = audit_inputs(trial.old.frozen_inputs(args, "calibration", profile), cal)
    held_count, reachable = audit_inputs(trial.held_inputs(folder, plan), held, points)
    trial.old.require(trial.summarize(held) == {key: result[key] for key in ("by_quality", "strata")}, "Aggregate outcome changed")
    intake = trial.old.read(folder / "intake.json")
    admitted = {r["slot"] for r in intake["rows"] if r["status"] == "qualified"}
    result["intake"] = {"planned": Counter(r["generator"] for r in plan["cases"]),
                        "qualified_by_split": Counter(r["publisher_split"] for r in plan["cases"] if r["slot"] in admitted),
                        "excluded_reasons": Counter(r["reason"] for r in intake["rows"] if r["status"] == "excluded"),
                        "qualified_assets": sum(len(r["receipts"]) for r in intake["rows"] if r["slot"] in admitted)}
    result["review"] = {"input_and_truth_bindings": len(cal) + len(held), "native_box_overlaps": calibration_count + held_count,
                        "held_texture_cell_coverage": sum(r["valid_cells"] for r in held) / sum(r["cells"] for r in held),
                        "held_native_geometry_oracle_without_rank_or_texture_support_gate": reachable,
                        "held_ranking_vs_matched_original": original_ranking(held), "code_sha256": trial.roster.sha(Path(__file__))}
    trial.old.write(args.publication, result)
    print(result["intake"], result["review"]["input_and_truth_bindings"], result["review"]["native_box_overlaps"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "encoder", "prior-f5", "output", "publication"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    review(parser.parse_args())


if __name__ == "__main__":
    main()
