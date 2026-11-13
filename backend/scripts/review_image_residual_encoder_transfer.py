# SPDX-License-Identifier: Apache-2.0
"""Read back the paired residual-encoder trial without fitting or rescoring it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import image_residual_encoder_transfer as trial


def overlap(mask, box):
    height, width = mask.shape
    predicted = intersection = 0
    target = int(mask.sum())
    if box is not None:
        x, y, r, b = box
        trial.require(0 <= x < r <= width and 0 <= y < b <= height, "Reported region falls outside source")
        predicted = (r - x) * (b - y)
        intersection = int(np.count_nonzero(mask[y:b, x:r]))
    union = predicted + target - intersection
    return {"reported": predicted, "target": target, "overlap": intersection,
            "iou": intersection / union if target else None, "recall": intersection / target if target else None}


def audit_rows(args, phase, profile, journal, points):
    observed = {(row["source"], row["role"], row["quality"]): row for row in journal}
    trial.require(len(observed) == len(journal), "Repeated assessment")
    measurements, maxima, ranking = 0, defaultdict(dict), defaultdict(Counter)
    for info, views, exact in trial.frozen_inputs(args, phase, profile):
        ordinary, edits = trial.authored(views, exact, info["recipient"])
        for role, rgb in (ordinary | edits).items():
            target = np.any(rgb != views["original"], axis=2) if role in trial.POSITIVE else np.zeros(rgb.shape[:2], bool)
            for quality in trial.QUALITIES:
                row = observed.pop((info["recipient"], role, quality))
                encoded = trial.encode(rgb, quality)
                trial.require(row["input_sha256"] == trial.sha(encoded) and row["shape"] == list(target.shape)
                              and row["truth_sha256"] == trial.sha(target.tobytes()) and row["generator"] == info["generator"],
                              "Source body, grouping, or exact mask changed")
                candidate = row["candidate"]
                point = points[str(quality)]
                group = info["recipient"]
                for method in ("candidate", "reference") if phase == "held" else ("candidate",):
                    result = row[method]
                    metric = overlap(target, result["box"])
                    useful = metric["iou"] is not None and metric["iou"] >= .30 and metric["recall"] >= .50
                    trial.require(result["overlap"] == metric and result["useful"] == useful, "Native-pixel overlap changed")
                    if phase == "held":
                        expected = result["score"] > point if method == "candidate" else result["box"] is not None
                        trial.require(result["flagged"] == expected, "Selected result differs from its fixed operating point")
                        if method == "candidate" and role in trial.POSITIVE:
                            key = f"{quality}/{info['generator']}/{role}"
                            ranking[key].update(views=1, useful_before_point=useful, useful_after_point=useful and expected)
                    measurements += 1
                if phase == "calibration":
                    maxima[str(quality)][group] = max(candidate["score"], maxima[str(quality)].get(group, -float("inf")))
    trial.require(not observed, "Unexpected records in measurement journal")
    return measurements, maxima, ranking


def review(args):
    path = args.output
    summary = trial.read(path / "aggregate.json")
    profile = trial.read(path / "freeze.json")
    trial.require(trial.sha((path / "freeze.json").read_bytes()) == summary["freeze_sha256"] and
                  profile["design"] == summary["design"] == trial.DESIGN and
                  all(trial.sha((path / name).read_bytes()) == digest for name, digest in summary["artifact_sha256"].items()) and
                  trial.sha((path / "fit-inputs.json").read_bytes()) == summary["fit"]["fit_inputs_sha256"] and
                  trial.sha((path / "encoder.pt").read_bytes()) == summary["fit"]["checkpoint_sha256"], "Trial commitment changed")
    calibration = trial.read(path / "calibration.json")
    held = trial.read(path / "held.json")
    measurements, maxima, _ = audit_rows(args, "calibration", profile, calibration, summary["points"])
    receipt = trial.read(path / "points.json")
    trial.require(dict(maxima) == receipt["maxima"] and receipt["points"] == summary["points"] and
                  all(len(groups) == 12 and sorted(groups.values())[-2] == receipt["points"][quality]
                      for quality, groups in maxima.items()), "Ordinary-only calibration differs")
    held_measurements, _, ranking = audit_rows(args, "held", profile, held, summary["points"])
    replay = trial.summarize(held)
    replay["qualities"] = {str(quality): data for quality, data in replay["qualities"].items()}
    trial.require(replay == {key: summary[key] for key in ("qualities", "strata")}, "Aggregate outcomes changed")
    summary["review"] = {"replayed_input_and_truth_bindings": len(calibration) + len(held),
                         "replayed_pixel_overlaps": measurements + held_measurements,
                         "held_candidate_ranking": ranking,
                         "code_sha256": trial.sha(Path(__file__).read_bytes())}
    trial.write(args.publication, summary)
    print(summary["review"]["replayed_input_and_truth_bindings"], summary["review"]["replayed_pixel_overlaps"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--rosters", default=Path(__file__).resolve().parents[1] / "data" / "independent-evaluation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    review(parser.parse_args())


if __name__ == "__main__":
    main()
