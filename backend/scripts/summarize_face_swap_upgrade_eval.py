"""Compare calibrated face-check responses on the paired RTFS-10K holdout."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_video_upgrade_eval import sha
from run_face_swap_upgrade_eval import CHECKS
from scipy.special import logit
from summarize_video_upgrade_eval import (
    calibrated,
    direction,
    fit,
    load_run,
    metrics,
    paired_delta,
    tendencies,
)

TEMPORAL = "candidate-videomae"


def summarize(manifest_path: Path, run_path: Path, temporal_path: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    with run_path.open() as file:
        run = json.loads(next(file))["run"]
        rows = [json.loads(line) for line in file]
    if sha(manifest_path) != run["manifest_sha256"]:
        raise ValueError("run does not match the frozen source manifest")
    if [r["id"] for r in rows] != [r["id"] for r in manifest["items"]]:
        raise ValueError("incomplete or disordered run")
    for row, item in zip(rows, manifest["items"], strict=True):
        if any(row[k] != item[k] for k in ("group", "label", "split", "family")):
            raise ValueError(f"source identity mismatch: {row['id']}")
        if item["status"] == "ready" and row["sha256"] != item["matched"]["sha256"]:
            raise ValueError(f"source bytes mismatch: {row['id']}")
    result = {
        "manifest_sha256": sha(manifest_path),
        "run_sha256": sha(run_path),
        **compare(rows, CHECKS),
    }
    if temporal_path is not None:
        header, temporal = load_run(temporal_path)
        if header["system"] != TEMPORAL or header["manifest_sha256"] != sha(manifest_path):
            raise ValueError("temporal run does not match the source manifest")
        if set(temporal) != {(row["id"], "matched") for row in rows}:
            raise ValueError("incomplete temporal run")
        combined = []
        for row in rows:
            candidate = temporal[row["id"], "matched"]
            if any(candidate[k] != row.get(k) for k in ("sha256", "group", "split", "label", "family")):
                raise ValueError(f"temporal run input mismatch: {row['id']}")
            combined.append({
                **row,
                "scores": {**row.get("scores", {}), TEMPORAL: {
                    "prob": candidate.get("score"), "status": candidate["status"],
                }},
            })
        result["temporal_comparison"] = {
            "run_sha256": sha(temporal_path),
            **compare(combined, (CHECKS[0], TEMPORAL)),
        }
    return result


def compare(rows: list[dict], checks: tuple[str, str]) -> dict:
    def probability(row: dict, check: str) -> float | None:
        return row.get("scores", {}).get(check, {}).get("prob")

    def logits(scores):
        return logit(np.clip(scores, 1e-7, 1 - 1e-7))

    paired = [row for row in rows if all(probability(row, check) is not None for check in checks)]
    calibration = [row for row in paired if row["split"] == "calibration"]
    test = [row for row in paired if row["split"] == "test"]
    all_test = [row for row in rows if row["split"] == "test"]
    y = np.asarray([row["label"] for row in test])
    if set(y) != {0, 1}:
        raise ValueError(f"paired face holdout lacks one class: {checks}")
    labels = [row["label"] for row in all_test]
    groups = np.asarray([row["group"] for row in test])
    results, estimates = {}, {}
    for check in checks:
        coef = fit(logits([probability(row, check) for row in calibration]), [row["label"] for row in calibration])
        scores = np.asarray([probability(row, check) for row in test])
        estimate = calibrated(logits(scores), coef)
        estimates[check] = estimate
        directions = []
        for row in all_test:
            score = probability(row, check)
            if score is None:
                directions.append(row.get("scores", {}).get(check, {}).get("status", row["status"]))
            else:
                directions.append(direction(float(calibrated(logits(score), coef))))
        families = np.asarray([row["family"] for row in test])
        results[check] = {
            "calibration_slope_intercept": coef,
            "paired_raw": metrics(y, scores),
            "paired_calibrated": metrics(y, estimate),
            "paired_per_generator": {
                family: metrics(y[(families == family) | (y == 0)], estimate[(families == family) | (y == 0)])
                for family in sorted(set(families[y == 1]))
            },
            "all_source_80_tendencies": tendencies(labels, directions),
            "coverage_by_family": {
                family: dict(Counter(d for d, row in zip(directions, all_test, strict=True) if row["family"] == family))
                for family in sorted({row["family"] for row in all_test})
            },
        }
    return {
        "all_test_sources": len(all_test),
        "paired_calibration_sources": len(calibration),
        "paired_test_sources": len(test),
        "paired_test_groups": len(set(groups)),
        "systems": results,
        "delta_candidate_minus_current_brier": paired_delta(
            groups, (y - estimates[checks[1]]) ** 2 - (y - estimates[checks[0]]) ** 2
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "scores", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--temporal", type=Path)
    args = parser.parse_args()
    result = summarize(args.manifest, args.scores, args.temporal)
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
