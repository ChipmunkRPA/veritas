"""Compare email-specific calibration on held-out authors and the independent business-email source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from calibrate_text_upgrade import (
    calibration_resamples,
    features,
    fit_platt,
    paired_delta,
    quality,
    read_predictions,
)
from prepare_text_upgrade import digest, file_hash
from scipy.special import expit


def read_cases(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    cases = {row["id"]: row for row in rows}
    if len(cases) != len(rows) or any(digest(row["text"]) != row["text_sha256"] for row in rows):
        raise ValueError("Invalid or repeated source text in manifest")
    return cases


def feature_matrix(predictions: dict, names: tuple[str, ...], rows: list[dict]) -> np.ndarray:
    raw = [[predictions[name][row["id"]] for name in names] for row in rows]
    if any(value is None for scores in raw for value in scores):
        raise ValueError(f"Some email responses are unavailable for {names}")
    return features(np.asarray(raw, dtype=float), names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("transfer_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--transfer", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    reference_cases = read_cases(args.reference_manifest)
    transfer_cases = read_cases(args.transfer_manifest)
    reference_sha, transfer_sha = file_hash(args.reference_manifest), file_hash(args.transfer_manifest)
    reference_predictions = dict(read_predictions(p, reference_sha, reference_cases) for p in args.reference)
    transfer_predictions = dict(read_predictions(p, transfer_sha, transfer_cases) for p in args.transfer)
    if reference_predictions.keys() != transfer_predictions.keys() or "desklib" not in reference_predictions:
        raise ValueError("Use the same distinct models on both sources, including the incumbent")
    if len(reference_predictions) != len(args.reference) or len(transfer_predictions) != len(args.transfer):
        raise ValueError("A model was supplied more than once")
    reference = list(reference_cases.values())
    transfer = [row for row in transfer_cases.values() if row["domain"] == "email"]
    fit = np.array([row["split"] == "fit" for row in reference])
    authors = np.array([row["author_group"] for row in reference])
    if set(authors[fit]) & set(authors[~fit]):
        raise ValueError("An author crosses reference fit and holdout")
    if {row["text_sha256"] for row in reference} & {row["text_sha256"] for row in transfer}:
        raise ValueError("Reference and transfer text overlaps")
    y_ref = np.array([int(row["label"] == "ai") for row in reference])
    y_transfer = np.array([int(row["label"] == "ai") for row in transfer])
    transfer_groups = np.array([row["group"] for row in transfer])
    report = {"reference_manifest_sha256": reference_sha, "transfer_manifest_sha256": transfer_sha, "models": {}}
    probabilities = {}
    methods = [(name,) for name in reference_predictions]
    if {"tropa", "modernbert"} <= reference_predictions.keys():
        methods.append(("tropa", "modernbert"))
    for names in methods:
        name = "+".join(names)
        x_ref = feature_matrix(reference_predictions, names, reference)
        x_transfer = feature_matrix(transfer_predictions, names, transfer)
        fit_model = fit_platt(x_ref[fit], y_ref[fit])
        p_ref = expit(fit_model["intercept"] + x_ref @ fit_model["slopes"])
        p_transfer = expit(fit_model["intercept"] + x_transfer @ fit_model["slopes"])
        summary = {"features": names, "calibration": fit_model, "reference_fit_authors": len(set(authors[fit])),
                   "reference_holdout_authors": len(set(authors[~fit])),
                   "reference_holdout": quality(y_ref[~fit], p_ref[~fit]), "transfer": quality(y_transfer, p_transfer),
                   "calibration_bootstrap": calibration_resamples(x_ref[fit], y_ref[fit], authors[fit])}
        generators = np.array([row["generator"] for row in reference])
        summary["held_out_generators"] = {
            generator: quality(y_ref[mask], p_ref[mask])
            for generator in sorted(set(generators) - {"human"})
            if (mask := ~fit & (generators == generator)).any()
        }
        probabilities[name] = (p_ref[~fit], p_transfer)
        report["models"][name] = summary
        print(name, "reference", summary["reference_holdout"], "transfer", summary["transfer"])
    for name, (p_ref, p_transfer) in probabilities.items():
        if name == "desklib":
            continue
        old_ref, old_transfer = probabilities["desklib"]
        report["models"][name].update(
            reference_paired_vs_desklib=paired_delta(authors[~fit], y_ref[~fit], old_ref, p_ref),
            transfer_paired_vs_desklib=paired_delta(transfer_groups, y_transfer, old_transfer, p_transfer),
        )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
