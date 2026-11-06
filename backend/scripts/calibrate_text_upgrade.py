"""Compare matched document detectors and fit monotone reference calibration on article fit groups."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from prepare_text_upgrade import digest, file_hash, read_jsonl
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEED = 84673
BOOTSTRAPS = 2000
CALIBRATION_BOOTSTRAPS = 200


def read_predictions(path: Path, manifest_sha: str, cases: dict[str, dict]) -> tuple[str, dict[str, float | None]]:
    rows = read_jsonl(path)
    metadata = next(rows)
    if metadata["manifest_sha256"] != manifest_sha or metadata["count"] != len(cases):
        raise ValueError(f"{path.name}: unexpected manifest or incomplete run")
    scores = {}
    for row in rows:
        key = row["id"]
        raw = row["raw"]
        if key in scores or row["text_sha256"] != cases[key]["text_sha256"]:
            raise ValueError(f"{path.name}: inconsistent item {key}")
        if raw is not None and (type(raw) not in {float, int} or not np.isfinite(raw)):
            raise ValueError(f"{path.name}: invalid model response")
        scores[key] = raw
    if scores.keys() != cases.keys():
        raise ValueError(f"{path.name}: missing predictions")
    return metadata["model"], scores


def features(raw: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    values = raw.copy()
    for i, name in enumerate(names):
        if name == "binoculars":
            values[:, i] *= -1
        else:
            if np.any((values[:, i] < 0) | (values[:, i] > 1)):
                raise ValueError(f"{name}: expected bounded raw response")
            values[:, i] = logit(np.clip(values[:, i], 1e-6, 1 - 1e-6))
    return values


def fit_platt(x: np.ndarray, y: np.ndarray) -> dict:
    # Monotone for each signal. Fixed ridge penalty on standardized slopes
    # keeps tiny samples and near-separation from yielding extreme fits.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    if (std <= 0).any() or len(set(y)) != 2:
        raise ValueError("Reference fit needs variation and both classes")
    z = np.column_stack((np.ones(len(x)), (x - mean) / std))

    def loss(params):
        logits = z @ params
        residual = expit(logits) - y
        objective = np.mean(np.logaddexp(0, logits) - y * logits) + 0.005 * np.square(params[1:]).sum()
        gradient = (residual @ z) / len(y)
        gradient[1:] += 0.01 * params[1:]
        return objective, gradient

    result = minimize(loss, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
                      bounds=[(None, None), *((0, None) for _ in std)])
    if not result.success:
        raise ValueError(f"Calibration failed: {result.message}")
    slopes = result.x[1:] / std
    intercept = float(result.x[0] - mean @ slopes)
    return {"intercept": intercept, "slopes": slopes.tolist(), "reference_ai_prior": float(y.mean())}


def quality(y: np.ndarray, p: np.ndarray) -> dict:
    ai = p >= 0.8
    original = p <= 0.2
    return {
        "count": len(y), "human": int((y == 0).sum()), "ai": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "brier": float(np.square(p - y).mean()),
        "ai_correct": int((ai & (y == 1)).sum()), "ai_false": int((ai & (y == 0)).sum()),
        "original_correct": int((original & (y == 0)).sum()),
        "original_false": int((original & (y == 1)).sum()),
        "inconclusive": int((~(ai | original)).sum()),
    }


def paired_delta(groups: np.ndarray, y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict:
    group_deltas = defaultdict(list)
    for group, error in zip(groups, (candidate - y)**2 - (baseline - y)**2, strict=True):
        group_deltas[group].append(error)
    deltas = np.array([np.mean(group_deltas[group]) for group in sorted(group_deltas)])
    samples = np.random.default_rng(SEED).choice(deltas, size=(BOOTSTRAPS, len(deltas)))
    return {"delta_brier": float(deltas.mean()),
            "bootstrap_ci95": np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist(),
            "groups": len(deltas)}


def calibration_resamples(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[dict]:
    keys = sorted(set(groups))
    members = [np.flatnonzero(groups == key) for key in keys]
    rng = np.random.default_rng(SEED)
    models = []
    for draw in rng.integers(0, len(keys), size=(CALIBRATION_BOOTSTRAPS, len(keys))):
        indices = np.concatenate([members[i] for i in draw])
        if len(set(y[indices])) == 2:
            models.append(fit_platt(x[indices], y[indices]))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--intervals", action="store_true", help="Store fit-group bootstrap parameters for each comparison")
    args = parser.parse_args()
    manifest_sha = file_hash(args.manifest)
    rows = json.loads(args.manifest.read_text())["cases"]
    cases = {case["id"]: case for case in rows}
    if len(cases) != len(rows) or any(digest(case["text"]) != case["text_sha256"] for case in rows):
        raise ValueError("Manifest does not match its content commitments")
    fit_groups = {case["group"] for case in rows if case["split"] == "fit"}
    if any(case["group"] in fit_groups for case in rows if case["split"] != "fit"):
        raise ValueError("Reference fit and evaluation groups overlap")
    predictions = dict(read_predictions(path, manifest_sha, cases) for path in args.predictions)
    if "desklib" not in predictions or len(predictions) != len(args.predictions):
        raise ValueError("One baseline and unique candidate model inputs are required")
    # Pair the comparison on cases all provided models can score. Report
    # differences in score availability rather than treating them as negatives.
    available = [case for case in rows if all(scores[case["id"]] is not None for scores in predictions.values())]
    domains = np.array([case["domain"] for case in available])
    labels = np.array([int(case["label"] == "ai") for case in available])
    groups = np.array([case["group"] for case in available])
    generators = np.array([case["generator"] for case in available])
    fit = np.array([case["split"] == "fit" for case in available])
    article = (domains == "article") & ~fit
    email = domains == "email"
    mixed = domains == "mixed"
    methods = [(name,) for name in predictions]
    if "binoculars" in predictions:
        methods += [(name, "binoculars") for name in predictions if name != "binoculars"]
    report = {
        "manifest_sha256": manifest_sha, "prediction_sha256": {p.name: file_hash(p) for p in args.predictions},
        "paired_cases": len(available),
        "unavailable": {name: sum(value is None for value in values.values()) for name, values in predictions.items()},
        "models": {},
    }
    lengths = np.log(np.array([len(case["text"].split()) for case in available], dtype=float))
    fit_lengths = lengths[fit]
    fit_labels = labels[fit]
    length_direction = 1 if fit_lengths[fit_labels == 1].mean() >= fit_lengths[fit_labels == 0].mean() else -1
    length_features = (length_direction * lengths).reshape(-1, 1)
    length_fit = fit_platt(length_features[fit], fit_labels)
    length_p = expit(length_fit["intercept"] + length_features @ length_fit["slopes"])
    report["length_only_control"] = {
        "article_holdout": quality(labels[article], length_p[article]),
        "email_transfer": quality(labels[email], length_p[email]),
    }
    calibrated = {}
    for names in methods:
        name = "+".join(names)
        raw = np.array([[predictions[m][case["id"]] for m in names] for case in available])
        x = features(raw, names)
        calibration = fit_platt(x[fit], labels[fit])
        p = expit(calibration["intercept"] + x @ calibration["slopes"])
        calibrated[name] = p
        summary = {"features": names, "calibration": calibration, "article_fit_count": int(fit.sum()),
                   "article_holdout": quality(labels[article], p[article]),
                   "email_transfer": quality(labels[email], p[email])}
        fake_articles = article & (labels == 1)
        summary["article_generators"] = {}
        for generator in sorted(set(generators[fake_articles])):
            mask = fake_articles & (generators == generator)
            summary["article_generators"][generator] = quality(labels[mask], p[mask])
        if args.intervals:
            summary["calibration_bootstrap"] = calibration_resamples(x[fit], labels[fit], groups[fit])
        fractions = np.array([case["ai_character_fraction"] for case in available if case["domain"] == "mixed"])
        rho = spearmanr(fractions, p[mixed]).statistic
        summary["mixed_challenge"] = {
            "count": len(fractions), "rank_correlation_with_ai_character_fraction": float(rho) if np.isfinite(rho) else None,
            "above_ai_cutoff": int((p[mixed] >= 0.8).sum()), "below_original_cutoff": int((p[mixed] <= 0.2).sum()),
            "midrange": int(((p[mixed] > 0.2) & (p[mixed] < 0.8)).sum()),
        }
        report["models"][name] = summary
    for name, p in calibrated.items():
        if name != "desklib":
            for subset, mask in (("article_holdout", article), ("email_transfer", email)):
                report["models"][name][f"{subset}_paired_vs_desklib"] = paired_delta(
                    groups[mask], labels[mask], calibrated["desklib"][mask], p[mask]
                )
        if "+" in name:
            report["models"][name]["article_holdout_paired_vs_primary_only"] = paired_delta(
                groups[article], labels[article], calibrated[name.split("+")[0]][article], p[article]
            )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    for name, result in report["models"].items():
        print(name, "articles", result["article_holdout"], "email", result["email_transfer"])


if __name__ == "__main__":
    main()
