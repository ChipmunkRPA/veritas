"""Fit and compare fixed speech heads on source-paired noisy feature packets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from scripts.audio_score_fusion_v83 import weights

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
TRIAL = ROOT / "backend/data/independent-evaluation/audio-paired-consistency-v223"
PUBLIC = PANEL / "audio-paired-consistency-freeze-v223.json"
CONDITIONS = ("clean", "noise_0", "noise_5")
ROLES = ("fusion_fit", "confidence_cal", "selection")
DIRECTIONS = ("ai", "original", "inconclusive", "unavailable")
WAVLM_WIDTH = 4608
FITS = {"raw_only": 0.0, "wavlm_raw": 0.0, "paired_wavlm_raw": 0.15}
BASELINES = ("current", "v13_complement", "v83_global")
CODE = (
    "backend/scripts/audio_paired_consistency_v223.py",
    "backend/scripts/audio_paired_input_v223.py",
    "backend/scripts/audio_score_fusion_v83.py",
    "backend/tests/test_audio_paired_input_v223.py",
    "backend/tests/test_audio_paired_consistency_v223.py",
    "docs/audio-paired-consistency-v223-method.md",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assessed(rows):
    require(
        all(row["status"] in ("ok", "not_applicable") for row in rows),
        "Every row must have a verified status",
    )
    return [row for row in rows if row["status"] == "ok"]


def bounded_logit(score):
    values = np.asarray(score, dtype=np.float64)
    require(
        np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Finite score in [0,1] required"
    )
    return np.clip(logit(np.clip(values, 1e-6, 1 - 1e-6)), -8, 8)


def matrix(rows, arm, transform):
    raw = bounded_logit([row["raw"] for row in rows])
    columns = [raw]
    if arm != "raw_only":
        features = np.asarray([row["wavlm"] for row in rows], dtype=np.float64)
        require(
            features.shape == (len(rows), WAVLM_WIDTH) and np.isfinite(features).all(),
            "Exact feature vectors required",
        )
        features = (features - transform["center"]) / transform["scale"] * np.sqrt(16 / WAVLM_WIDTH)
        columns.append(features)
    return np.column_stack([*columns, np.ones(len(rows))])


def fit_transform(rows, row_weights):
    features = np.asarray([row["wavlm"] for row in rows], dtype=np.float64)
    require(
        features.shape == (len(rows), WAVLM_WIDTH) and np.isfinite(features).all(),
        "Exact fitting vectors required",
    )
    center = row_weights @ features
    scale = np.maximum(np.sqrt(row_weights @ (features - center) ** 2), 1e-6)
    return {"center": center.tolist(), "scale": scale.tolist()}


def pair_deltas(rows, design):
    clean = {row["base_id"]: i for i, row in enumerate(rows) if row["condition"] == "clean"}
    noisy_indices = [
        i for i, row in enumerate(rows) if row["condition"] != "clean" and row["base_id"] in clean
    ]
    require(bool(noisy_indices), "Paired fit requires noisy variants with assessed clean references")
    clean_indices = [clean[rows[i]["base_id"]] for i in noisy_indices]
    require(
        all(
            all(rows[i][key] == rows[j][key] for key in ("base_id", "group", "label", "kind"))
            for i, j in zip(noisy_indices, clean_indices, strict=True)
        ),
        "Paired fitting rows must bind the same foreground, truth and group",
    )
    return design[noisy_indices] - design[clean_indices], weights([rows[i] for i in noisy_indices])


def objective(beta, design, target, row_weights, ridge, deltas, pair_weights, strength):
    logits = design @ beta
    loss = row_weights @ (np.logaddexp(0, logits) - target * logits) + 0.5 * np.dot(ridge, beta**2)
    gradient = design.T @ (row_weights * (expit(logits) - target)) + ridge * beta
    if strength:
        drift = deltas @ beta
        loss += 0.5 * strength * (pair_weights @ (drift**2))
        gradient += strength * (deltas.T @ (pair_weights * drift))
    return float(loss), gradient


def fit_head(rows, arm):
    require(
        arm in FITS and rows and all(row["role"] == ROLES[0] for row in rows),
        "Only assigned fitting rows may fit the head",
    )
    ready = assessed(rows)
    row_weights = weights(ready)
    transform = None if arm == "raw_only" else fit_transform(ready, row_weights)
    design = matrix(ready, arm, transform)
    deltas, pair_weights = pair_deltas(ready, design)
    target = np.asarray([row["label"] for row in ready], dtype=np.float64)
    ridge = np.full(design.shape[1], 0.05)
    ridge[0], ridge[-1] = 0.001, 0
    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        args=(design, target, row_weights, ridge, deltas, pair_weights, FITS[arm]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None), *([(None, None)] * (design.shape[1] - 1))],
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    )
    require(result.success and np.isfinite(result.x).all(), "Fixed head fit failed to converge")
    return {
        "arm": arm,
        "transform": transform,
        "coefficients": result.x.tolist(),
        "fitting_assessed": len(ready),
        "paired_fitting_deltas": len(deltas),
        "optimizer_iterations": int(result.nit),
    }


def method_scores(rows, heads):
    ready = assessed(rows)
    baseline = {
        "current": [row["original_p"] for row in ready],
        "v13_complement": [row["candidates"]["complement"]["p"] for row in ready],
        "v83_global": [row["v83_global_p"] for row in ready],
    }
    for name, head in heads.items():
        baseline[name] = expit(matrix(ready, name, head["transform"]) @ head["coefficients"])
    result = {}
    for name, values in baseline.items():
        bounded_logit(values)
        result[name] = {row["id"]: float(value) for row, value in zip(ready, values, strict=True)}
    return result


def group_tail_cutpoint(rows, scores, truth, budget):
    tails = {}
    for row in rows:
        if row["label"] == truth and row["id"] in scores:
            group, value = row["group"], scores[row["id"]]
            extreme = max if truth == 0 else min
            tails[group] = extreme(tails.get(group, value), value)
    require(bool(tails) and 0 <= budget <= len(tails), "Calibration tail budget needs assessed source groups")
    ordered = sorted(tails.values(), reverse=truth == 0)
    if budget == len(ordered):
        return 0.0 if truth == 0 else 1.0
    return ordered[budget]


def match_budgets(rows, all_scores):
    require(
        all(row["role"] == ROLES[1] for row in rows), "Only calibration sources can set diagnostic cutpoints"
    )
    noisy = [row for row in assessed(rows) if row["condition"] != "clean"]
    false_ai = len({row["group"] for row in noisy if row["label"] == 0 and row["original_decision"] == "ai"})
    false_original = len(
        {row["group"] for row in noisy if row["label"] == 1 and row["original_decision"] == "original"}
    )
    points = {}
    for method, scores in all_scores.items():
        high = group_tail_cutpoint(noisy, scores, truth=0, budget=false_ai)
        low = group_tail_cutpoint(noisy, scores, truth=1, budget=false_original)
        points[method] = {"ai_above": high, "original_below": min(low, high)}
    return {
        "current_false_ai_group_budget": false_ai,
        "current_wrong_original_group_budget": false_original,
        "points": points,
    }


def score_direction(value, point):
    if value is None:
        return "unavailable"
    if value > point["ai_above"]:
        return "ai"
    if value < point["original_below"]:
        return "original"
    return "inconclusive"


def rank_auc(rows, scores):
    ai = [scores[row["id"]] for row in rows if row["label"] == 1 and row["id"] in scores]
    genuine = [scores[row["id"]] for row in rows if row["label"] == 0 and row["id"] in scores]
    require(bool(ai and genuine), "Rank comparison needs both assessed classes")
    difference = np.asarray(ai)[:, None] - np.asarray(genuine)
    wins, ties = int((difference > 0).sum()), int((difference == 0).sum())
    return {
        "comparisons": int(difference.size),
        "wins": wins,
        "ties": ties,
        "auc": (wins + 0.5 * ties) / difference.size,
    }


def decision_counts(rows, directions):
    counts = {}
    for label, role in ((0, "genuine"), (1, "synthetic")):
        truth = [row for row in rows if row["label"] == label]
        values = Counter(directions[row["id"]] for row in truth)
        wrong = "ai" if label == 0 else "original"
        counts[role] = {
            "planned": len(truth),
            "directions": {key: values[key] for key in DIRECTIONS},
            "wrong_direction_groups": len({row["group"] for row in truth if directions[row["id"]] == wrong}),
        }
    return counts


def drift(rows, scores, condition):
    clean = {
        row["base_id"]: scores[row["id"]]
        for row in rows
        if row["condition"] == "clean" and row["id"] in scores
    }
    pairs = [
        row
        for row in rows
        if row["condition"] == condition and row["id"] in scores and row["base_id"] in clean
    ]
    by_truth = {}
    for label, name in ((0, "genuine"), (1, "synthetic")):
        values = [
            float(bounded_logit(scores[row["id"]]) - bounded_logit(clean[row["base_id"]]))
            for row in pairs
            if row["label"] == label
        ]
        by_truth[name] = {
            "pairs": len(values),
            "median_absolute_logit_drift": float(np.median(np.abs(values))) if values else None,
        }
    return by_truth


def compare(rows, scores, calibrated):
    selected = {name: [row for row in rows if row["condition"] == name] for name in CONDITIONS}
    selected["noisy_combined"] = [row for row in rows if row["condition"] != "clean"]
    current_directions = {
        row["id"]: row["original_decision"] if row["status"] == "ok" else "unavailable" for row in rows
    }
    require(
        all(value in DIRECTIONS for value in current_directions.values()), "Exact serving directions required"
    )
    result = {
        "recorded_current_directions": {
            name: decision_counts(group, current_directions) for name, group in selected.items()
        }
    }
    for name, values in scores.items():
        point = calibrated["points"][name]
        directions = {row["id"]: score_direction(values.get(row["id"]), point) for row in rows}
        result[name] = {
            "matched_calibration_point_directions": {
                condition: decision_counts(group, directions) for condition, group in selected.items()
            },
            "raw_ranks": {condition: rank_auc(group, values) for condition, group in selected.items()},
            "score_drift": {condition: drift(rows, values, condition) for condition in CONDITIONS[1:]},
        }
    return result


def identity(source):
    return {
        "schema": "chipmunk.audio-paired-consistency-freeze/223",
        "source_identity": source,
        "roles": list(ROLES),
        "conditions": list(CONDITIONS),
        "pair_penalty_by_arm": FITS,
        "baseline_score_methods": list(BASELINES),
        "code_sha256": {name: digest(ROOT / name) for name in CODE},
        "numeric_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        "fitting_source": "fusion_fit only; separate confidence_cal matched group budgets",
        "confirmation_feature_or_media_access_permitted": False,
    }


def main():
    from scripts.audio_paired_input_v223 import load_roles, source_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve any earlier fitting attempt")
        frozen = identity(source_identity())
        TRIAL.mkdir(mode=0o700)
        write(TRIAL / "freeze.json", frozen)
        write(PUBLIC, frozen)
        print(json.dumps({"freeze_sha256": digest(PUBLIC), "status": "frozen"}))
        return
    frozen = read(PUBLIC)
    require(
        read(TRIAL / "freeze.json") == frozen == identity(source_identity()),
        "Exact prospective feature and method freeze required",
    )
    require(not (TRIAL / "fitted.json").exists(), "Preserve any earlier fitted object")
    roles, source = load_roles()
    require(source == frozen["source_identity"] and tuple(roles) == ROLES, "Exact frozen roles required")
    heads = {name: fit_head(roles[ROLES[0]], name) for name in FITS}
    write(TRIAL / "fitted.json", heads)
    calibrated = match_budgets(roles[ROLES[1]], method_scores(roles[ROLES[1]], heads))
    write(TRIAL / "calibration.json", calibrated)
    report = {
        "schema": "chipmunk.audio-paired-consistency-result/223",
        "status": "complete",
        "freeze_sha256": digest(PUBLIC),
        "private_fit_sha256": digest(TRIAL / "fitted.json"),
        "private_calibration_sha256": digest(TRIAL / "calibration.json"),
        "fit_coverage": {
            name: {
                key: head[key]
                for key in ("fitting_assessed", "paired_fitting_deltas", "optimizer_iterations")
            }
            for name, head in heads.items()
        },
        "calibration_group_budgets": {key: value for key, value in calibrated.items() if key != "points"},
        "consumed_selection": compare(roles[ROLES[2]], method_scores(roles[ROLES[2]], heads), calibrated),
        "new_neural_calls": 0,
        "confirmation_feature_or_media_read": False,
        "serving_admitted": False,
    }
    write(TRIAL / "summary.json", report)
    print(json.dumps({"status": "complete", "summary_sha256": digest(TRIAL / "summary.json")}))


if __name__ == "__main__":
    main()
