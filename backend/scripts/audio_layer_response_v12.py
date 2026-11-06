"""Prospective fixed-layer versus unprojected-final response mapping.

Synthetic tests only until the complete source/execution freeze is published.
Both arms keep the same objective, variance budget and independent guards.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.audio_calibration_v3_methods import clipped_logit
from scripts.audio_layer_features_v12 import LAYER_IDS, WIDTH
from scripts.audio_multisource_v8 import balanced_weights
from scripts.review_audio_noise_v4 import valid

ARMS = ("final_control", "layers")
METHOD = {
    "version": "audio-fixed-layer-response/12.0",
    "arms": {
        "final_control": "Unprojected original final pooled1024-vector",
        "layers": "Concatenate original masked means of hidden-state indices8,16,24, each1024-wide",
    },
    "layer_ids": list(LAYER_IDS),
    "pretrained_parameters": "All frozen; original waveforms/fbank/masks and neural output unchanged",
    "predictors": "Raw clipped logit, selected full representation and intercept; no metadata/quality predictors",
    "normalization": "Fitting-only weighted center/diagonal standard deviations, floor1e-6; sqrt(16/d) factor gives variance budget at most16; no head-direction projection in either arm",
    "ridge": {"raw": 0.001, "representation": 0.05, "intercept": 0.0},
    "target_smoothing": "Same connected fitting-group counts by truth class as v9/v11",
    "weighting": "Same equal truth/family/group/source/assessed-variant weights, sum1",
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "original_slope": "Explicit raw-coordinate coefficient nonnegative; overlapping representation unconstrained, no physical monotonicity claim",
    "reference_prior": 0.5,
    "guards": "Unchanged independent-calibration worst nonconformity per group/truth, retaining unassessed variants",
    "stability": "Both raw and mapped logit changes at most ln2, with original six-decimal arithmetic",
    "selection": "Only arms passing every unchanged service-baseline gate eligible; prefer final control if both pass unless layers add>=5 correct with neither false-call direction higher and Brier no worse",
    "deployment_eligible": False,
}


def feature(row, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed layer arm")
    if not valid(row.get("raw")):
        return None
    if arm == "layers":
        if row.get("layer_ids") != list(LAYER_IDS):
            raise ValueError("Exact preselected layer order required")
        value = np.asarray(row.get("layer_embeddings"), np.float64)
        shape = (len(LAYER_IDS), WIDTH)
    else:
        value = np.asarray(row.get("embedding"), np.float64)
        shape = (WIDTH,)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Every assessed source requires its complete finite representation")
    return value.reshape(-1)


def design(row, transform, arm):
    values = feature(row, arm)
    if values is None:
        return None
    values = (values - transform["center"]) / transform["scale"] * transform["variance_factor"]
    return np.r_[clipped_logit(row["raw"]), values, 1.0]


def fit(rows, arm):
    if not rows or any(r["role"] != "fit" for r in rows):
        raise ValueError("Only fitting sources may learn normalization and coefficients")
    selected = [r for r in rows if valid(r.get("raw"))]
    w = balanced_weights(selected)
    e = np.asarray([feature(r, arm) for r in selected])
    center = w @ e
    scale = np.maximum(np.sqrt(w @ (e - center) ** 2), 1e-6)
    transform = {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "variance_factor": float(np.sqrt(16 / e.shape[1])),
        "fit_sources": len({r["base_id"] for r in selected}),
    }
    x = np.asarray([design(r, transform, arm) for r in selected])
    y = np.asarray([r["label"] for r in selected])
    counts = [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)]
    if min(counts) < 2:
        raise ValueError("At least two fitting groups per truth class required")
    target = np.where(y == 1, (counts[1] + 1) / (counts[1] + 2), 1 / (counts[0] + 2))
    ridge = np.full(x.shape[1], METHOD["ridge"]["representation"])
    ridge[0], ridge[-1] = METHOD["ridge"]["raw"], METHOD["ridge"]["intercept"]

    def objective(beta):
        z = x @ beta
        return (
            float(w @ (np.logaddexp(0, z) - target * z) + 0.5 * np.dot(ridge, beta**2)),
            x.T @ (w * (expit(z) - target)) + ridge * beta,
        )

    initial = np.zeros(x.shape[1])
    initial[0] = 1
    config = METHOD["solver"]
    solved = minimize(
        objective,
        initial,
        jac=True,
        method=config["method"],
        bounds=[(0, None)] + [(None, None)] * (len(initial) - 1),
        options={k: config[k] for k in ("ftol", "gtol", "maxiter")},
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("Fixed layer-response objective did not converge: " + solved.message)
    return {
        "method": METHOD,
        "arm": arm,
        "transform": transform,
        "coefficients": solved.x.tolist(),
        "fit_groups_by_class": counts,
        "reference_prior": 0.5,
    }


def predict(row, fitted):
    x = design(row, fitted["transform"], fitted["arm"])
    return None if x is None else float(expit(x @ fitted["coefficients"]))


def calibrate(rows, fitted):
    if not rows or any(r["role"] != "cal" for r in rows):
        raise ValueError("Only independent calibration sources may set guards")
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Integer truth labels required")
        p = predict(row, fitted)
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    result = {
        str(c): [
            {
                "group": g,
                "maximum": None if any(v is None for v in values) else max(values),
                "rows": len(values),
            }
            for g, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }
    if min(len(result[str(c)]) for c in (0, 1)) < 9:
        raise ValueError("At least nine independent calibration groups per truth class required")
    return result


def select(summaries):
    passing = [a for a in ARMS if summaries[a]["advance_to_fresh_confirmation"]]
    if len(passing) < 2:
        return passing[0] if passing else None
    a, b = [summaries[arm]["groups"]["all"]["candidate"] for arm in ARMS]
    gain = b["correct_ai"] + b["correct_original"] - a["correct_ai"] - a["correct_original"]
    return (
        "layers"
        if gain >= 5
        and b["false_ai"] <= a["false_ai"]
        and b["false_original"] <= a["false_original"]
        and b["brier"] <= a["brier"]
        else "final_control"
    )
