"""Full frozen-representation supervised response, research only, never serving.

Retains all head-orthogonal coordinates instead of selecting sixteen PCs.
Fitting-only diagonal normalization has the same total variance budget of16;
the original class/family/source weighting, objective and guards stay fixed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.audio_calibration_v3_methods import clipped_logit
from scripts.audio_multisource_v8 import balanced_weights
from scripts.review_audio_noise_v4 import valid

METHOD = {
    "version": "audio-full-frozen-representation/11.0",
    "pretrained_parameters": "All frozen; exact original1024-dimensional predictor-head input",
    "predictors": "Original clipped raw logit, all1024 head-orthogonal coordinates and intercept; no metadata or waveform-quality predictors",
    "normalization": "Fitting-only weighted center and diagonal standard deviations, floor1e-6; multiply residual coordinates by sqrt(16/1024) to match PCA16 total-variance budget",
    "original_ridge": 0.001,
    "residual_ridge": 0.05,
    "intercept_ridge": 0.0,
    "original_slope": "nonnegative",
    "target_smoothing": "Unchanged original class-specific connected fitting-group counts",
    "weighting": "Unchanged equal truth/family/group/source/assessed-variant weights, sum1",
    "objective": "Same weighted smoothed binary log loss plus diagonal L2 as v9; only representation changes",
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "reference_prior": 0.5,
    "guards": "Unchanged independent-calibration worst group/truth nonconformity, retaining unassessed values",
    "stability": "Unchanged raw and mapped logit change<=ln2 at six-decimal probabilities",
    "new_pretrained_checkpoint": False,
    "deployment_eligible": False,
}


def design(row, transform):
    if not valid(row.get("raw")):
        return None
    e = np.asarray(row.get("embedding"), np.float64)
    h = np.asarray(transform["unit_head"], np.float64)
    if e.shape != (1024,) or h.shape != e.shape or not np.isfinite(e).all():
        raise ValueError("Every assessed input requires its finite original1024-vector")
    residual = e - np.dot(e, h) * h
    residual = (residual - transform["center"]) / transform["scale"]
    return np.r_[clipped_logit(row["raw"]), residual * transform["variance_factor"], 1.0]


def fit(rows, head_direction):
    if not rows or any(r["role"] != "fit" for r in rows):
        raise ValueError("Only fitting sources may learn normalization or coefficients")
    selected = [r for r in rows if valid(r.get("raw"))]
    w = balanced_weights(selected)
    e = np.asarray([r.get("embedding") for r in selected], np.float64)
    h = np.asarray(head_direction, np.float64)
    if (
        e.shape != (len(selected), 1024)
        or h.shape != (1024,)
        or not np.isfinite(e).all()
        or not np.isfinite(h).all()
        or np.linalg.norm(h) <= 0
    ):
        raise ValueError("Finite full representations and original head required")
    h = h / np.linalg.norm(h)
    residual = e - np.outer(e @ h, h)
    center = w @ residual
    scale = np.maximum(np.sqrt(w @ (residual - center) ** 2), 1e-6)
    transform = {
        "unit_head": h.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "variance_factor": float(np.sqrt(16 / 1024)),
        "fit_sources": len({r["base_id"] for r in selected}),
    }
    x = np.asarray([design(r, transform) for r in selected])
    y = np.asarray([r["label"] for r in selected])
    counts = [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)]
    if min(counts) < 2:
        raise ValueError("At least two fitting groups per truth class required")
    target = np.where(y == 1, (counts[1] + 1) / (counts[1] + 2), 1 / (counts[0] + 2))
    ridge = np.full(x.shape[1], METHOD["residual_ridge"])
    ridge[0], ridge[-1] = METHOD["original_ridge"], METHOD["intercept_ridge"]

    def objective(beta):
        z = x @ beta
        return (
            float(w @ (np.logaddexp(0, z) - target * z) + 0.5 * np.dot(ridge, beta**2)),
            x.T @ (w * (expit(z) - target)) + ridge * beta,
        )

    start = np.zeros(x.shape[1])
    start[0] = 1
    config = METHOD["solver"]
    result = minimize(
        objective,
        start,
        jac=True,
        method=config["method"],
        bounds=[(0, None)] + [(None, None)] * (len(start) - 1),
        options={k: config[k] for k in ("ftol", "gtol", "maxiter")},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Fixed full-representation objective did not converge: " + result.message)
    return {
        "method": METHOD,
        "transform": transform,
        "coefficients": result.x.tolist(),
        "fit_groups_by_class": counts,
        "fit_originals_by_class": [len({r["base_id"] for r in selected if r["label"] == c}) for c in (0, 1)],
        "reference_prior": 0.5,
    }


def predict(row, fitted):
    x = design(row, fitted["transform"])
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
