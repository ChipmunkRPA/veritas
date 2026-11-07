"""Two fixed noise-aware feature learners; own fitting and calibration only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.audio_calibration_v3_methods import clipped_logit
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_wavlm_features_v13 import DIMENSIONS, SPEC
from scripts.review_audio_conditional_v10 import guards
from scripts.review_audio_noise_v4 import valid

ARMS = ("standalone", "complement")
METHOD = {
    "version": "audio-noise-aware-wavlm/13.0", "backbone": SPEC,
    "arms": {"standalone": "4608 frozen WavLM mean/std features and intercept",
             "complement": "Same WavLM features plus original clipped raw logit and intercept"},
    "weighting": "Same equal truth/family/group/source/assessed-variant fitting weights",
    "normalization": "Fitting-only weighted center/std floor1e-6 and sqrt(16/4608) variance factor",
    "ridge": {"representation": 0.05, "raw": 0.001, "intercept": 0.0},
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "target_smoothing": "Original connected fitting-group truth-class counts",
    "guards": "Independent calibration worst group/truth, preserving unassessed values",
    "stability": "Original raw and candidate logit change<=ln2 at six-decimal probabilities",
    "scope": "Original four-second assessed scope and statuses; no source metadata as predictors",
    "selection": "Every original16 gates required; prefer standalone unless complement adds>=5 correct with neither wrong direction higher and Brier no worse",
    "reference_prior": 0.5, "deployment_eligible": False,
}


def feature(row):
    if row.get("raw") is None:
        if row.get("wavlm") is not None:
            raise ValueError("Unassessed original input cannot supply candidate features")
        return None
    if isinstance(row["raw"], bool) or not valid(row["raw"]):
        raise ValueError("Original raw probability must be finite and in range")
    x = np.asarray(row.get("wavlm"), np.float64)
    if x.shape != (DIMENSIONS,) or not np.isfinite(x).all():
        raise ValueError("All finite frozen features required for an assessed source")
    return x


def design(row, transform, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed candidate arm")
    x = feature(row)
    if x is None:
        return None
    x = (x - transform["center"]) / transform["scale"] * transform["variance_factor"]
    return np.r_[clipped_logit(row["raw"]), x, 1.] if arm == "complement" else np.r_[x, 1.]


def fit(rows, arm):
    if not rows or any(r["role"] != "fit" for r in rows) or arm not in ARMS:
        raise ValueError("Only original fitting roles may learn either fixed arm")
    selected = [r for r in rows if feature(r) is not None]
    weights = balanced_weights(selected)
    matrix = np.asarray([feature(r) for r in selected])
    center = weights @ matrix
    transform = {"center": center.tolist(),
                 "scale": np.maximum(np.sqrt(weights @ (matrix - center) ** 2), 1e-6).tolist(),
                 "variance_factor": float(np.sqrt(16 / DIMENSIONS)),
                 "fit_sources": len({r["base_id"] for r in selected})}
    x = np.asarray([design(r, transform, arm) for r in selected])
    counts = [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)]
    if min(counts) < 2:
        raise ValueError("At least two fitting groups per truth class required")
    y = np.asarray([r["label"] for r in selected])
    target = np.where(y == 1, (counts[1] + 1) / (counts[1] + 2), 1 / (counts[0] + 2))
    ridge = np.full(x.shape[1], METHOD["ridge"]["representation"])
    ridge[-1] = 0.
    start = np.zeros(x.shape[1])
    bounds = [(None, None)] * x.shape[1]
    if arm == "complement":
        ridge[0], start[0], bounds[0] = METHOD["ridge"]["raw"], 1., (0, None)

    def objective(beta):
        z = x @ beta
        return (float(weights @ (np.logaddexp(0, z) - target * z) + .5 * np.dot(ridge, beta**2)),
                x.T @ (weights * (expit(z) - target)) + ridge * beta)

    solver = METHOD["solver"]
    solved = minimize(objective, start, jac=True, method=solver["method"], bounds=bounds,
                      options={k: solver[k] for k in ("ftol", "gtol", "maxiter")})
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("Fixed noise-aware objective failed to converge: " + solved.message)
    return {"method": METHOD, "arm": arm, "transform": transform,
            "coefficients": solved.x.tolist(), "fit_groups_by_class": counts, "reference_prior": .5}


def predict(row, fitted):
    x = design(row, fitted["transform"], fitted["arm"])
    return None if x is None else float(expit(x @ fitted["coefficients"]))


def calibrate(rows, fitted):
    if not rows or any(r["role"] != "cal" or type(r["label"]) is not int or r["label"] not in (0, 1) for r in rows):
        raise ValueError("Independent calibration roles and integer truth labels required")
    result = guards(rows, [predict(r, fitted) for r in rows])
    if min(len(result[str(c)]) for c in (0, 1)) < 9:
        raise ValueError("At least nine independent calibration groups per truth class required")
    return result


def select(summaries):
    allowed = [a for a in ARMS if summaries[a]["advance_to_fresh_confirmation"]]
    if len(allowed) < 2:
        return allowed[0] if allowed else None
    a, b = [summaries[arm]["groups"]["all"]["candidate"] for arm in ARMS]
    gain = b["correct_ai"] + b["correct_original"] - a["correct_ai"] - a["correct_original"]
    return ("complement" if gain >= 5 and b["false_ai"] <= a["false_ai"]
            and b["false_original"] <= a["false_original"] and b["brier"] <= a["brier"] else "standalone")
