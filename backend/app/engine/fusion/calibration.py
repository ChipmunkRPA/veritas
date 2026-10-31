"""Stage 3 Platt calibration per context with shrinkage (ML §4.4; D-14).

- η' = a_k·η + b_k and s = σ(η'), fitted on `cal` by class-balanced log-loss, so s is a probability under a 50:50 prior
  and s/(1−s) is a likelihood ratio.
- Targets use Platt's smoothing t₁ = (n₁+1)/(n₁+2), t₀ = 1/(n₀+2), which keeps a finite on separable calibration data;
  a ≥ 0 keeps s non-decreasing in η.
- Per context k: a_k = (n_k·â_k + n₀·a_g)/(n_k + n₀) with n₀ = 200, and b_k the same way. A context with fewer than 100
  items of either class uses the global parameters.
- **D-45** (image stackers): `global_intercept` keeps b_k = b_g for every context (only the slopes are per context), and
  item `weights` (resolution balancing) scale the class-balanced log-loss within each class.
- A deployment with base rate π adds logit(π) to η' (`prior_logit`, `VERITAS_PRIOR_AI`); the default adds nothing.
- The [0.02, 0.98] clamp applies after calibration for display. The exact decomposition holds on the unclamped logit.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from app.engine.fusion.features import PLATT_CONTEXTS

SHRINKAGE_N0 = 200
MIN_CLASS_N = 100
PROB_FLOOR, PROB_CEILING = 0.02, 0.98
_OPTIONS = {"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10}


def _counts(y: np.ndarray) -> tuple[int, int]:
    n1 = int((y == 1).sum())
    return len(y) - n1, n1


def fit_platt(eta, y, *, weights=None, intercept: float | None = None) -> tuple[float, float]:
    """(a, b) minimising class-balanced log-loss against Platt's smoothed targets; a ≥ 0. `weights` scale items within their
    class (each class keeps half the total weight); `intercept` fixes b."""
    eta = np.asarray(eta, dtype=np.float64)
    y = np.asarray(y)
    if eta.shape != y.shape or eta.ndim != 1:
        raise ValueError("η and labels must be 1-D arrays of equal length")
    n0, n1 = _counts(y)
    if n0 == 0 or n1 == 0 or n0 + n1 != len(y):
        raise ValueError("Platt calibration needs both classes and labels in {0, 1}")
    n = len(y)
    target = np.where(y == 1, (n1 + 1) / (n1 + 2), 1.0 / (n0 + 2))
    weight = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape or not (np.isfinite(w).all() and (w > 0).all()):
            raise ValueError("Platt weights must be positive, finite and one per item")
        for label in (0, 1):
            mask = y == label
            weight[mask] = (n / 2.0) * w[mask] / w[mask].sum()
    b_bounds = (None, None) if intercept is None else (float(intercept), float(intercept))

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        z = params[0] * eta + params[1]
        loss = float(np.dot(weight, np.logaddexp(0.0, z) - target * z))
        residual = weight * (expit(z) - target)
        return loss, np.array([float(np.dot(residual, eta)), float(residual.sum())])

    start = np.array([1.0, 0.0 if intercept is None else float(intercept)])
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0.0, None), b_bounds], options=_OPTIONS)
    return float(result.x[0]), float(result.x[1])


def fit_calibration(eta, y, contexts: Sequence[str], *, shrinkage_n0: int = SHRINKAGE_N0,
                    min_class_n: int = MIN_CLASS_N, weights=None, global_intercept: bool = False, centre: float = 0.0) -> dict:
    """The artefact's `calibration` block: global Platt plus shrunk per-context parameters. With `global_intercept` every
    context keeps the global b and only its slope is fitted (b fixed) and shrunk (D-45).

    D-50: Platt is fitted on η − `centre` (the stacker bias) and every stored intercept is b_k = b′_k − a_k·centre. With
    `global_intercept` the effective intercept a_k·centre + b_k is then the same b′ in every context, so the context-only
    logit does not move with the Platt context. `platt_params` and the runtime read (a_k, b_k) on η as before."""
    centre = float(centre)
    eta = np.asarray(eta, dtype=np.float64) - centre
    y = np.asarray(y)
    contexts = np.asarray(list(contexts), dtype=object)
    if contexts.shape != y.shape:
        raise ValueError("contexts and labels differ in length")
    unknown = set(contexts.tolist()) - set(PLATT_CONTEXTS)
    if unknown:
        raise ValueError(f"unknown calibration contexts {sorted(unknown)}")
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    a_g, b_g = fit_platt(eta, y, weights=w)
    n0, n1 = _counts(y)
    doc = {"method": "platt", "shrinkage_n0": shrinkage_n0, "min_class_n": min_class_n, "centre": centre,
           "global": {"a": a_g, "b": b_g - a_g * centre, "n0": n0, "n1": n1}, "per_context": {}}
    if global_intercept:
        doc["intercept"] = "global"
    if w is not None:
        doc["weights"] = "class-balanced × item weights (resolution balancing, D-45)"
    for k in PLATT_CONTEXTS:
        mask = contexts == k
        if not mask.any():
            continue
        k0, k1 = _counts(y[mask])
        if k0 < min_class_n or k1 < min_class_n:
            doc["per_context"][k] = {"a": a_g, "b": b_g - a_g * centre, "n0": k0, "n1": k1, "source": "global"}
            continue
        a_hat, b_hat = fit_platt(eta[mask], y[mask], weights=None if w is None else w[mask],
                                 intercept=b_g if global_intercept else None)
        nk = k0 + k1
        a_k = (nk * a_hat + shrinkage_n0 * a_g) / (nk + shrinkage_n0)
        b_centred = b_g if global_intercept else (nk * b_hat + shrinkage_n0 * b_g) / (nk + shrinkage_n0)
        doc["per_context"][k] = {
            "a": a_k, "b": b_centred - a_k * centre,
            "a_hat": a_hat, "b_hat": b_hat - a_hat * centre, "n0": k0, "n1": k1, "source": "shrunk",
        }
    return doc


def platt_params(doc: dict, context: str) -> tuple[float, float]:
    """(a_k, b_k) for a context; a context absent from `cal` uses the global parameters."""
    entry = doc["per_context"].get(context) or doc["global"]
    return float(entry["a"]), float(entry["b"])


def prior_logit(prior_ai: float | None) -> float:
    if prior_ai is None:
        return 0.0
    if not 0.0 < prior_ai < 1.0:
        raise ValueError("the deployment prior must lie strictly between 0 and 1")
    return math.log(prior_ai / (1.0 - prior_ai))


def clamp_probability(s: float) -> float:
    return min(max(s, PROB_FLOOR), PROB_CEILING)
