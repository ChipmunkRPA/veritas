"""Grouped bootstrap of Stage 2 + Stage 3 (ML §5.2): B refits on resampled `fit` / `cal` groups → B × P float32.

- Columns: the stacker coefficients in `FeatureSpace.names` order, then `platt:global:a|b` and `platt:<k>:a|b` for every
  context in PLATT_CONTEXTS (a context absent from a replicate's `cal` gets the replicate's global parameters).
- λ is the value chosen on the full `fit` (no nested CV). Replicate b draws from
  `default_rng(SeedSequence(seed).spawn(B)[b])`; a resample that lacks a class draws again from the same generator, so the
  matrix is deterministic given the seed.
- At inference s_b = σ(a_{k,b}·(x·γ_b) + b_{k,b}) and the 90 % interval is [q₀.₀₅, q₀.₉₅] of the s_b. It captures fuser
  weight and calibration uncertainty only, not detector epistemic error or distribution shift.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from app.engine.fusion.calibration import fit_calibration, platt_params
from app.engine.fusion.features import PLATT_CONTEXTS, FeatureSpace, fp_guarded
from app.engine.fusion.policy import LEGACY, SizePolicy
from app.engine.fusion.stacker import balancing_weights, fit_stacker, resolution_weights

MAX_REDRAWS = 100


@dataclass(frozen=True)
class SplitData:
    a: np.ndarray  # design matrix
    y: np.ndarray
    groups: np.ndarray
    strata: np.ndarray
    contexts: np.ndarray  # Platt context per row
    min_side: np.ndarray | None = None  # per row, NaN when unknown (resolution balancing, D-45)


def bootstrap_columns(space: FeatureSpace) -> list[str]:
    cols = list(space.names) + ["platt:global:a", "platt:global:b"]
    for k in PLATT_CONTEXTS:
        cols += [f"platt:{k}:a", f"platt:{k}:b"]
    return cols


def grouped_resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a resample of whole groups with replacement (groups in sorted order, members in row order)."""
    keys, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(keys))]
    draw = rng.integers(0, len(keys), size=len(keys))
    return np.concatenate([members[i] for i in draw])


def _draw(data: SplitData, rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_REDRAWS):
        idx = grouped_resample(data.groups, rng)
        if len(np.unique(data.y[idx])) == 2:
            return idx
    raise ValueError("bootstrap resamples keep missing a class; the split has too few groups of one class")


@fp_guarded
def refit_matrix(fit: SplitData, cal: SplitData, space: FeatureSpace, l2: float, *, n_boot: int, seed: int,
                 policy: SizePolicy = LEGACY, fixed_missing: Collection[str] = ()) -> np.ndarray:
    """Replicates refit with the artefact's size policy: resolution weights recomputed on each resample (D-45)."""
    if n_boot < 1:
        raise ValueError("the bootstrap needs B ≥ 1")
    if policy.resolution_balance and (fit.min_side is None or cal.min_side is None):
        raise ValueError("resolution balancing needs the min side of every row")
    out = np.zeros((n_boot, len(bootstrap_columns(space))), dtype=np.float32)
    for b, child in enumerate(np.random.SeedSequence(seed).spawn(n_boot)):
        rng = np.random.default_rng(child)
        fi, ci = _draw(fit, rng), _draw(cal, rng)
        fit_mult = resolution_weights(fit.y[fi], fit.min_side[fi], strata=fit.strata[fi]) if policy.resolution_balance else None
        weights = balancing_weights(fit.y[fi], fit.strata[fi], multipliers=fit_mult)
        coef = fit_stacker(fit.a[fi], fit.y[fi], weights, space, l2, fixed_missing=fixed_missing).coef
        cal_mult = (balancing_weights(cal.y[ci], cal.strata[ci], multipliers=resolution_weights(cal.y[ci], cal.min_side[ci], strata=cal.strata[ci]))
                    if policy.resolution_balance else None)  # D-50: the same ω as `_fit_head`
        calibration = fit_calibration(cal.a[ci] @ coef, cal.y[ci], cal.contexts[ci], weights=cal_mult,
                                      global_intercept=policy.global_intercept, centre=float(coef[0]))  # D-50: this replicate's bias
        platt = [calibration["global"]["a"], calibration["global"]["b"]]
        for k in PLATT_CONTEXTS:
            platt += list(platt_params(calibration, k))
        out[b] = np.concatenate([coef, np.asarray(platt)])
        if not np.isfinite(out[b]).all():
            raise FloatingPointError(f"bootstrap replicate {b} has non-finite parameters")
    return out


@fp_guarded
def interval(boot: np.ndarray, space: FeatureSpace, x: np.ndarray, context: str, *, prior_logit: float = 0.0,
             quantiles: Sequence[float] = (0.05, 0.95)) -> tuple[float, float]:
    """The per-item interval of s over bootstrap replicates (default 90 %)."""
    p = len(space.names)
    if boot.ndim != 2 or boot.shape[1] != len(bootstrap_columns(space)):
        raise ValueError("bootstrap matrix does not match the feature space")
    k = 0 if context not in PLATT_CONTEXTS else 1 + PLATT_CONTEXTS.index(context)
    matrix = boot.astype(np.float64)
    eta = matrix[:, :p] @ np.asarray(x, dtype=np.float64)
    s = expit(matrix[:, p + 2 * k] * eta + matrix[:, p + 2 * k + 1] + prior_logit)
    if not np.isfinite(s).all():
        raise FloatingPointError("non-finite bootstrap probabilities")
    lo, hi = np.quantile(s, list(quantiles))
    return float(lo), float(hi)
