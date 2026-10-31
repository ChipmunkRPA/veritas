"""Stage 2 monotone logistic stacker (ML §4.2; ARCHITECTURE §4.4).

    η(x) = b + βᵀc + Σ_j [w_j⁺ z_j⁺ + w_j⁻ z_j⁻] + Σ_j Σ_{g∈G} [v_{j,g}⁺ z_j⁺ + v_{j,g}⁻ z_j⁻]·g + Σ_j μ_j m_j

- **Monotone in every context.** η is non-decreasing in z_j when the effective coefficient w + Σ_{g active} v_g is ≥ 0
  on both sides for every set of groups that can be active together. has_face and jpeg_q<70 co-occur, so ML's
  v = u − w with u ≥ 0 is not enough. The solver uses w = t + Σ_g r_g and v_g = u_g − r_g with t, r, u ≥ 0. That covers
  exactly the feasible set {w ≥ 0, w + Σ_g min(v_g, 0) ≥ 0}, and every constraint becomes a bound for L-BFGS-B.
- **Objective.** Σ_i ω_i·logloss_i + λ·(‖w‖² + ‖v‖² + ‖β‖² + ‖μ‖²) on the natural coefficients (bias unpenalised),
  plus 1e-9·‖θ₊‖² on the bounded solver parameters, which only removes the null direction of the reparameterisation.
  ω are class- and stratum-balancing weights normalised to Σω = n.
- **λ** from {0.01, 0.03, 0.1, 0.3, 1, 3} by 5-fold StratifiedGroupKFold on `fit` (seed 7), minimising the ω-weighted
  out-of-fold log-loss; ties keep the smaller λ.
- **Determinism.** Callers sort rows by item_id; θ starts at zero; tolerances are tight.
- **D-45.** `resolution_weights` multiply the reals' ω by P(b | fake) / P(b | real) over min-side buckets; `fixed_missing`
  bounds a detector's missing coefficient μ_j to [0, 0] (a size-driven not applicable carries no weight).
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from app.engine.fusion.features import FeatureSpace, fp_guarded

L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CV_SPLITS = 5
IDENTIFIABILITY_EPS = 1e-9
SOLVER_OPTIONS = {"maxiter": 20000, "maxfun": 50000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 30}


@dataclass(frozen=True)
class StackerFit:
    coef: np.ndarray  # natural coefficients γ in FeatureSpace.names order
    l2: float
    objective: float
    n_iter: int
    converged: bool
    message: str


def _labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("labels must be a 1-D array of 0 (real) and 1 (AI or manipulated)")
    if len(np.unique(y)) < 2:
        raise ValueError("fitting needs both classes")
    return y.astype(np.float64)


RESOLUTION_EDGES = (384, 768, 1025, 1537)  # min_side buckets <384, 384–767, 768–1024, 1025–1536, >1536 (D-45)
RESOLUTION_MIN_CLASS_N = 20
RESOLUTION_CLIP = (0.1, 10.0)
RESOLUTION_MAX_ITER, RESOLUTION_TOL = 10_000, 1e-13  # D-50: the stratum-weighted fixed point converges geometrically under clipping


def balancing_weights(y, strata: Sequence, *, multipliers=None) -> np.ndarray:
    """ω: each class carries half the total weight, split equally across its strata; Σω = n. `multipliers` (e.g.
    `resolution_weights`) scale items within their class, which is then renormalised to half the total weight."""
    y = _labels(y)
    strata = np.asarray([str(s) for s in strata], dtype=object)
    if strata.shape != y.shape:
        raise ValueError("labels and strata differ in length")
    out = np.zeros(len(y))
    for label in (0.0, 1.0):
        mask = y == label
        cells = sorted(set(strata[mask].tolist()))
        for cell in cells:
            member = mask & (strata == cell)
            out[member] = len(y) / (2.0 * len(cells) * member.sum())
    if multipliers is not None:
        m = np.asarray(multipliers, dtype=np.float64)
        if m.shape != y.shape or not (np.isfinite(m).all() and (m > 0).all()):
            raise ValueError("multipliers must be positive, finite and one per item")
        out = out * m
        for label in (0.0, 1.0):
            mask = y == label
            out[mask] *= (len(y) / 2.0) / out[mask].sum()
    return out


def resolution_buckets(y, min_side, *, min_class_n: int = RESOLUTION_MIN_CLASS_N) -> np.ndarray:
    """Min-side bucket index per item (−1 without a min side). A bucket where either class has fewer than `min_class_n`
    items merges into its right neighbour (the left one for the last bucket), repeatedly, until every bucket holds both."""
    y = np.asarray(y)
    side = np.asarray(min_side, dtype=np.float64)
    if side.shape != y.shape:
        raise ValueError("labels and min sides differ in length")
    known = np.isfinite(side)
    raw = np.where(known, np.searchsorted(RESOLUTION_EDGES, np.nan_to_num(side), side="right"), -1)
    groups = [[k] for k in range(len(RESOLUTION_EDGES) + 1)]

    def smaller_class(members: list[int]) -> int:
        mask = known & np.isin(raw, members)
        return int(min((y[mask] == 0).sum(), (y[mask] == 1).sum()))

    while len(groups) > 1:
        small = next((i for i, g in enumerate(groups) if smaller_class(g) < min_class_n), None)
        if small is None:
            break
        lo, hi = sorted((small, small + 1 if small + 1 < len(groups) else small - 1))
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    out = np.full(len(y), -1)
    for index, members in enumerate(groups):
        out[known & np.isin(raw, members)] = index
    return out


def resolution_weights(y, min_side, *, strata: Sequence | None = None, min_class_n: int = RESOLUTION_MIN_CLASS_N,
                       clip: tuple[float, float] = RESOLUTION_CLIP) -> np.ndarray:
    """D-45 multipliers: 1 for fakes; for a real in bucket b the fake share of b over the real share of b, clipped. Items
    without a min side keep 1.

    - Without `strata`: shares are item counts, P(b | fake) / P(b | real) (J-2).
    - D-50, with `strata`: shares are masses under the class- and stratum-balancing weights (`balancing_weights(y, strata)`),
      and the multipliers are recomputed while clipping moves them. For every unclipped bucket the real share under
      `balancing_weights(y, strata, multipliers=…)` then equals the fake share; clipped buckets absorb the rest."""
    y = _labels(y)
    buckets = resolution_buckets(y, min_side, min_class_n=min_class_n)
    out = np.ones(len(y))
    known = buckets >= 0
    n1, n0 = int(((y == 1) & known).sum()), int(((y == 0) & known).sum())
    if n1 == 0 or n0 == 0:
        return out
    if strata is None:
        for b in np.unique(buckets[known]):
            reals = (buckets == b) & (y == 0)
            if reals.any():
                out[reals] = float(np.clip((((buckets == b) & (y == 1)).sum() / n1) / (reals.sum() / n0), *clip))
        return out
    base = balancing_weights(y, strata)
    ids = np.unique(buckets[known])
    fake_mass = np.asarray([base[(y == 1) & (buckets == b)].sum() for b in ids])
    real_mass = np.asarray([base[(y == 0) & (buckets == b)].sum() for b in ids])
    target, has_real, m = fake_mass / fake_mass.sum(), real_mass > 0, np.ones(len(ids))
    for _ in range(RESOLUTION_MAX_ITER):  # bucket-level fixed point: m_b ← clip(m_b · fake share_b / real share_b)
        share = m * real_mass / (m * real_mass).sum()
        new = m.copy()
        new[has_real] = np.clip(m[has_real] * target[has_real] / share[has_real], *clip)
        done = float(np.max(np.abs(new - m))) <= RESOLUTION_TOL
        m = new
        if done:
            break
    for k, b in enumerate(ids):
        out[(y == 0) & (buckets == b)] = m[k]
    return out


def reparameterisation(space: FeatureSpace, fixed_missing: Collection[str] = ()) -> tuple[
        np.ndarray, list[tuple[float | None, float | None]], np.ndarray]:
    """T with γ = T·θ, the L-BFGS-B bounds of θ, and the mask of bounded θ entries. The missing coefficients of
    `fixed_missing` detectors are bounded to [0, 0]."""
    unknown = sorted(set(fixed_missing) - set(space.detectors))
    if unknown:
        raise ValueError(f"fixed missing weights name detectors outside the feature space: {unknown}")
    n_ctx, n_groups, n_det = len(space.context_terms), len(space.groups), len(space.detectors)
    free = 1 + n_ctx + n_det
    t = np.zeros((len(space.names), free + n_det * 2 * (1 + 2 * n_groups)))
    col = 0
    for k in range(1 + n_ctx):  # bias and context terms
        t[k, col] = 1.0
        col += 1
    for j in range(n_det):  # missing-mask coefficients μ_j
        t[space.offset(j) + space.block - 1, col] = 1.0
        col += 1
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * free
    for j, det in enumerate(space.detectors):
        if det in fixed_missing:
            bounds[1 + n_ctx + j] = (0.0, 0.0)
    for j in range(n_det):
        off = space.offset(j)
        for side in (0, 1):  # z⁺ then z⁻
            w_idx = off + side
            t[w_idx, col] = 1.0  # t
            col += 1
            bounds.append((0.0, None))
            for gi in range(n_groups):
                v_idx = off + 2 + side * n_groups + gi
                t[w_idx, col] = 1.0  # r_g adds to w …
                t[v_idx, col] = -1.0  # … and subtracts from v_g
                t[v_idx, col + 1] = 1.0  # u_g
                col += 2
                bounds += [(0.0, None), (0.0, None)]
    bounded = np.zeros(t.shape[1], dtype=bool)
    bounded[free:] = True
    return t, bounds, bounded


@fp_guarded
def fit_stacker(a: np.ndarray, y, weights: np.ndarray, space: FeatureSpace, l2: float, *,
                fixed_missing: Collection[str] = ()) -> StackerFit:
    a = np.asarray(a, dtype=np.float64)
    y = _labels(y)
    weights = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(space.names):
        raise ValueError(f"design matrix needs {len(space.names)} columns, got shape {a.shape}")
    if a.shape[0] != len(y) or len(weights) != len(y):
        raise ValueError("design matrix, labels and weights differ in length")
    if not np.isfinite(a).all() or not (weights > 0).all():
        raise ValueError("design matrix must be finite and weights positive")
    if l2 < 0:
        raise ValueError("λ must be ≥ 0")
    t, bounds, bounded = reparameterisation(space, fixed_missing)
    b = a @ t
    penalty = np.ones(len(space.names))
    penalty[0] = 0.0
    eps = IDENTIFIABILITY_EPS * bounded

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = b @ theta
        gamma = t @ theta
        loss = float(np.dot(weights, np.logaddexp(0.0, eta) - y * eta))
        reg = l2 * float(np.dot(penalty, gamma * gamma)) + float(np.dot(eps, theta * theta))
        residual = weights * (expit(eta) - y)
        grad = b.T @ residual + t.T @ (2.0 * l2 * penalty * gamma) + 2.0 * eps * theta
        return loss + reg, grad

    result = minimize(objective, np.zeros(t.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds,
                      options=SOLVER_OPTIONS)
    coef = t @ result.x
    if not (np.isfinite(coef).all() and math.isfinite(result.fun)):
        raise FloatingPointError("stacker fit produced non-finite coefficients")
    return StackerFit(coef=coef, l2=float(l2), objective=float(result.fun), n_iter=int(result.nit),
                      converged=bool(result.success), message=str(result.message))


def weighted_logloss(eta: np.ndarray, y, weights: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(np.dot(weights, np.logaddexp(0.0, eta) - y * eta) / weights.sum())


@fp_guarded
def select_l2(a: np.ndarray, y, weights: np.ndarray, groups: Sequence, space: FeatureSpace, *,
              grid: Sequence[float] = L2_GRID, seed: int = 7, n_splits: int = CV_SPLITS,
              fixed_missing: Collection[str] = ()) -> tuple[float, list[dict]]:
    """λ with the lowest ω-weighted out-of-fold log-loss under StratifiedGroupKFold (whole groups per fold)."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = _labels(y).astype(int)
    groups = np.asarray([str(g) for g in groups], dtype=object)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"grouped CV needs at least {n_splits} groups, got {len(np.unique(groups))}")
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(a, y, groups))
    table = []
    for l2 in sorted(float(v) for v in grid):
        num = den = 0.0
        for train, held in folds:
            coef = fit_stacker(a[train], y[train], weights[train], space, l2, fixed_missing=fixed_missing).coef
            eta = a[held] @ coef
            if not np.isfinite(eta).all():
                raise FloatingPointError("non-finite out-of-fold stacker logits")
            num += float(np.dot(weights[held], np.logaddexp(0.0, eta) - y[held] * eta))
            den += float(weights[held].sum())
        table.append({"l2": l2, "cv_logloss": num / den})
    best = min(table, key=lambda row: row["cv_logloss"])
    return best["l2"], table
