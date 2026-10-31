"""Exact additive contributions (ML §4.6; ARCHITECTURE §4.4 invariants 1 and 5).

With Platt calibration, logit(s) = b'_k + Σ φ' exactly:
- detector row: φ'_j = a_k·[(w⁺ + Σ_g v⁺_g·g)·z⁺ + (w⁻ + Σ_g v⁻_g·g)·z⁻]; a missing detector's row is exactly 0;
- missing row: a_k·μ_j for a detector that did not run;
- a detector outside its context gate has a zero detector row (z None) and no missing row;
- so does an `inactive` detector, one that failed, timed out or was skipped (D-46);
- context prior: b'_k = a_k·(b + βᵀc) + b_k, plus logit(π) when a deployment prior is configured.

`pp` = σ(logit s) − σ(logit s − φ') in probability units; `effective_weight` = a_k × the context-effective coefficient on
the side of z's sign. The baseline is a detector reporting p = 0.5 (z = 0); for an additive model φ' is the exact
Shapley value against that baseline.

The neutral-point view (D-63; BUILD_SPEC N-17) is the same sum written against another baseline, for an explanation a
reader can follow. `decompose` is the bench decomposition and stays as it is.
- A check's neutral point z⁰_j is the score at which its stand-alone likelihood ratio is 1: a_j·z⁰_j + b_j = 0 with the
  check's univariate Platt of the artefact (`signal_calibration`).
- Row of a scoring check: φ'_j(z_j) − φ'_j(z⁰_j), the change in log-odds when the check moves from its neutral point to its
  score. Its sign is the sign of the check's stand-alone log likelihood ratio wherever its effective weight is not 0.
- Start: b'_k + Σ φ'_j(z⁰_j) over the scoring checks, the estimate with every scoring check at its neutral point, in this
  context. It does not depend on what the checks scored.
- Rows of a check that did not run (the learned missing term) are those of `decompose`. The total is the same logit.
- A check without a stand-alone calibration, or whose calibration has no neutral point inside the range of its scores
  (a slope of 0, or |z⁰| above the clip), keeps the 50 % baseline (z⁰ = 0) and its row says so (`standalone` False).
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from app.engine.fusion.features import Z_CLIP, Context, FeatureSpace


@dataclass(frozen=True)
class ContributionRow:
    kind: str  # "detector" | "missing"
    detector: str
    z: float | None
    logit: float
    pp: float
    effective_weight: float | None


@dataclass(frozen=True)
class Decomposition:
    context_prior: float
    rows: tuple[ContributionRow, ...]
    logit: float  # context_prior + Σ rows.logit (= logit s)


def effective_coefficients(space: FeatureSpace, coef: np.ndarray, ctx: Context) -> list[tuple[float, float]]:
    """(coefficient on z⁺, coefficient on z⁻) per detector in this context, before the Platt slope."""
    terms = ctx.terms()
    g = [terms[name] for name in space.groups]
    n = len(space.groups)
    out = []
    for j in range(len(space.detectors)):
        off = space.offset(j)
        plus = coef[off] + sum(coef[off + 2 + gi] * g[gi] for gi in range(n))
        minus = coef[off + 1] + sum(coef[off + 2 + n + gi] * g[gi] for gi in range(n))
        out.append((float(plus), float(minus)))
    return out


def decompose(space: FeatureSpace, coef: np.ndarray, z: Sequence[float | None], ctx: Context, a: float, b: float, *,
              prior_logit: float = 0.0, inactive: Collection[str] = ()) -> Decomposition:
    x = space.row(z, ctx, inactive)
    terms = ctx.terms()
    n_ctx = len(space.context_terms)
    context_prior = a * float(np.dot(coef[: 1 + n_ctx], x[: 1 + n_ctx])) + b + prior_logit
    parts: list[tuple[str, str, float | None, float, float | None]] = []
    for j, (det, (plus, minus)) in enumerate(zip(space.detectors, effective_coefficients(space, coef, ctx), strict=True)):
        off = space.offset(j)
        if not space.gate_open(j, terms) or det in inactive:
            parts.append(("detector", det, None, 0.0, None))
            continue
        if x[off + space.block - 1] == 1.0:
            parts.append(("detector", det, None, 0.0, None))
            parts.append(("missing", det, None, a * float(coef[off + space.block - 1]), None))
            continue
        zp, zm = float(x[off]), float(x[off + 1])
        parts.append(("detector", det, zp + zm, a * (plus * zp + minus * zm), a * (plus if zp + zm >= 0 else minus)))
    total = math.fsum([context_prior, *(p[3] for p in parts)])
    rows = tuple(ContributionRow(kind=kind, detector=det, z=zval, logit=phi,
                                 pp=float(expit(total) - expit(total - phi)), effective_weight=weight)
                 for kind, det, zval, phi, weight in parts)
    return Decomposition(context_prior=context_prior, rows=rows, logit=total)


@dataclass(frozen=True)
class NeutralRow:
    kind: str  # "detector" | "missing"
    detector: str
    z: float | None
    neutral_z: float | None  # the z the row is measured from; None for a missing row and for a check that did not score
    standalone: bool  # False: no usable stand-alone calibration, the row keeps the 50 % baseline (neutral_z 0)
    logit: float
    pp: float
    effective_weight: float | None


@dataclass(frozen=True)
class NeutralView:
    start: float  # logit of the estimate with every scoring check at its neutral point, in this context
    rows: tuple[NeutralRow, ...]
    logit: float  # the decomposition's logit (= logit s); start + Σ rows.logit equals it up to rounding


def neutral_point(entry: Mapping[str, object] | None) -> float | None:
    """The z at which a check's stand-alone likelihood ratio is 1, from its univariate Platt; None when it has none inside
    the range a score can take."""
    if entry is None:
        return None
    a, b = float(entry["a"]), float(entry["b"])  # type: ignore[arg-type]
    if not (math.isfinite(a) and math.isfinite(b)) or a == 0.0:
        return None
    z0 = -b / a
    return z0 if abs(z0) <= Z_CLIP else None


def neutral_view(space: FeatureSpace, coef: np.ndarray, z: Sequence[float | None], ctx: Context, a: float, b: float,
                 signal_calibration: Mapping[str, Mapping[str, object] | None], *, prior_logit: float = 0.0,
                 inactive: Collection[str] = ()) -> NeutralView:
    """The decomposition of `decompose` for the same input, measured from each scoring check's neutral point (D-63)."""
    actual = decompose(space, coef, z, ctx, a, b, prior_logit=prior_logit, inactive=inactive)
    scoring = {r.detector for r in actual.rows if r.kind == "detector" and r.z is not None}
    points = {det: neutral_point(signal_calibration.get(det)) for det in space.detectors if det in scoring}
    at_neutral = [(points[det] if points[det] is not None else 0.0) if det in scoring else value
                  for det, value in zip(space.detectors, z, strict=True)]
    base = decompose(space, coef, at_neutral, ctx, a, b, prior_logit=prior_logit, inactive=inactive)
    base_logit = {r.detector: r.logit for r in base.rows if r.kind == "detector" and r.z is not None}
    if set(base_logit) != scoring:  # a neutral point is a finite score, so the same checks score in both
        raise ValueError("the neutral input does not score the same checks")
    start = math.fsum([base.context_prior, *base_logit.values()])
    rows = []
    for r in actual.rows:
        if r.kind == "detector" and r.z is not None:
            point = points[r.detector]
            phi = r.logit - base_logit[r.detector]
            rows.append(NeutralRow(kind="detector", detector=r.detector, z=r.z, neutral_z=0.0 if point is None else point,
                                   standalone=point is not None, logit=phi,
                                   pp=float(expit(actual.logit) - expit(actual.logit - phi)), effective_weight=r.effective_weight))
        else:
            rows.append(NeutralRow(kind=r.kind, detector=r.detector, z=None, neutral_z=None, standalone=False, logit=r.logit,
                                   pp=r.pp, effective_weight=r.effective_weight))
    return NeutralView(start=start, rows=tuple(rows), logit=actual.logit)
