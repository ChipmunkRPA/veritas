"""Stage 4 class-conditional split-conformal sets and bands (ML §4.5).

- **Thresholds** on `cal`, per context k when n_{k,y} ≥ 100, otherwise the global value, separately per class:
  - reals: nonconformity s; q_real(α) = the ⌈(n₀+1)(1−α)⌉-th smallest s among reals;
  - fakes: nonconformity 1 − s; q_ai(α) = the ⌈(n₁+1)(1−α)⌉-th smallest (1 − s) among fakes;
  - when ⌈(n+1)(1−α)⌉ > n the quantile is +∞, stored as null: that class is never excluded.
- **Set** C_α(x) = {real if s ≤ q_real(α)} ∪ {ai if 1 − s ≤ q_ai(α)}. Under exchangeability
  P(real ∈ C | real) ≥ 1 − α and P(ai ∈ C | ai) ≥ 1 − α.
- **Bands**, first matching rule wins: very_likely_ai (C_0.01 = C_0.05 = {ai}), very_likely_authentic (C_0.01 = C_0.05 =
  {real}), likely_ai (C_0.05 = {ai}), likely_authentic (C_0.05 = {real}), otherwise uncertain (C_0.05 = both, or ∅ = unlike
  both calibration populations). A strong band needs the moderate set to be the same singleton (D-49): with too few fakes for
  an α = 0.01 AI threshold, C_0.01 can be {ai} while C_0.05 is ∅, and the moderate rules then decide.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from app.engine.fusion.calibration import MIN_CLASS_N
from app.engine.fusion.features import PLATT_CONTEXTS

ALPHAS = (0.05, 0.01)


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def conformal_quantile(nonconformity, alpha: float) -> float | None:
    """The ⌈(n+1)(1−α)⌉-th smallest value (exact rational arithmetic), or None for +∞."""
    values = np.sort(np.asarray(nonconformity, dtype=np.float64))
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("conformal quantile needs at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("α must lie strictly between 0 and 1")
    k = math.ceil((len(values) + 1) * (1 - Fraction(str(alpha))))
    return None if k > len(values) else float(values[k - 1])


def fit_conformal(s, y, contexts: Sequence[str], *, alphas: Sequence[float] = ALPHAS,
                  min_class_n: int = MIN_CLASS_N) -> dict:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    contexts = np.asarray(list(contexts), dtype=object)
    if not (s.shape == y.shape == contexts.shape):
        raise ValueError("scores, labels and contexts differ in length")
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("conformal calibration needs both classes")
    doc: dict = {"q_real": {}, "q_ai": {}, "n_real": {}, "n_ai": {}, "min_class_n": min_class_n,
                 "alphas": [alpha_key(a) for a in alphas]}
    scopes = [("global", np.ones(len(s), dtype=bool))] + [(k, contexts == k) for k in PLATT_CONTEXTS]
    for scope, mask in scopes:
        reals, fakes = s[mask & (y == 0)], s[mask & (y == 1)]
        if scope == "global" or len(reals) >= min_class_n:
            doc["q_real"][scope] = {alpha_key(a): conformal_quantile(reals, a) for a in alphas}
            doc["n_real"][scope] = len(reals)
        if scope == "global" or len(fakes) >= min_class_n:
            doc["q_ai"][scope] = {alpha_key(a): conformal_quantile(1.0 - fakes, a) for a in alphas}
            doc["n_ai"][scope] = len(fakes)
    return doc


def thresholds(doc: dict, context: str, alpha: float) -> tuple[float | None, float | None]:
    key = alpha_key(alpha)
    q_real = doc["q_real"].get(context, doc["q_real"]["global"])[key]
    q_ai = doc["q_ai"].get(context, doc["q_ai"]["global"])[key]
    return q_real, q_ai


def prediction_set(s: float, q_real: float | None, q_ai: float | None) -> frozenset[str]:
    out = set()
    if q_real is None or s <= q_real:
        out.add("real")
    if q_ai is None or 1.0 - s <= q_ai:
        out.add("ai")
    return frozenset(out)


def calibrated(doc: dict, alpha: float) -> bool:
    """The artefact holds global thresholds of both classes at this α."""
    key = alpha_key(alpha)
    return key in (doc.get("q_real") or {}).get("global", {}) and key in (doc.get("q_ai") or {}).get("global", {})


def band(s: float, doc: dict, context: str, *, alpha_strong: float = 0.01,
         alpha_moderate: float = 0.05) -> tuple[str, dict[str, list[str]]]:
    """(band key, the prediction sets at α = 0.05, 0.01 and the policy levels). The strong bands need a singleton set at
    `alpha_strong`, the moderate bands at `alpha_moderate` (workspace policy thresholds, API-03; defaults ML §4.5)."""
    sets = {alpha_key(a): sorted(prediction_set(s, *thresholds(doc, context, a)))
            for a in dict.fromkeys((*ALPHAS, alpha_moderate, alpha_strong))}
    strong, moderate = set(sets[alpha_key(alpha_strong)]), set(sets[alpha_key(alpha_moderate)])
    if strong == moderate == {"ai"}:
        return "very_likely_ai", sets
    if strong == moderate == {"real"}:
        return "very_likely_authentic", sets
    if moderate == {"ai"}:
        return "likely_ai", sets
    if moderate == {"real"}:
        return "likely_authentic", sets
    return "uncertain", sets


def coverage(s, y, doc: dict, contexts: Sequence[str], alpha: float) -> dict[str, float | None]:
    """Class-conditional coverage P(y ∈ C_α | y) on a labelled set."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    hits = {0: [], 1: []}
    for value, label, context in zip(s, y, contexts, strict=True):
        members = prediction_set(float(value), *thresholds(doc, context, alpha))
        hits[int(label)].append(("ai" if label == 1 else "real") in members)
    return {name: (float(np.mean(hits[label])) if hits[label] else None) for name, label in (("real", 0), ("ai", 1))}
