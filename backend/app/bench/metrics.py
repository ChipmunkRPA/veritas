"""Benchmark metrics (ML §2.8, §5.1, §5.3). Pure numpy; s = fused probability, y = 1 for AI or manipulated.

- **AUC.** Mann–Whitney U / (n₁n₀) with average ranks for ties.
- **Operational threshold τ_α** from `cal` reals: the ⌈(n₀+1)(1−α)⌉-th smallest s. Pooled over real strata with equal
  stratum weights: each real in stratum k weighs 1/(K·n_k), and τ_α is the smallest s whose weighted cumulative share
  reaches ⌈(n₀+1)(1−α)⌉/n₀. With equally sized strata (or one stratum) this is exactly the rank rule. When the rank
  exceeds n₀, τ_α is +∞ (None): nothing is flagged.
- **Small strata** (ADR 0007, H-8): a stratum with fewer reals than the rank rule needs at α = 0.05 (19) cannot carry an
  equal weight, since one of its items would outweigh the α mass. Such strata are pooled into one group, and a group still
  below 19 joins the largest stratum (`effective_strata`); with one group left this is the plain rank rule.
- **Rates at τ.** An item is flagged when s ≥ τ. TPR = flagged fakes / fakes, FPR = flagged reals / reals, each with a
  Wilson 95 % interval and, with zero events, the rule-of-three bound 3/n.
- **ECE.** 15 bins of equal class-balanced mass (each class carries half of the total weight), ECE = Σ_b (W_b/W)·|ȳ_b − s̄_b|
  with weighted bin means; MCE = the largest bin gap. Brier and log-loss are class-balanced means.
- **EER.** The ROC point where FPR = FNR, linearly interpolated between adjacent thresholds.
- **Robustness deltas** at a fixed τ: ΔTPR = TPR_clean − TPR_t, ΔFPR = FPR_t − FPR_clean; ΔAUC = AUC_clean − AUC_t.
- **Grouped bootstrap.** Resamples whole `group_key`s with replacement (B = 2,000, percentile 95 % interval). Replicate
  indices come from one seeded generator, so every statistic on the same `GroupedBootstrap` sees the same resamples and
  differences between systems are paired (§5.3).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import rankdata

Z95 = 1.959964
ECE_BINS = 15
BOOTSTRAP_B = 2000
MIN_STRATUM_N = 19  # the rank rule's minimum n₀ at α = 0.05 (ADR 0007)
SMALL_STRATA = "<small strata>"
_EPS = 1e-12


def _xy(s, y) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError("scores and labels must be 1-D arrays of equal length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if len(y) and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (real) or 1 (AI or manipulated)")
    return s, y.astype(np.int64)


def auc(s, y) -> float | None:
    s, y = _xy(s, y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def merge_map(cal_strata: Sequence[str], strata: Sequence[str] = (), min_n: int = MIN_STRATUM_N) -> dict[str, str]:
    """{stratum: weighting stratum} from the reals of `cal_strata` (`effective_strata`'s rule). A stratum of `strata` without
    `cal_strata` reals counts as below `min_n` (D-51: the ablation cross-fit relabels its out-of-fold reals with this map)."""
    counts = Counter(str(k) for k in cal_strata)
    for k in strata:
        counts.setdefault(str(k), 0)
    small = {k for k, n in counts.items() if n < min_n}
    if not small or len(counts) == 1:
        return {k: k for k in counts}
    large = {k: n for k, n in counts.items() if k not in small}
    joins_largest = large and sum(counts[k] for k in small) < min_n
    target = min(large, key=lambda k: (-large[k], k)) if joins_largest else SMALL_STRATA
    return {k: target if k in small else k for k in counts}


def effective_strata(strata: Sequence[str], min_n: int = MIN_STRATUM_N) -> list[str]:
    """The weighting strata of τ_α: strata below `min_n` reals pooled into `SMALL_STRATA`, which joins the largest stratum
    (ties: the first in sorted order) while it holds fewer than `min_n` reals itself."""
    labels = [str(k) for k in strata]
    mapping = merge_map(labels, min_n=min_n)
    return [mapping[k] for k in labels]


def operational_threshold(s_real, alpha: float, strata: Sequence[str] | None = None) -> float | None:
    """τ_α among calibration reals (see the module docstring); None means +∞."""
    s = np.asarray(s_real, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("τ_α needs at least one finite real score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("α must lie strictly between 0 and 1")
    labels = None if strata is None else np.asarray(list(strata), dtype=object)
    if labels is not None and labels.shape != s.shape:
        raise ValueError("strata and scores differ in length")
    n0 = len(s)
    rank = math.ceil((n0 + 1) * (1 - Fraction(str(alpha))))
    if rank > n0:
        return None
    if labels is None:
        return float(np.sort(s)[rank - 1])
    weights = stratum_weights(labels.tolist())
    order = np.argsort(s, kind="mergesort")
    values, cumulative = s[order], np.cumsum(weights[order])
    target = rank / n0
    # the weighted share at v counts every real with s ≤ v, so ties are taken together
    last_of_tie = np.r_[values[1:] != values[:-1], True]
    hits = np.flatnonzero(last_of_tie & (cumulative >= target - _EPS))
    return float(values[hits[0]]) if len(hits) else float(values[-1])


def stratum_weights(strata: Sequence[str] | None, n: int | None = None) -> np.ndarray:
    """Equal stratum weights of τ_α: each real in weighting stratum k (`effective_strata`) weighs 1/(K·n_k); without strata
    every one of the `n` reals weighs 1/n. The weights sum to 1."""
    if strata is None:
        if n is None or n < 1:
            raise ValueError("unweighted stratum weights need n ≥ 1")
        return np.full(n, 1.0 / n)
    keys, inverse, counts = np.unique(np.asarray(effective_strata([str(k) for k in strata]), dtype=str), return_inverse=True,
                                      return_counts=True)
    return 1.0 / (len(keys) * counts[inverse])


@dataclass(frozen=True)
class OperatingPoint:
    """τ_α and q, the weight of reals scoring exactly τ (D-53, admission harness only). `tied_real_share` is the weighted
    share of reals at τ; `served_rule_weighted_fpr` the weighted share with s ≥ τ (the served rule)."""
    tau: float | None
    q: float
    tied_real_share: float
    served_rule_weighted_fpr: float

    def as_json(self) -> dict:
        return {"tau": self.tau, "q": self.q, "tied_real_share": self.tied_real_share,
                "served_rule_weighted_fpr": self.served_rule_weighted_fpr}


def operating_point(s_real, alpha: float, strata: Sequence[str] | None = None) -> OperatingPoint:
    """τ as `operational_threshold` gives it, plus q ∈ [0, 1] such that flagging reals above τ with weight 1 and reals at τ
    with weight q gives the weighted flagged share 1 − rank/n₀ (the rank rule's share; D-53). None τ: q 0, nothing flagged."""
    tau = operational_threshold(s_real, alpha, strata)
    s = np.asarray(s_real, dtype=np.float64)
    if tau is None:
        return OperatingPoint(None, 0.0, 0.0, 0.0)
    n0 = len(s)
    target = 1.0 - math.ceil((n0 + 1) * (1 - Fraction(str(alpha)))) / n0
    weights = stratum_weights(strata, n0)
    above, tied = float(weights[s > tau].sum()), float(weights[s == tau].sum())
    q = min(1.0, max(0.0, (target - above) / tied)) if tied > 0 else 0.0
    return OperatingPoint(tau, q, tied, above + tied)


def flag_weights(s, point: OperatingPoint) -> np.ndarray:
    """Per-item flag weight at an operating point: 1 above τ, q at τ, 0 below; all zero when τ is None (D-53)."""
    s = np.asarray(s, dtype=np.float64)
    if point.tau is None:
        return np.zeros(s.shape)
    return np.where(s > point.tau, 1.0, np.where(s == point.tau, point.q, 0.0))


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float | None:
        return self.k / self.n if self.n else None

    def wilson(self, z: float = Z95) -> tuple[float, float] | None:
        return wilson(self.k, self.n, z)

    def as_json(self) -> dict:
        ci = self.wilson()
        return {"k": self.k, "n": self.n, "value": self.value, "wilson95": list(ci) if ci else None,
                "rule_of_three": 3.0 / self.n if self.n and self.k == 0 else None}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval: centre (p̂ + z²/2n)/(1 + z²/n), half-width z·√(p̂(1−p̂)/n + z²/4n²)/(1 + z²/n)."""
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("k must lie between 0 and n")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def flagged(s, tau: float | None) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return np.zeros(s.shape, dtype=bool) if tau is None else s >= tau


def tpr_at(s, y, tau: float | None) -> Rate:
    s, y = _xy(s, y)
    return Rate(int(flagged(s[y == 1], tau).sum()), int((y == 1).sum()))


def fpr_at(s, y, tau: float | None) -> Rate:
    s, y = _xy(s, y)
    return Rate(int(flagged(s[y == 0], tau).sum()), int((y == 0).sum()))


def balanced_accuracy(s, y, tau: float | None) -> float | None:
    tpr, fpr = tpr_at(s, y, tau), fpr_at(s, y, tau)
    if tpr.value is None or fpr.value is None:
        return None
    return (tpr.value + 1.0 - fpr.value) / 2.0


def balance_weights(y) -> np.ndarray:
    """Per-item weights that give each present class half of the total weight n."""
    y = np.asarray(y)
    n, n1 = len(y), int((y == 1).sum())
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return np.ones(n)
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


def ece(s, y, *, bins: int = ECE_BINS, balance: bool = True) -> dict | None:
    """ECE, MCE and the reliability table on equal-mass bins (weighted mass when balanced)."""
    s, y = _xy(s, y)
    if len(s) == 0:
        return None
    if bins < 1:
        raise ValueError("ECE needs at least one bin")
    w = balance_weights(y) if balance else np.ones(len(s))
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    total = float(w.sum())
    midpoints = np.cumsum(w) - w / 2.0
    index = np.minimum((midpoints / total * bins).astype(np.int64), bins - 1)
    table, gaps, value = [], [], 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        wb = float(w[mask].sum())
        mean_s, mean_y = float(np.dot(w[mask], s[mask]) / wb), float(np.dot(w[mask], y[mask]) / wb)
        gap = abs(mean_y - mean_s)
        value += wb / total * gap
        gaps.append(gap)
        table.append({"bin": b, "n": int(mask.sum()), "weight": wb / total, "mean_s": mean_s, "mean_y": mean_y,
                      "s_min": float(s[mask].min()), "s_max": float(s[mask].max())})
    return {"ece": value, "mce": max(gaps), "bins": bins, "balanced": balance, "table": table}


def brier(s, y, *, balance: bool = True) -> float | None:
    s, y = _xy(s, y)
    if len(s) == 0:
        return None
    w = balance_weights(y) if balance else np.ones(len(s))
    return float(np.dot(w, (s - y) ** 2) / w.sum())


def log_loss(s, y, *, balance: bool = True) -> float | None:
    s, y = _xy(s, y)
    if len(s) == 0:
        return None
    q = np.clip(s, _EPS, 1 - _EPS)
    loss = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    w = balance_weights(y) if balance else np.ones(len(s))
    return float(np.dot(w, loss) / w.sum())


def eer(s, y) -> float | None:
    """Equal error rate from the ROC over every distinct threshold (flag s ≥ t)."""
    s, y = _xy(s, y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return None
    thresholds = np.r_[np.unique(s), np.inf]
    fpr = np.array([(s[y == 0] >= t).mean() for t in thresholds])
    fnr = np.array([(s[y == 1] < t).mean() for t in thresholds])
    diff = fpr - fnr  # non-increasing in t: from +(1 − 0) down to −(0 − 1)
    cross = np.flatnonzero(diff <= 0)[0]
    if diff[cross] == 0 or cross == 0:
        return float(fpr[cross])
    a, b = diff[cross - 1], diff[cross]
    frac = a / (a - b)
    return float(fpr[cross - 1] + frac * (fpr[cross] - fpr[cross - 1]))


def robustness_deltas(clean_s, clean_y, pert_s, pert_y, tau: float | None) -> dict:
    """Fixed-threshold ΔTPR / ΔFPR (what production experiences) and the re-thresholded ΔAUC (ranking damage)."""
    t_clean, t_pert = tpr_at(clean_s, clean_y, tau), tpr_at(pert_s, pert_y, tau)
    f_clean, f_pert = fpr_at(clean_s, clean_y, tau), fpr_at(pert_s, pert_y, tau)
    a_clean, a_pert = auc(clean_s, clean_y), auc(pert_s, pert_y)

    def diff(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    return {"tpr_clean": t_clean.as_json(), "tpr_t": t_pert.as_json(), "fpr_clean": f_clean.as_json(),
            "fpr_t": f_pert.as_json(), "auc_clean": a_clean, "auc_t": a_pert,
            "delta_tpr": diff(t_clean.value, t_pert.value), "delta_fpr": diff(f_pert.value, f_clean.value),
            "delta_auc": diff(a_clean, a_pert)}


def selective(decided: Sequence[int | None], y) -> dict:
    """decided: 1 (AI), 0 (non-AI) or None (inconclusive) per item. Inconclusive rate and the error among decided items."""
    y = np.asarray(y)
    if len(decided) != len(y):
        raise ValueError("decisions and labels differ in length")
    made = [(d, int(t)) for d, t in zip(decided, y, strict=True) if d is not None]
    errors = sum(d != t for d, t in made)
    return {"n": len(y), "inconclusive": Rate(len(y) - len(made), len(y)).as_json(),
            "selective_risk": Rate(errors, len(made)).as_json()}


def class_coverage(contains_truth: Sequence[bool], y) -> dict:
    """Class-conditional coverage P(y ∈ C | y) per class."""
    hits = np.asarray(contains_truth, dtype=bool)
    y = np.asarray(y)
    if hits.shape != y.shape:
        raise ValueError("coverage flags and labels differ in length")
    return {name: Rate(int(hits[y == label].sum()), int((y == label).sum())).as_json()
            for name, label in (("real", 0), ("ai", 1))}


class GroupedBootstrap:
    """Percentile intervals over resampled groups; the same seed replays the same resamples for every statistic."""

    def __init__(self, groups: Sequence[str], *, n_boot: int = BOOTSTRAP_B, seed: int = 7):
        if n_boot < 1:
            raise ValueError("the bootstrap needs B ≥ 1")
        keys, inverse = np.unique(np.asarray(list(groups), dtype=str), return_inverse=True)
        if len(keys) == 0:
            raise ValueError("the bootstrap needs at least one group")
        order = np.argsort(inverse, kind="mergesort")
        bounds = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(keys)))]
        self._members = [order[bounds[i]:bounds[i + 1]] for i in range(len(keys))]
        self.n_groups, self.n_boot, self.seed = len(keys), n_boot, seed

    def indices(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_boot):
            draw = rng.integers(0, self.n_groups, size=self.n_groups)
            yield np.concatenate([self._members[i] for i in draw])

    def values(self, stat: Callable[[np.ndarray], float | None]) -> np.ndarray:
        out = [stat(idx) for idx in self.indices()]
        return np.asarray([v for v in out if v is not None and math.isfinite(v)], dtype=np.float64)

    def ci(self, stat: Callable[[np.ndarray], float | None], level: float = 0.95) -> dict:
        return _interval(self.values(stat), self.n_boot, level)

    def paired(self, stat_a: Callable[[np.ndarray], float | None], stat_b: Callable[[np.ndarray], float | None],
               level: float = 0.95) -> dict:
        """Interval of stat_a − stat_b on the same resamples."""
        diffs = []
        for idx in self.indices():
            a, b = stat_a(idx), stat_b(idx)
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                diffs.append(a - b)
        return _interval(np.asarray(diffs, dtype=np.float64), self.n_boot, level)


def _interval(values: np.ndarray, n_boot: int, level: float) -> dict:
    if not 0.0 < level < 1.0:
        raise ValueError("the interval level must lie strictly between 0 and 1")
    if len(values) == 0:
        return {"lo": None, "hi": None, "B": n_boot, "valid": 0, "level": level}
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return {"lo": float(lo), "hi": float(hi), "B": n_boot, "valid": int(len(values)), "level": level}
