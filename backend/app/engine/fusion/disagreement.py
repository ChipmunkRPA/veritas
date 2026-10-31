"""Per-detector calibration and model disagreement (ML §5.4).

- Each detector gets a univariate Platt on `cal` over the items where it ran: z̃_j = a_j·z_j + b_j (calibrated logit).
  This is also the detector's stand-alone probability for display (M-2).
- D = √( Σ_j ω_j (z̃_j − z̄)² / Σ_j ω_j ) over detectors that ran, with ω_j = |effective weight| and z̄ the ω-weighted
  mean. D is undefined (None) when fewer than two detectors with ω > 0 ran.
- d95 = the 95th percentile of D over `cal` items where D is defined (numpy's "higher" method, an observed value).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.engine.fusion.calibration import fit_platt


def fit_signal_calibration(z: np.ndarray, y, detectors: Sequence[str]) -> dict[str, dict | None]:
    """Univariate Platt per detector; None when the detector ran on items of only one class."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y)
    if z.shape != (len(y), len(detectors)):
        raise ValueError("z must be n × J")
    out: dict[str, dict | None] = {}
    for j, det in enumerate(detectors):
        ran = np.isfinite(z[:, j])
        n1 = int((y[ran] == 1).sum())
        n0 = int(ran.sum()) - n1
        if n0 == 0 or n1 == 0:
            out[det] = None
            continue
        a, b = fit_platt(z[ran, j], y[ran])
        out[det] = {"method": "platt", "a": a, "b": b, "n0": n0, "n1": n1}
    return out


def calibrated_logit(entry: dict | None, z: float | None) -> float | None:
    if entry is None or z is None or not math.isfinite(z):
        return None
    return float(entry["a"]) * z + float(entry["b"])


def disagreement_index(z_tilde: Sequence[float | None], weights: Sequence[float | None]) -> float | None:
    pairs = [(float(v), abs(float(w))) for v, w in zip(z_tilde, weights, strict=True)
             if v is not None and w is not None and abs(w) > 0]
    if len(pairs) < 2:
        return None
    total = math.fsum(w for _, w in pairs)
    mean = math.fsum(v * w for v, w in pairs) / total
    return math.sqrt(math.fsum(w * (v - mean) ** 2 for v, w in pairs) / total)


def d95(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return float(np.quantile(np.asarray(defined, dtype=np.float64), 0.95, method="higher"))
