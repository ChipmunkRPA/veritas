"""Multi-generator/voice-balanced frozen-feature learner; research only, not serving."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.audio_calibration_v3_methods import clipped_logit
from scripts.audio_guard_v7 import initial
from scripts.audio_residual_head_v6 import design, predict
from scripts.review_audio_noise_v4 import valid

METHOD = {
    "version": "audio-multisource-voice-balanced/8.0",
    "backbone": "Existing commercial-permission speech backbone; every pretrained parameter frozen",
    "components": 16,
    "weights": "Equal truth classes, generator families within class, connected groups within class/family, sources within group/class/family and assessed variants within source",
    "projection": "Fitting-only head-orthogonal weighted PCA16, standard deviation floor1e-6",
    "predictors": "Original clipped raw logit,16 residual coordinates and intercept; no metadata predictors",
    "original_ridge": 0.001,
    "residual_ridge": 0.05,
    "original_slope": "nonnegative",
    "target_smoothing": "Connected truth-class fitting group counts, not variant/source counts",
    "reference_prior": 0.5,
    "guard": "Independent-calibration worst nonconformity per connected group AND truth class; paired genuine/cloned voices may occur in both class pools",
    "original_stability": "Both candidate calibrated logit and original raw logit changes must be at most ln2, six-decimal probabilities",
    "new_pretrained_checkpoint": False,
    "deployment_eligible": False,
}


def balanced_weights(rows):
    if not rows:
        raise ValueError("Both truth classes require assessed fitting rows")
    if any(r["label"] not in (0, 1) or isinstance(r["label"], bool) for r in rows):
        raise ValueError("Integer truth labels required")
    variants = Counter((r["label"], r["generator"], r["group"], r["base_id"]) for r in rows)
    sources, groups, families = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        label, family, group = row["label"], row["generator"], row["group"]
        sources[label, family, group].add(row["base_id"])
        groups[label, family].add(group)
        families[label].add(family)
    if not families[0] or not families[1]:
        raise ValueError("Both truth classes require assessed fitting rows")
    result = []
    for row in rows:
        label, family, group, source = row["label"], row["generator"], row["group"], row["base_id"]
        result.append(
            0.5
            / len(families[label])
            / len(groups[label, family])
            / len(sources[label, family, group])
            / variants[label, family, group, source]
        )
    w = np.asarray(result, np.float64)
    if not np.isfinite(w).all() or not math.isclose(float(w.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("Invalid source/class/family weighting")
    return w


def fit(rows, head_direction):
    if not rows or any(row["role"] != "fit" for row in rows):
        raise ValueError("Only fitting roles may learn parameters")
    selected = [r for r in rows if valid(r.get("raw"))]
    if any(r.get("embedding") is None for r in selected):
        raise ValueError("An assessed source requires its original representation")
    w = balanced_weights(selected)
    e = np.asarray([r["embedding"] for r in selected], np.float64)
    h = np.asarray(head_direction, np.float64)
    if (
        e.ndim != 2
        or h.shape != (e.shape[1],)
        or not np.isfinite(e).all()
        or not np.isfinite(h).all()
        or np.linalg.norm(h) <= 0
    ):
        raise ValueError("Finite fitting representations/head required")
    h = h / np.linalg.norm(h)
    e = e - np.outer(e @ h, h)
    center = w @ e
    _, singular, directions = np.linalg.svd((e - center) * np.sqrt(w[:, None]), full_matrices=False)
    if len(singular) < 16 or singular[15] <= 1e-8:
        raise ValueError("Insufficient fitting rank for the fixed PCA16")
    transform = {
        "unit_head": h.tolist(),
        "center": center.tolist(),
        "directions": directions[:16].tolist(),
        "scale": np.maximum(singular[:16], 1e-6).tolist(),
        "fit_sources": len({r["base_id"] for r in selected}),
    }
    x = np.asarray([design(row, transform) for row in selected])
    y = np.asarray([r["label"] for r in selected])
    counts = [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)]
    if min(counts) < 2:
        raise ValueError("At least two fitting groups in each truth class required")
    target = np.where(y == 1, (counts[1] + 1) / (counts[1] + 2), 1 / (counts[0] + 2))
    ridge = np.full(18, 0.05)
    ridge[0], ridge[-1] = 0.001, 0

    def objective(beta):
        z = x @ beta
        return (
            float(w @ (np.logaddexp(0, z) - target * z) + 0.5 * np.dot(ridge, beta**2)),
            x.T @ (w * (expit(z) - target)) + ridge * beta,
        )

    start = np.zeros(18)
    start[0] = 1
    solved = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] + [(None, None)] * 17,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("Fixed multi-source objective did not converge")
    return {
        "transform": transform,
        "coefficients": solved.x.tolist(),
        "fit_originals_by_class": [len({r["base_id"] for r in selected if r["label"] == c}) for c in (0, 1)],
        "fit_groups_by_class": counts,
        "reference_prior": 0.5,
        "method": METHOD,
    }


def calibrate(rows, fitted):
    if not rows or any(r["role"] != "cal" for r in rows):
        raise ValueError("Only independent calibration roles may set guards")
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        if row["label"] not in (0, 1) or isinstance(row["label"], bool):
            raise ValueError("Integer truth labels required")
        p = predict(row, fitted)
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    result = {
        str(c): [
            {
                "group": group,
                "maximum": None if any(p is None for p in values) else max(values),
                "rows": len(values),
            }
            for group, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }
    if min(len(result[str(c)]) for c in (0, 1)) < 9:
        raise ValueError("At least nine independent calibration groups per truth class required")
    return result


def final(p, band_p, raw, band_raw, guards):
    proposed = initial(p, guards)
    delta_p, delta_raw = None, None
    if proposed == "original":
        if valid(p) and valid(band_p):
            delta_p = abs(clipped_logit(round(p, 6)) - clipped_logit(round(band_p, 6)))
        if valid(raw) and valid(band_raw):
            delta_raw = abs(clipped_logit(round(raw, 6)) - clipped_logit(round(band_raw, 6)))
    veto = proposed == "original" and (
        delta_p is None or delta_raw is None or max(delta_p, delta_raw) > math.log(2)
    )
    return {
        "proposed": proposed,
        "decision": "inconclusive" if veto else proposed,
        "calibrated_stability_log_odds_change": delta_p,
        "raw_stability_log_odds_change": delta_raw,
        "original_stability_veto": veto,
    }
