"""Separate source-balanced mapping to proposed-interval retained-AI fraction."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.writing_span_geometry_v7 import require


def logit(raw):
    return np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))


def objective(theta, x, y, weights):
    eta = theta[0] + theta[1] * x
    residual = weights * (expit(eta) - y)
    loss = float(np.dot(weights, np.logaddexp(0, eta) - y * eta) + 0.0005 * theta[1] ** 2)
    return loss, np.asarray([residual.sum(), np.dot(residual, x) + 0.001 * theta[1]])


def fit(rows, cases):
    require(
        rows and len(rows) == len(cases) == 288 and all(c["role"] == "cal" for c in cases),
        "Only all288 separate calibration-role cases may fit the mapping",
    )
    require(len({c["group"] for c in cases}) == 32, "All32 calibration groups required")
    x, y, weights = [], [], []
    for row, case in zip(rows, cases, strict=True):
        require(
            row["id"] == case["id"] and row["status"] == "ok", "Exact complete calibration response required"
        )
        values = []
        for p in row["proposals"]:
            a, b = p["start"], p["end"]
            require(
                0 <= a < b <= len(case["text"]) and math.isfinite(p["raw"]) and 0 <= p["raw"] <= 1,
                "Valid source-bound interval response required",
            )
            target = [case["labels"][i] for i in range(a, b) if not case["text"][i].isspace()]
            if target:
                require(all(v in (0, 1) for v in target), "Unknown local calibration provenance")
                values.append((p["raw"], sum(target) / len(target)))
        require(values, "No usable proposed interval in a calibration document")
        for score, target in values:
            x.append(float(logit(score)))
            y.append(target)
            weights.append(1 / (288 * len(values)))
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    require(
        np.isfinite(x).all() and np.isfinite(y).all() and 0 < np.dot(weights, y) < 1,
        "Both local-source classes required for calibration",
    )
    weights /= weights.sum()
    mean, scale = float(np.dot(weights, x)), float(np.sqrt(np.dot(weights, (x - np.dot(weights, x)) ** 2)))
    scale = max(scale, 1e-8)
    z = (x - mean) / scale
    initial = [float(logit(float(np.dot(weights, y)))), 0.0]
    fitted = minimize(
        objective,
        initial,
        args=(z, y, weights),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0, None)],
        options={"maxiter": 500, "gtol": 1e-9, "ftol": 2.22e-16, "maxls": 50},
    )
    loss, gradient = objective(fitted.x, z, y, weights)
    projected = [
        abs(float(gradient[0])),
        abs(float(gradient[1])) if fitted.x[1] > 1e-10 else max(0.0, -float(gradient[1])),
    ]
    require(
        fitted.success and max(projected) <= 1e-6, "Calibration solver did not meet fixed convergence checks"
    )
    return {
        "intercept": float(fitted.x[0]),
        "slope": float(fitted.x[1]),
        "mean": mean,
        "scale": scale,
        "loss": loss,
        "projected_gradient": max(projected),
        "source_groups": 32,
        "documents": 288,
        "target": "Retained publisher-AI non-whitespace character fraction inside this proposed interval",
        "customer_confidence": False,
    }


def predict(raw, fitted):
    require(
        type(raw) in (int, float) and math.isfinite(raw) and 0 <= raw <= 1, "Finite raw response required"
    )
    return float(
        expit(fitted["intercept"] + fitted["slope"] * (float(logit(raw)) - fitted["mean"]) / fitted["scale"])
    )
