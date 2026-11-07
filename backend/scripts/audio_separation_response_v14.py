"""Weakly supervised any-source classifier for two unordered voice estimates.

An AI mixture label is never copied onto both estimated sources. A shared
source response is learned through a noisy-OR bag objective; all genuine bags
penalize either source becoming falsely positive. Independent calibration is
still required, including separator artifacts on single-voice/noise/music cases.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from scripts.audio_calibration_v3_methods import clipped_logit
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_separation_v14 import SPEC
from scripts.review_audio_conditional_v10 import guards
from scripts.review_audio_noise_v4 import valid

ARMS = ("separated", "complement")
DIMENSIONS = 1024
SOURCE_WIDTH = DIMENSIONS + 2
METHOD = {
    "version": "audio-two-source-multiple-instance/14.0", "separator": SPEC,
    "features": "Both unordered complete source estimates, original frozen1024-dimensional voice embedding and raw response per estimate; original mixture raw score retained",
    "assessment": "Original four-second VAD/window applicability unchanged; unassessed sources remain unassessed",
    "arms": {"separated": "Shared source classifier, any-source noisy-OR probability",
             "complement": "Same bag logit plus nonnegative original mixture raw-logit coefficient and intercept"},
    "training_target": "Original mixture truth only; an AI mixture implies at least one source, never two AI source labels",
    "source_identity": "No original foreground/reference waveform, condition, speaker/generator ID or truth mask as predictor or routing oracle",
    "normalization": "Fitting-only weighted stem center/std floor1e-6; each of two stems gets half its bag weight; embedding variance budget16",
    "weighting": "Unchanged equal truth/family/connected-group/source/assessed-variant bag weights",
    "model": "One shared1026-coefficient source logistic response; noisy-OR logit=logsumexp(z1,z2,z1+z2), permutation invariant",
    "explicit_raw_slopes": "Stem raw and mixture raw coefficients nonnegative; unconstrained embeddings do not imply physical monotonicity",
    "ridge": {"representation": 0.05, "stem_raw": 0.001, "mixture_raw": 0.001, "intercepts": 0.0},
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "initialization": "One fixed start: stem raw slope1, other stem coefficients0 except intercept-log(1+sqrt2); complement mixture slope1/intercept0",
    "nonconvexity": "One deterministic local optimizer; convergence is not a proof of global optimality or source identification",
    "target_smoothing": "Original connected fitting-group truth-class counts",
    "guards": "Own independent-calibration worst group/truth nonconformity with original unassessed groups retained",
    "stability": "Original raw and candidate logit change<=ln2 at six-decimal probabilities; separator rerun on required filtered original mixture",
    "selection": "Every original16 gates required; prefer separated unless complement adds>=5 correct with neither wrong direction higher and Brier no worse",
    "reference_prior": 0.5, "deployment_eligible": False,
}


def features(row):
    if row.get("raw") is None:
        if row.get("stems") is not None:
            raise ValueError("Unassessed mixture cannot acquire assessed candidate estimates")
        return None
    if isinstance(row["raw"], bool) or not valid(row["raw"]):
        raise ValueError("Original finite mixture response required")
    stems = row.get("stems")
    if not isinstance(stems, list) or len(stems) != 2:
        raise ValueError("Both complete unordered estimated sources required")
    embeddings, responses = [], []
    for stem in stems:
        vector = np.asarray(stem.get("embedding"), np.float64)
        if (vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all()
                or isinstance(stem.get("raw"), bool) or not valid(stem.get("raw"))):
            raise ValueError("Each estimated source requires finite frozen features and raw response")
        embeddings.append(vector)
        responses.append(clipped_logit(stem["raw"]))
    return np.array(embeddings), np.array(responses)


def design(row, transform):
    value = features(row)
    if value is None:
        return None
    embedding, response = value
    normalized = (embedding - transform["center"]) / transform["scale"] * transform["variance_factor"]
    return np.column_stack([response, normalized, np.ones(2)])


def bag_logit(z):
    z = np.asarray(z, np.float64)
    if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise ValueError("Two finite source logits per bag required")
    terms = np.column_stack([z[:, 0], z[:, 1], z.sum(axis=1)])
    output = logsumexp(terms, axis=1)
    mass = np.exp(terms - output[:, None])
    derivative = np.column_stack([mass[:, 0] + mass[:, 2], mass[:, 1] + mass[:, 2]])
    return output, derivative


def objective(beta, x, mixture, targets, weights, ridge, arm):
    if arm not in ARMS:
        raise ValueError("Unknown predeclared source arm")
    source = beta[:x.shape[-1]]
    z = np.einsum("nkd,d->nk", x, source)
    eta, derivative = bag_logit(z)
    if arm == "complement":
        eta = eta + beta[-2] * mixture + beta[-1]
    error = weights * (expit(eta) - targets)
    gradient = np.einsum("n,nk,nkd->d", error, derivative, x)
    if arm == "complement":
        gradient = np.r_[gradient, error @ mixture, error.sum()]
    loss = weights @ (np.logaddexp(0, eta) - targets * eta) + .5 * np.dot(ridge, beta**2)
    return float(loss), gradient + ridge * beta


def fit(rows, arm):
    if not rows or any(r["role"] != "fit" for r in rows) or arm not in ARMS:
        raise ValueError("Only original fitting roles may train the source classifier")
    selected = [r for r in rows if features(r) is not None]
    weights = balanced_weights(selected)
    embedding = np.array([features(r)[0] for r in selected])
    center = np.einsum("n,nkd->d", weights / 2, embedding)
    variance = np.einsum("n,nkd->d", weights / 2, (embedding - center)**2)
    transform = {"center": center.tolist(), "scale": np.maximum(np.sqrt(variance), 1e-6).tolist(),
                 "variance_factor": float(np.sqrt(16 / DIMENSIONS)),
                 "fit_sources": len({r["base_id"] for r in selected})}
    x = np.array([design(r, transform) for r in selected])
    mixture = np.array([clipped_logit(r["raw"]) for r in selected])
    counts = [len({r["group"] for r in selected if r["label"] == c}) for c in (0, 1)]
    if min(counts) < 2:
        raise ValueError("At least two connected fitting groups per truth class required")
    target = np.array([(counts[1] + 1) / (counts[1] + 2) if r["label"] else 1 / (counts[0] + 2) for r in selected])
    width = SOURCE_WIDTH + (2 if arm == "complement" else 0)
    ridge = np.full(width, .05)
    ridge[0], ridge[SOURCE_WIDTH - 1] = .001, 0.
    start = np.zeros(width)
    start[0], start[SOURCE_WIDTH - 1] = 1., -float(np.log1p(np.sqrt(2)))
    bounds = [(None, None)] * width
    bounds[0] = (0, None)
    if arm == "complement":
        ridge[-2], ridge[-1], start[-2], bounds[-2] = .001, 0., 1., (0, None)
    solved = minimize(objective, start, args=(x, mixture, target, weights, ridge, arm), jac=True,
        method="L-BFGS-B", bounds=bounds, options={k: METHOD["solver"][k] for k in ("ftol", "gtol", "maxiter")})
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError("Fixed any-source fitting did not converge: " + solved.message)
    return {"method": METHOD, "arm": arm, "transform": transform, "coefficients": solved.x.tolist(),
            "fit_groups_by_class": counts, "reference_prior": .5, "optimizer_iterations": int(solved.nit)}


def predict(row, fitted):
    validate_fitted(fitted)
    x = design(row, fitted["transform"])
    if x is None:
        return None
    beta = np.asarray(fitted["coefficients"])
    value = bag_logit((x @ beta[:SOURCE_WIDTH])[None])[0][0]
    if fitted["arm"] == "complement":
        value += beta[-2] * clipped_logit(row["raw"]) + beta[-1]
    return float(expit(value))


def validate_fitted(fitted):
    if fitted.get("arm") not in ARMS or fitted.get("method") != METHOD:
        raise ValueError("Exact declared fitted arm/method required")
    transform = fitted["transform"]
    center, scale, beta = [np.asarray(x, np.float64) for x in
                           (transform["center"], transform["scale"], fitted["coefficients"])]
    expected = SOURCE_WIDTH + (2 if fitted["arm"] == "complement" else 0)
    if (center.shape != (DIMENSIONS,) or scale.shape != center.shape or beta.shape != (expected,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.isfinite(beta).all()
            or (scale < 1e-6).any() or transform["variance_factor"] != np.sqrt(16 / DIMENSIONS)
            or beta[0] < 0 or (fitted["arm"] == "complement" and beta[-2] < 0)):
        raise ValueError("Finite complete fitting-only mapping and declared raw slopes required")


def calibrate(rows, fitted):
    if not rows or any(r["role"] != "cal" or type(r["label"]) is not int or r["label"] not in (0, 1) for r in rows):
        raise ValueError("Independent calibration roles and integer truth required")
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
            and b["false_original"] <= a["false_original"] and b["brier"] <= a["brier"] else "separated")
