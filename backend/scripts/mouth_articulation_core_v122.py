"""Small genuine-reference audio/mouth interactions; features, not AI probabilities.

Independent experiment inspired by frame-level interaction in C-MIND,
https://arxiv.org/html/2610.30872v1. No author code, weights or data included.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import numpy as np

from scripts.av_conditioning_clock_v109 import select_chunk

RANK = 16
RIDGE = 0.1
VISUAL_DIM = 2304  # current 768 + interpolation residual 768 + scaled motion 768
AUDIO_DIM = 3840  # 5 states x 384 channels x (mean, standard deviation)
METHOD = "native-mouth-reference-interaction/122"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def temporal_visual(before, current, after, times):
    """Blind native-picture context; no condition, known edit bounds or labels."""
    values = [np.asarray(v, np.float64) for v in (before, current, after)]
    require(
        all(v.shape == (768,) and np.isfinite(v).all() for v in values),
        "Three complete finite mouth vectors required",
    )
    require(
        len(times) == 3 and all(type(t) in (str, int, Fraction) for t in times),
        "Exact native source times required",
    )
    left, center, right = map(Fraction, times)
    require(
        left < center < right and max(center - left, right - center) <= Fraction(2, 25),
        "Observed nearby native context required; do not bridge gaps",
    )
    a, b, c = values
    interpolated = a * float((right - center) / (right - left)) + c * float((center - left) / (right - left))
    residual = np.abs(b - interpolated)
    motion = np.abs(c - a) * float(Fraction(1, 25) / (right - left))
    return np.concatenate((b, residual, motion))


def speech_summary(hidden, plan):
    """Keep the reviewed five-state order; do not treat a token as local proof."""
    require(
        plan["source_sample_status"] == "supported" and all(plan["retained_tick_slots"]),
        "Fully observed speech tick prefix required",
    )
    chunk = select_chunk(hidden, plan).reshape(10, 5, 384).astype(np.float64)
    result = np.concatenate((chunk.mean(axis=0).ravel(), chunk.std(axis=0).ravel()))
    require(
        result.shape == (AUDIO_DIM,) and np.isfinite(result).all(),
        "Complete finite native speech summary required",
    )
    return result


def projection(x, weights, rank=RANK):
    x, weights = np.asarray(x, np.float64), np.asarray(weights, np.float64)
    require(
        x.ndim == 2
        and x.shape[0] == len(weights)
        and min(x.shape) > rank
        and np.isfinite(x).all()
        and np.isfinite(weights).all()
        and (weights > 0).all(),
        "Complete weighted reference matrix required",
    )
    weights = weights / weights.sum()
    center = np.sum(weights[:, None] * x, axis=0)
    _, singular, components = np.linalg.svd((x - center) * np.sqrt(weights[:, None]), full_matrices=False)
    components = components[:rank].copy()
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    scale = np.maximum(singular[:rank], 1e-6)
    return {"center": center.tolist(), "components": components.tolist(), "scale": scale.tolist()}


def project(x, spec):
    x = np.asarray(x, np.float64)
    center, components, scale = (np.asarray(spec[k], np.float64) for k in ("center", "components", "scale"))
    require(
        x.ndim == 2
        and center.shape == (x.shape[1],)
        and components.ndim == 2
        and components.shape[1] == x.shape[1]
        and scale.shape == (components.shape[0],)
        and all(np.isfinite(v).all() for v in (x, center, components, scale))
        and (scale > 0).all(),
        "Complete finite matching reference projection required",
    )
    return ((x - center) @ components.T) / scale


def fit_reference(rows):
    """Fit only untouched fitting-role data; calibration/AI rows are forbidden."""
    require(
        rows and all(r["study_role"] == "fusion_fit" and r["condition"] == "original" for r in rows),
        "Only untouched fitting sources may learn reference relation",
    )
    counts = Counter(r["source_group"] for r in rows)
    require(
        len(counts) >= 2 and all(type(k) is str and k for k in counts),
        "Multiple explicit fitting sources required",
    )
    weights = np.asarray([1 / (len(counts) * counts[r["source_group"]]) for r in rows])
    visual, audio = (np.asarray([r[k] for r in rows], np.float64) for k in ("visual", "audio"))
    require(
        visual.shape == (len(rows), VISUAL_DIM) and audio.shape == (len(rows), AUDIO_DIM),
        "Complete native reference dimensions required",
    )
    vp, ap = projection(visual, weights), projection(audio, weights)
    v, a = project(visual, vp), project(audio, ap)
    mapping = np.linalg.solve(
        a.T @ (weights[:, None] * a) + RIDGE * np.eye(RANK), a.T @ (weights[:, None] * v)
    )
    require(np.isfinite(mapping).all(), "Finite genuine-reference alignment required")
    return {
        "method": METHOD,
        "rank": RANK,
        "ridge": RIDGE,
        "visual_projection": vp,
        "audio_projection": ap,
        "audio_to_visual": mapping.tolist(),
        "fit_sources": sorted(counts),
        "fit_points": len(rows),
        "response_kind": "features_only_not_probability",
        "deployment_eligible": False,
    }


def representations(visual, audio, reference):
    require(
        reference["method"] == METHOD
        and reference["rank"] == RANK
        and reference["ridge"] == RIDGE
        and reference["response_kind"] == "features_only_not_probability",
        "Exact reference interaction specification required",
    )
    v = project(visual, reference["visual_projection"])
    a = project(audio, reference["audio_projection"])
    mapping = np.asarray(reference["audio_to_visual"], np.float64)
    require(
        mapping.shape == (RANK, RANK)
        and np.isfinite(mapping).all()
        and v.shape == a.shape
        and v.shape[1] == RANK,
        "Corresponding finite projected branches required",
    )
    aligned = a @ mapping
    return {
        "visual_only": v,
        "concatenation": np.concatenate((v, aligned), axis=1),
        "interaction": np.concatenate((v, aligned, v * aligned, np.abs(v - aligned)), axis=1),
    }
