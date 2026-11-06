"""Frozen mouth-chip features and source-balanced linear probe; research only."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import cv2
import numpy as np

METHOD = "direct-mouth-dino-linear/6.0"
SIZE, BATCH = 112, 12
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def mouth_chip(aligned):
    height, width = aligned.shape[:2]
    bounds = [
        math.floor(width * 0.28),
        math.floor(height * 0.60),
        math.ceil(width * 0.72),
        math.ceil(height * 0.92),
    ]
    x0, y0, x1, y1 = bounds
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("Insufficient native mouth-chip resolution")
    return aligned[y0:y1, x0:x1].copy(), bounds


def preprocessed(chips):
    pixels = (
        np.stack([cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) for c in chips]).astype(
            np.float32
        )
        / 255.0
    )
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(0, 3, 1, 2))


def features(chips, dino):
    import torch

    pixels = preprocessed(chips)
    with torch.inference_mode():
        tokens = dino.model(pixel_values=torch.from_numpy(pixels).to(dino.device)).last_hidden_state.float()
        cls = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
        patches = torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=-1)
        values = torch.cat((cls, patches), dim=-1).cpu().numpy()
    if values.shape != (len(chips), 768) or not np.isfinite(values).all():
        raise ValueError("Unexpected frozen mouth-feature representation")
    return values, [hashlib.sha256(p.tobytes()).hexdigest() for p in pixels]


def source_weights(rows):
    counts = Counter((r["group"], r["label"], r["condition"]) for r in rows)
    conditions = {}
    for group, label, condition in counts:
        conditions.setdefault((group, label), set()).add(condition)
    actors = {r["group"] for r in rows}
    if any((g, label) not in conditions for g in actors for label in (0, 1)):
        raise ValueError("Both classes are required for every fitting actor")
    weights = [
        len(rows)
        / (
            2
            * len(actors)
            * len(conditions[(r["group"], r["label"])])
            * counts[(r["group"], r["label"], r["condition"])]
        )
        for r in rows
    ]
    return np.asarray(weights, np.float64)


def train_head(rows):
    from sklearn.linear_model import LogisticRegression

    fit = [r for r in rows if r["role"] == "fit"]
    point = [r for r in rows if r["role"] == "point"]
    if {r["group"] for r in fit} & {r["group"] for r in point}:
        raise ValueError("Fitting and point-selection actors overlap")
    if not fit or not point:
        raise ValueError("Both independent actor roles are required")
    x = np.asarray([r["features"] for r in fit], np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Missing or invalid fitting features")
    weights = source_weights(fit)
    mean = np.average(x, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, weights=weights, axis=0)), 0.05)
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=0)
    model.fit((x - mean) / scale, [r["label"] for r in fit], sample_weight=weights)
    if int(model.n_iter_.max()) >= 2000:
        raise ValueError("The predeclared linear head did not converge")
    head = {
        "method": METHOD,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "fit_groups": sorted({r["group"] for r in fit}),
        "point_groups": sorted({r["group"] for r in point}),
        "fit_frames": len(fit),
        "point_frames": len(point),
        "iterations": int(model.n_iter_.max()),
        "response_kind": "raw signed margin; not AI probability",
        "deployment_eligible": False,
    }
    controls = [r for r in point if not r["label"]]
    if {r["condition"] for r in controls} != {"original", "local-light", "local-resample"}:
        raise ValueError("All ordinary point-selection conditions are required")
    head["review_point"] = math.nextafter(
        max(0.0, max(margins([r["features"]], head)[0] for r in controls)), math.inf
    )
    return head


def margins(values, head):
    x = np.asarray(values, np.float64)
    return ((x - np.asarray(head["mean"])) / np.asarray(head["scale"])) @ np.asarray(head["coef"]) + head[
        "intercept"
    ]


def margin_windows(centers, tracks, values, table):
    windows = []
    for track, samples in tracks.items():
        indices = {r.frame_index for r in samples}
        blocks = [
            (i, float(np.mean([values[k] for k in (i - 1, i, i + 1)])))
            for i in sorted(centers)
            if all(k in indices and k in values for k in (i - 1, i, i + 1))
            and table.seconds(i + 1) - table.seconds(i - 1) <= 0.13
        ]
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        responses = np.array([v for _, v in blocks])
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < 0.52 or np.max(np.diff(times[start - 2 : end + 2])) > 0.55:
                    continue
                windows.append(
                    {
                        "track_id": track,
                        "start_s": float(times[start]),
                        "end_s": float(times[end - 1]),
                        "response": float(responses[start:end].mean()),
                        "before_margin": float(responses[start - 2 : start].mean()),
                        "after_margin": float(responses[end : end + 2].mean()),
                        "centers": [i for i, _ in blocks[start - 2 : end + 2]],
                    }
                )
    return sorted(windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))[:12]
