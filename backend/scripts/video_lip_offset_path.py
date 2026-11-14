# SPDX-License-Identifier: Apache-2.0
"""Experimental shot-level comparison of constant and temporary AV lag."""

from __future__ import annotations

import numpy as np

from app.detectors.video import lipsync
from scripts.video_local_alignment_model import mouth_features, sound_features

LAGS = np.arange(-9, 10)
REFERENCE_LAGS = np.flatnonzero(np.abs(LAGS) <= 6)
WIDTHS = (8, 12, 16, 22)
FLANK = 18
TRANSITION_MARGIN = 2
RIDGE = .20
COMPONENTS = 8
MIN_SPEECH = .40


def _center_scale(samples):
    center = np.median(samples, axis=0)
    spread = np.maximum(samples.std(axis=0), .01)
    return center, spread


def _unit(values, scale):
    center, spread = scale
    return ((values - center) / spread).clip(-5, 5)


def _inverse_root(values):
    covariance = values.T @ values / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * (np.maximum(eigenvalues, 0) + RIDGE) ** -.5) @ eigenvectors.T


def fit(items):
    """Fit regularized canonical correlations on synchronized speech only."""
    pictures, sounds, used = [], [], []
    for patches, frames, pcm in items:
        visual = mouth_features(patches, frames)
        audio = sound_features(pcm)
        speech = lipsync.vad(pcm.astype(np.float32) / 32768.).speech
        cap = min(len(visual), len(audio), len(speech))
        keep = np.isfinite(visual[:cap]).all(axis=1) & np.isfinite(audio[:cap]).all(axis=1) & speech[:cap]
        if keep.sum() >= 24:
            pictures.append(visual[:cap][keep])
            sounds.append(audio[:cap][keep])
            used.append(int(keep.sum()))
    if len(pictures) < 6 or sum(used) < 400:
        raise ValueError("Insufficient observed synchronized speech for the frozen fit")
    visual, audio = np.concatenate(pictures), np.concatenate(sounds)
    video_scale, audio_scale = _center_scale(visual), _center_scale(audio)
    visual, audio = _unit(visual, video_scale), _unit(audio, audio_scale)
    left, right = _inverse_root(visual), _inverse_root(audio)
    u, values, vt = np.linalg.svd((visual @ left).T @ (audio @ right) / len(visual), full_matrices=False)
    weights = np.sqrt(values[:COMPONENTS])
    return (video_scale, left @ (u[:, :COMPONENTS] * weights),
            audio_scale, right @ (vt[:COMPONENTS].T * weights)), used


def grid_scores(model, patches, frames, pcm):
    vs, vp, aus, ap = model
    visual = _unit(mouth_features(patches, frames), vs) @ vp
    audio = _unit(sound_features(pcm), aus) @ ap
    speech = lipsync.vad(pcm.astype(np.float32) / 32768.).speech
    size = min(len(visual), len(audio), len(speech))
    costs = np.full((size, len(LAGS)), np.nan)
    ticks = np.arange(abs(LAGS[0]), size - LAGS[-1])
    gathered = audio[ticks[:, None] + LAGS]
    lhs = visual[ticks]
    norms = np.linalg.norm(lhs, axis=1)[:, None] * np.linalg.norm(gathered, axis=2)
    scores = np.einsum("tc,tlc->tl", lhs, gathered)
    costs[ticks] = np.divide(scores, norms, out=np.zeros_like(scores), where=norms > 1e-8)
    valid = np.isfinite(lhs).all(axis=1) & np.isfinite(gathered).all(axis=(1, 2)) & (norms.min(axis=1) > 1e-8)
    costs[ticks[~valid]] = np.nan
    return costs, speech[:size]


def _indices(a, b, observed):
    return np.flatnonzero(observed[a:b]) + a


def compare(scores, speech, shot):
    """Require one lag before and after a bounded excursion in one verified shot."""
    measured = speech & np.isfinite(scores).all(axis=1)
    found = {"tested": 0, "best": None}
    win = -np.inf
    side = FLANK + TRANSITION_MARGIN
    for width in WIDTHS:
        for start in range(side, len(scores) - width - side + 1):
            stop = start + width
            if not (shot[start - side] >= 0 and np.all(shot[start - side:stop + side] == shot[start - side])):
                continue
            left = _indices(start - side, start - TRANSITION_MARGIN, measured)
            inner = _indices(start + 1, stop - 1, measured)
            right = _indices(stop + TRANSITION_MARGIN, stop + side, measured)
            if (min(len(left), len(right)) < FLANK * MIN_SPEECH or len(inner) < max(5, (width - 2) * MIN_SPEECH)
                    or inner[-1] - inner[0] < (width - 2) / 2):
                continue
            found["tested"] += 1
            before, center, after = [scores[indices].mean(axis=0) for indices in (left, inner, right)]
            normal = (len(left) * before + len(inner) * center + len(right) * after)
            constant = float(normal.max())
            for base in REFERENCE_LAGS:
                # Use the same observed samples for both timing hypotheses.
                # Both flanks must favor the common lag over the changed lag.
                allowed = (abs(LAGS - LAGS[base]) >= 3) & (abs(LAGS - LAGS[base]) <= 8)
                entrance = before[base] - before
                exit_ = after[base] - after
                interior = center - center[base]
                improvement = (normal[base] + len(inner) * interior - constant) / len(inner)
                response = np.minimum.reduce((2 * entrance, interior, 2 * exit_, improvement))
                allowed &= (np.minimum(before[base], after[base]) > .10) & (center > .10)
                rank = np.where(allowed, response, -np.inf)
                changed = int(rank.argmax())
                strongest = float(rank[changed])
                if strongest > max(win, 0.):
                    win = strongest
                    entries = [entrance[changed], interior[changed], exit_[changed], improvement[changed]]
                    found["best"] = {"span": [start, stop], "reference": int(LAGS[base]), "changed": int(LAGS[changed]),
                                     "support_ticks": [len(left), len(inner), len(right)],
                                     "gains": [round(float(entry), 8) for entry in entries], "score": round(win, 8)}
    return found
