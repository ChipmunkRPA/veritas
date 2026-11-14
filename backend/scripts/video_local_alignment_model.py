# SPDX-License-Identifier: Apache-2.0
"""Train a small audio/mouth alignment scorer from within-recording mismatches."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.detectors.video import lipsync, xcorr

CONTEXT = (-2, -1, 0, 1, 2)
TRAIN_SHIFTS = (-9, -6, -4, 4, 6, 9)
BANDS = (100, 200, 350, 550, 800, 1150, 1600, 2200, 3000, 4000, 5500, 7000)
LAGS = np.arange(-9, 10)
WIDTHS = (14, 16, 18, 20, 22)
FLANK, GAP = 12, 1


def _context(values):
    rows = np.full((len(values), len(CONTEXT) * values.shape[1]), np.nan)
    if len(values) >= len(CONTEXT):
        rows[2:-2] = np.concatenate([values[2 + shift:len(values) - 2 + shift] for shift in CONTEXT], axis=1)
    return rows


def mouth_features(patches, frames):
    values = np.full((len(frames), 25), np.nan)
    for tick, patch in patches.items():
        standardized = (patch - patch.mean()) / max(5., patch.std())
        cells = standardized[:, 1:-1].reshape(4, 6, 6, 5).mean(axis=(1, 3)).ravel()
        values[tick] = [xcorr.aperture(patch), *cells]
    observed = np.isfinite(values[:, 0])
    if observed.any():
        values[observed] -= np.median(values[observed], axis=0)
    out = _context(values)
    # A repeated observation cannot stand in for a measured motion sequence.
    for tick in range(2, len(frames) - 2):
        if len(set(frames[tick - 2:tick + 3])) != 5:
            out[tick] = np.nan
    return out


def sound_features(pcm):
    hop = lipsync.HOP
    count = len(pcm) // hop
    chunks = pcm[:count * hop].astype(np.float64).reshape(count, hop) / 32768.
    spectrum = np.abs(np.fft.rfft(np.hanning(hop) * chunks)) ** 2
    bins = np.fft.rfftfreq(hop, 1. / lipsync.SR)
    power = np.stack([spectrum[:, (bins >= a) & (bins < b)].sum(axis=1)
                      for a, b in zip(BANDS, BANDS[1:])], axis=1)
    relative = np.log(power + 1e-6) - np.log(power.sum(axis=1, keepdims=True) + 1e-5)
    level = np.log(1e-4 + np.sqrt(np.mean(chunks ** 2, axis=1)))
    values = np.column_stack((level, relative))
    values -= np.median(values, axis=0)
    return _context(values)


def _joint(mouth, speech):
    return (mouth[:, :, None] * speech[:, None, :]).reshape(len(mouth), -1)


def fit(excerpts):
    visual, sound, labels = [], [], []
    for patches, grid, _, pcm in excerpts:
        faces = mouth_features(patches, grid.frames)
        voices = sound_features(pcm)
        activity = lipsync.vad(pcm.astype(np.float32) / 32768).speech
        margin = max(TRAIN_SHIFTS)
        ticks = np.arange(margin + 2, min(len(faces), len(voices) - margin - 2))
        ticks = ticks[np.isfinite(faces[ticks, 0]) & activity[ticks]]
        if not ticks.size:
            continue
        for lag in (0, *TRAIN_SHIFTS):
            visual.extend(faces[ticks])
            sound.extend(voices[ticks + lag])
            labels.extend([lag == 0] * len(ticks))
    if len(labels) < 700:
        raise ValueError("Insufficient same-time training observations")
    head = LogisticRegression(C=.05, class_weight="balanced", max_iter=400)
    models, features = [], []
    for inputs in (visual, sound):
        scale = StandardScaler().fit(inputs)
        normalized = scale.transform(inputs)
        pca = PCA(n_components=12, whiten=True, svd_solver="full").fit(normalized)
        models.append((scale, pca))
        features.append(pca.transform(normalized).clip(-5, 5))
    head.fit(_joint(*features), labels)
    return models, head, sum(labels)


def profile(model, patches, frames, pcm):
    models, head, _ = model
    features = []
    for (scale, pca), values in zip(models, (mouth_features(patches, frames), sound_features(pcm)), strict=True):
        valid = np.isfinite(values[:, 0])
        embedded = np.full((len(values), pca.n_components_), np.nan)
        if valid.any():
            embedded[valid] = pca.transform(scale.transform(values[valid])).clip(-5, 5)
        features.append(embedded)
    visual, sound = features
    out = np.full((len(visual), len(LAGS)), np.nan)
    for column, lag in enumerate(LAGS):
        ticks = np.arange(max(0, -lag), min(len(visual), len(sound) - lag))
        ticks = ticks[np.isfinite(visual[ticks, 0]) & np.isfinite(sound[ticks + lag, 0])]
        if ticks.size:
            out[ticks, column] = head.decision_function(_joint(visual[ticks], sound[ticks + lag]))
    return out


def bracket(start, stop):
    return ((start - GAP - FLANK, start - GAP), (start, stop), (stop + GAP, stop + GAP + FLANK))


def locate(values, activity):
    """Find a local offset supported at the same observed pictures in both lag hypotheses."""
    available = np.isfinite(values).all(axis=1)
    profiles = {}
    for width in set(WIDTHS) | {FLANK}:
        for start in range(len(values) - width + 1):
            rows = slice(start, start + width)
            good = available[rows]
            if good.sum() * 4 < 3 * width or activity[rows].sum() < 3:
                continue
            profiles[start, start + width] = values[rows][good].mean(axis=0)
    lag_delta = np.abs(LAGS[None, :] - LAGS[:, None])
    allowed = (lag_delta >= 4) & (lag_delta <= 8) & (np.abs(LAGS[:, None]) <= 4)
    best, count = None, 0
    for width in WIDTHS:
        for start in range(FLANK + GAP, len(values) - width - FLANK - GAP + 1):
            stop = start + width
            before, inside, after = (profiles.get(span) for span in bracket(start, stop))
            if before is None or inside is None or after is None:
                continue
            count += 1
            score = np.minimum(np.minimum(before[:, None] - before, after[:, None] - after), inside - inside[:, None])
            score = np.where(allowed, score, -np.inf)
            reference, changed = np.unravel_index(np.argmax(score), score.shape)
            gains = [before[reference] - before[changed], inside[changed] - inside[reference], after[reference] - after[changed]]
            gains = [round(float(value), 6) for value in gains]
            if min(gains) <= (best or {}).get("score", 0):
                continue
            best = {"ticks": [start, stop], "reference_lag": int(LAGS[reference]), "changed_lag": int(LAGS[changed]),
                    "score": min(gains), "gains": gains}
    return count, best
