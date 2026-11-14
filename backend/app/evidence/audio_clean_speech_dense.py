"""Find a short drop in sound shared with an aligned clean earlier copy."""

from __future__ import annotations

import numpy as np

from app.evidence import audio_clean_speech as earlier

BASIS = "dense_clean_reference_v1"
WIDTH = 1920
HOP = 640
INTERVAL_FRAMES = 24
FLANK_FRAMES = 16
GUARD_FRAMES = 2
POINT = 0.31758884


def _accumulate(values):
    return np.r_[0, np.cumsum(values)]


def _averages(strength, gain, cutoff, cap, bounds):
    weights = np.minimum(strength, cap) ** 2 * (strength >= cutoff)
    sums, weighted_gain, count = (_accumulate(values) for values in (weights, weights * gain, weights > 0))
    averages = [(weighted_gain[b] - weighted_gain[a]) / np.maximum(sums[b] - sums[a], 1e-15)
                for a, b in bounds]
    support = [count[b] - count[a] for a, b in bounds]
    return averages, support


def _strongest(first, second):
    frames = 1 + (earlier.SAMPLES - WIDTH) // HOP
    starts = np.arange(1, frames - INTERVAL_FRAMES)
    ends = starts + INTERVAL_FRAMES
    bounds = (
        (np.maximum(0, starts - FLANK_FRAMES - GUARD_FRAMES), np.maximum(0, starts - GUARD_FRAMES)),
        (starts, ends),
        (np.minimum(frames, ends + GUARD_FRAMES), np.minimum(frames, ends + GUARD_FRAMES + FLANK_FRAMES)),
    )
    measurements, deviations = [], []
    for frequency, strength, gain, cutoff, cap in earlier.band_stats(first, second, np.hanning(WIDTH), hop=HOP):
        (left, middle, right), support = _averages(strength, gain, cutoff, cap, bounds)
        flank = np.minimum(left, right)
        valid = (flank > .25) & (np.maximum(left, right) <= 2.2 * flank) & (np.min(support, axis=0) >= 3)
        deviations.append(np.where(valid, 1 - middle / np.maximum(flank, 1e-10), np.nan))
        measurements.append((frequency, left, middle, right, support))

    values = np.stack(deviations)
    eligible = np.flatnonzero(np.sum(np.isfinite(values), axis=0) >= 2)
    if not eligible.size:
        return None
    scores = np.nanmedian(values[:, eligible], axis=0)
    top = int(np.argmax(scores))
    index = int(eligible[top])
    start = int(starts[index]) * HOP + (WIDTH - HOP) // 2
    bands = [
        {"band_hz": list(hz), "before": round(float(left[index]), 8), "within": round(float(middle[index]), 8),
         "after": round(float(right[index]), 8), "active_frames": [int(count[index]) for count in support]}
        for (hz, left, middle, right, support), band in zip(measurements, values, strict=True)
        if np.isfinite(band[index])
    ]
    return {"relative_drop": round(float(scores[top]), 8), "samples": [start, start + INTERVAL_FRAMES * HOP],
            "bands": bands}


def measure(first, second):
    """Return a disposition and strongest supported measurement before the operating point."""
    if not earlier.matched(first, second, np.hanning(earlier.WIDTH)):
        return "reference_match_not_supported", None
    result = _strongest(first, second)
    return ("local_support_not_available", None) if result is None else ("measured", result)


def locate(first, second):
    status, result = measure(first, second)
    if result is None:
        return status, None
    if result["relative_drop"] <= POINT:
        return "no_local_change", None
    return "local_change", result
