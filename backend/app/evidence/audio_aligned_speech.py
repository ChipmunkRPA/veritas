"""Locate one speech difference between two time-aligned earlier-copy recordings."""

from __future__ import annotations

import math
from fractions import Fraction

RATE = 16_000
HOP = 320
WINDOW = 640
FLANK = 20
SIZES = (6, 10, 16, 24, 36, 54, 81, 120)
MIN_SAMPLES = 29 * RATE // 5
MAX_SAMPLES = 12 * RATE
SELECTION_POINT = 0.443620
PROCESSING_LIMIT = 0.10
BASIS = "paired_speech_alignment/1"


def resampled(pcm, up, down):
    import numpy as np
    from scipy.signal import resample_poly

    return np.clip(np.rint(resample_poly(pcm.astype(np.float64), up, down)), -32768, 32767).astype("<i2")


def spectral_features(pcm):
    import numpy as np
    from scipy.fft import dct

    frame = np.lib.stride_tricks.sliding_window_view(np.pad(pcm.astype(np.float64), WINDOW // 2), WINDOW)[::HOP]
    frame = frame[:len(pcm) // HOP]
    level = np.sqrt(np.mean(frame**2, axis=1))
    active = level >= max(10, np.percentile(level, 85) * 0.12)
    power = abs(np.fft.rfft(frame * np.hanning(WINDOW), n=1024))**2
    hz = np.fft.rfftfreq(1024, 1 / RATE)
    mel = 2595 * np.log10(1 + hz / 700)
    corners = np.linspace(2595 * np.log10(1 + 200 / 700), 2595 * np.log10(1 + 6800 / 700), 30)
    low, center, high = corners[:-2, None], corners[1:-1, None], corners[2:, None]
    filters = np.maximum(0, np.minimum((mel - low) / (center - low), (high - mel) / (high - center)))
    bands = power @ (filters / filters.sum(axis=1, keepdims=True)).T
    floor = max(1., float(np.percentile(bands, 90))) * 1e-5
    features = dct(np.log(np.maximum(bands, floor)), norm="ortho", axis=1)[:, 1:13]
    features -= np.median(features[active], axis=0) if active.any() else np.median(features, axis=0)
    padded = np.pad(features, ((2, 2), (0, 0)), mode="edge")
    delta = (padded[4:] - padded[:-4]) * 0.5
    return np.concatenate((features, delta), axis=1), active


def path_indices(cost):
    """Evaluate the three DTW predecessors on one antidiagonal at a time."""
    import numpy as np

    m, n = cost.shape
    total = np.full((m + 1, n + 1), np.inf)
    previous = np.zeros((m + 1, n + 1), dtype=np.uint8)
    total[0, 0] = 0
    for diagonal in range(2, m + n + 1):
        i = np.arange(max(1, diagonal - n), min(m, diagonal - 1) + 1)
        j = diagonal - i
        within_band = abs(j - 1 - (i - 1) * (n - 1) / (m - 1)) <= RATE * 0.8 / HOP
        i, j = i[within_band], j[within_band]
        values = np.stack((total[i-1, j-1], total[i-1, j] + 0.08, total[i, j-1] + 0.08))
        previous[i, j] = np.argmin(values, axis=0)
        total[i, j] = np.min(values, axis=0) + cost[i-1, j-1]
    i, j = m, n
    result = []
    while i and j:
        result.append((i-1, j-1))
        step = previous[i, j]
        i -= step != 2
        j -= step != 1
    if i or j or not math.isfinite(total[m, n]):
        raise ValueError("Could not reconstruct a complete alignment")
    return np.array(result[::-1]).T


def measure(first, second):
    import numpy as np
    from scipy.spatial.distance import cdist

    a, active_a = spectral_features(first)
    b, active_b = spectral_features(second)
    cost = np.sqrt(cdist(a, b, metric="sqeuclidean") / a.shape[1])
    earlier, submitted = path_indices(cost)
    n = len(b)
    counts = np.bincount(submitted, minlength=n)
    matches = np.bincount(submitted, weights=cost[earlier, submitted], minlength=n) / counts
    positions = np.bincount(submitted, weights=earlier, minlength=n) / counts
    activity = np.bincount(submitted, weights=active_a[earlier] & active_b[submitted], minlength=n) / counts
    path_median = round(float(np.median(matches)), 6)
    best = None
    if path_median <= 0.7:
        cumulative = np.r_[0., np.cumsum(matches)]
        spoken = np.r_[0., np.cumsum(activity)]
        for width in SIZES:
            start = np.arange(FLANK, n - FLANK - width + 1)
            end = start + width
            middle = (cumulative[end] - cumulative[start]) / width
            left = (cumulative[start] - cumulative[start-FLANK]) / FLANK
            right = (cumulative[end+FLANK] - cumulative[end]) / FLANK
            supported = ((left <= 0.7) & (right <= 0.7)
                         & (spoken[start] - spoken[start-FLANK] >= 0.3 * FLANK)
                         & (spoken[end+FLANK] - spoken[end] >= 0.3 * FLANK))
            scores = np.where(supported, middle - np.maximum(left, right), -np.inf)
            if not np.any(supported):
                continue
            k = int(np.argmax(scores))
            candidate = {"score": round(float(scores[k]), 6), "span": [int(start[k]) * HOP, int(end[k]) * HOP],
                         "earlier_span": [round(positions[start[k]]) * HOP, (round(positions[end[k]-1]) + 1) * HOP],
                         "means": [round(float(value), 6) for value in (left[k], middle[k], right[k])]}
            if best is None or (candidate["score"], -candidate["span"][0]) > (best["score"], -best["span"][0]):
                best = candidate
    return {"median_path_cost": path_median, "selected": best}


def processing_residual(first, second):
    import numpy as np

    if abs(len(second) - len(first)) * 50 > len(first):
        return None
    n = len(first)
    m = len(second)
    center = Fraction(2*m - 1, 2*n)
    ratio = None
    for denominator in range(1, 257):
        nearest = center.limit_denominator(denominator)
        if m - 1 < nearest * n <= m:
            ratio = nearest
            break
    if ratio is None:
        return None
    reference = resampled(first, ratio.numerator, ratio.denominator) if ratio != 1 else first
    if len(reference) != m:
        return None
    n = m
    half = 24
    features = np.lib.stride_tricks.sliding_window_view(reference[:n].astype(np.float64), 2 * half + 1)
    expected = second[half:n-half].astype(np.float64)
    blocks = len(expected) // WINDOW
    design = features[:blocks * WINDOW].reshape(blocks, WINDOW, -1)
    target = expected[:blocks * WINDOW].reshape(blocks, WINDOW)
    energy = np.sqrt(np.mean(target**2, axis=1))
    active = energy >= max(32, np.percentile(energy, 85) * 0.12)
    fold = np.arange(blocks) % 2
    values = []
    for training in (0, 1):
        fit = (fold == training) & active
        test = (fold != training) & active
        if min(fit.sum(), test.sum()) < 12:
            return None
        weights, *_ = np.linalg.lstsq(design[fit, ::8].reshape(-1, design.shape[-1]), target[fit, ::8].reshape(-1),
                                      rcond=1e-6)
        error = np.sqrt(np.mean((design[test] @ weights - target[test])**2, axis=1))
        values.append(float(np.percentile(error / energy[test], 95)))
    return round(max(values), 6)


def locate(first, second):
    """Use the frozen clean-copy selection point and ordinary processing control."""
    measurement = measure(first, second)
    candidate = measurement["selected"]
    residual = None
    selected = None
    if candidate is None:
        status = "insufficient_matching_speech"
    elif candidate["score"] <= SELECTION_POINT or candidate["earlier_span"][0] >= candidate["earlier_span"][1]:
        status = "no_local_difference"
    else:
        residual = processing_residual(first, second)
        if residual is not None and residual <= PROCESSING_LIMIT:
            status = "broad_processing_difference"
        else:
            status = "localized_difference"
            selected = candidate
    return status, selected, measurement["median_path_cost"], residual
