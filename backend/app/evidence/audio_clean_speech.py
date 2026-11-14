"""Local earlier-copy speech evidence on the frozen six-second sample grid."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE = 16_000
SAMPLES = 6 * RATE
WIDTH = 3_840
HOP = 1_280
POINT = 0.12831161
RANGE_SAMPLES = 9 * HOP
BANDS = ((300, 900), (900, 1750), (1750, 3400))


def _frames(pcm, frequency, weight, *, hop):
    sos = butter(2, frequency, fs=RATE, btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, pcm.astype(np.float64))
    frames = np.lib.stride_tricks.sliding_window_view(filtered, len(weight))[::hop]
    return frames - (frames @ weight / weight.sum())[:, None]


def matched(before, after, weight):
    earlier, submitted = (_frames(pcm, (250, 3800), weight, hop=HOP) * np.sqrt(weight) for pcm in (before, after))
    ref = np.einsum("ij,ij->i", earlier, earlier)
    sub = np.einsum("ij,ij->i", submitted, submitted)
    joint = np.einsum("ij,ij->i", earlier, submitted)
    strength = np.sqrt(ref / weight.sum())
    active = (strength >= max(12, float(np.quantile(strength, .8)) * .25)) & (sub > 0)
    corr = joint / np.sqrt(np.maximum(ref * sub, 1e-10))
    return (np.count_nonzero(active) >= 36 and np.count_nonzero(active & (corr >= .22)) >= 18
            and np.quantile(corr[active], .7) >= .30)


def band_stats(before, after, weight, *, hop=HOP):
    for frequency in BANDS:
        earlier, submitted = (_frames(pcm, frequency, weight, hop=hop) for pcm in (before, after))
        power = (earlier * earlier) @ weight
        joint = (earlier * submitted) @ weight
        strength = np.sqrt(power / weight.sum())
        yield frequency, strength, joint / np.maximum(power, 1e-8), max(12., .45 * np.quantile(strength, .85)), np.quantile(strength, .9)


def _strongest(before, after, weight):
    stats = list(band_stats(before, after, weight))
    centers = np.arange(1 + (SAMPLES - WIDTH) // HOP) * HOP + WIDTH // 2
    best = None
    for start in range(8, len(centers) - 17):
        stop = start + 9
        regions = (slice(start - 8, start - 1), slice(start, stop), slice(stop + 1, stop + 8))
        scores, support = [], []
        for frequency, strength, gain, cutoff, cap in stats:
            values = []
            for region in regions:
                weights = np.minimum(strength[region], cap) ** 2
                mask = strength[region] >= cutoff
                if np.count_nonzero(mask) < 3:
                    break
                values.append(float(np.dot(weights[mask], gain[region][mask]) / weights[mask].sum()))
            if len(values) != 3:
                continue
            left, local, right = values
            if min(left, right) <= .25 or max(left, right) / min(left, right) > 2.2:
                continue
            scores.append(1 - local / min(left, right))
            support.append({
                "band_hz": list(frequency), "before": round(left, 8), "within": round(local, 8), "after": round(right, 8),
            })
        if len(scores) >= 2:
            score = round(float(np.median(scores)), 8)
            if best is None or score > best["relative_drop"]:
                first = int(centers[start] - HOP // 2)
                best = {"relative_drop": score, "samples": [first, first + RANGE_SAMPLES], "bands": support}
    return best


def locate(earlier, submitted):
    """Accept native signed 16-bit samples at the pre-verified common clock."""
    weight = np.hanning(WIDTH)
    if not matched(earlier, submitted, weight):
        return "reference_match_not_supported", None
    result = _strongest(earlier, submitted, weight)
    if result is None:
        return "local_support_not_available", None
    if result["relative_drop"] <= POINT:
        return "no_local_change", None
    return "local_change", result
