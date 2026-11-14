# SPDX-License-Identifier: Apache-2.0
"""Compare which soundtrack ranges survive stronger flanks and a timing alternative."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

from app.evidence import video_reference_sound_difference as base

ARMS = ("current", "strong_flanks_first", "check_for_delay")


def candidates(earlier, submitted):
    """Enumerate corroborated ranges before choosing a single strongest measurement."""
    stats = [base._band_statistics(earlier, submitted, band) for band in base.speech.BANDS]
    count = 1 + (base.SAMPLES - base.WINDOW) // base.HOP
    for width in base.WIDTHS:
        for start in range(base.MIN_SUPPORT + base.GUARD, count - width - base.MIN_SUPPORT - base.GUARD + 1):
            end = start + width
            bounds = np.array([(max(0, start - base.GUARD - base.FLANK), start - base.GUARD), (start, end),
                               (end + base.GUARD, min(count, end + base.GUARD + base.FLANK))])
            bands = []
            for band, profile in zip(base.speech.BANDS, stats, strict=True):
                row = base._band_reading(profile, bounds)
                if (row and row["local_relative_gain"] < base.MAX_LOCAL_RELATIVE_GAIN
                        and row["correlation"][1] < base.MAX_LOCAL_CORRELATION):
                    bands.append({"band_hz": list(band), **row})
            if len(bands) >= 2:
                yield {"samples": [(n + 1) * base.HOP for n in (start, end)], "bands": bands,
                       "strength": round(float(np.median([1 - row["local_relative_gain"] for row in bands])), 6)}


def _precise_delay(earlier, submitted, a, b):
    # Confirm a delay on less than a full segment so a small boundary error
    # does not dominate the comparison, then test the same offset over it all.
    window = 1280
    context = 5760
    start = (a + b - window) // 2
    lo, hi = max(0, start - context), min(len(earlier) - window, start + context)
    local = earlier[lo:hi+window]
    query = submitted[start:start+window]
    numerator = correlate(local, query, mode="valid", method="fft")
    power = np.r_[0., np.cumsum(local**2)]
    norm = np.sqrt(np.maximum(0, power[window:] - power[:-window]) * np.dot(query, query))
    correlation = np.divide(numerator, norm, out=np.zeros_like(numerator), where=norm > 1e-12)
    peak = int(correlation.argmax())
    delay = start - (lo + peak)
    if abs(delay) < 960 or correlation[peak] < .42 or a - delay < 0 or b - delay > len(earlier):
        return None
    source, delivered = earlier[a-delay:b-delay], submitted[a:b]
    value = np.dot(source, delivered) / max(1e-12, np.linalg.norm(source) * np.linalg.norm(delivered))
    if value >= .37:
        return {"lag_samples": delay, "short_match": round(float(correlation[peak]), 6),
                "range_match": round(float(value), 6)}
    return None


def inspect(earlier, submitted):
    all_ranges = sorted(candidates(earlier, submitted), key=lambda row: -row["strength"])
    if not all_ranges:
        return dict.fromkeys(ARMS)
    best = all_ranges[0]
    strict = next((row for row in all_ranges if base.strong_flanks(row)), None)
    current = best if base.strong_flanks(best) else None
    mild = next((row for row in all_ranges if base.strong_flanks(row, .52)), None)
    if mild:
        a, b = mild["samples"]
        # Low-order common band filtering suppresses the strongest train rumble.
        filter_ = butter(2, (240, 3400), fs=base.RATE, btype="bandpass", output="sos")
        x, y = (sosfiltfilt(filter_, pcm.astype(np.float64) / 32768) for pcm in (earlier, submitted))
        if _precise_delay(x, y, a, b):
            mild = None
    return {"current": current, "strong_flanks_first": strict, "check_for_delay": mild}
