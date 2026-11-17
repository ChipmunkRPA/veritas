"""Bounded comparison of local matching-sound loss between two aligned clips."""
from __future__ import annotations

import numpy as np

from app.evidence import audio_clean_speech as speech

RATE = speech.RATE
SAMPLES = 4 * RATE
SHORT_MIN_SAMPLES = 30_000
WINDOW, HOP = 1920, 640
WEIGHT = np.hanning(WINDOW)
WIDTHS = (8, 12, 16, 20)
FLANK, GUARD = 14, 2
MIN_SUPPORT = 4
MIN_FLANK_CORRELATION = .42
MIN_FLANK_GAIN = .55
MAX_LOCAL_CORRELATION = .30
MAX_LOCAL_RELATIVE_GAIN = .38
CLEAN_FLANK_CORRELATION = .84


def strong_flanks(reading, floor=CLEAN_FLANK_CORRELATION):
    """Require two bands with reliable matching sound on both sides."""
    return sum(min(band["correlation"][0], band["correlation"][2]) >= floor for band in reading["bands"]) >= 2


def _band_statistics(earlier, submitted, band, *, weight=WEIGHT, hop=HOP, floor=.14):
    frames = [speech._frames(pcm, band, weight, hop=hop) * np.sqrt(weight) for pcm in (earlier, submitted)]
    power = np.square(frames[0]).sum(axis=1)
    delivered = np.square(frames[1]).sum(axis=1)
    cross = (frames[0] * frames[1]).sum(axis=1)
    active = power >= max(weight.sum() * 12**2, np.quantile(power, .85) * floor)
    active_weight = np.minimum(power, np.quantile(power, .90)) * active
    gain = cross / np.maximum(power, 1e-10)
    correlation = cross / np.sqrt(np.maximum(power * delivered, 1e-10))
    return [np.r_[0, np.cumsum(values)] for values in (
        active, active_weight, active_weight * gain, active_weight * correlation, delivered * active,
    )]


def _band_reading(stats, bounds, *, min_support=MIN_SUPPORT):
    support, weight, gain, agreement, delivered = (prefix[bounds[:, 1]] - prefix[bounds[:, 0]] for prefix in stats)
    if min(support) < min_support or np.any(weight <= 0):
        return None
    gain, agreement = gain / weight, agreement / weight
    flank_gain = min(gain[0], gain[2])
    flank_correlation = min(agreement[0], agreement[2])
    if (flank_gain < MIN_FLANK_GAIN or max(gain[0], gain[2]) > 2 * flank_gain
            or flank_correlation < MIN_FLANK_CORRELATION or delivered[1] < 1e4):
        return None
    return {"matching_gain": [round(float(x), 6) for x in gain],
            "correlation": [round(float(x), 6) for x in agreement],
            "active_windows": support.astype(int).tolist(),
            "local_relative_gain": round(float(gain[1] / flank_gain), 6)}


def measure(earlier, submitted, *, recover_supported=False, short_native=False):
    """Compare known shared-time sound on either side of a possible local source change."""
    samples = len(earlier)
    minimum = SHORT_MIN_SAMPLES if short_native else SAMPLES
    if samples != len(submitted) or not minimum <= samples <= SAMPLES:
        return "outside_scope", None
    statistics = [_band_statistics(earlier, submitted, band) for band in speech.BANDS]
    count = 1 + (samples - WINDOW) // HOP
    best = None
    supported = None
    measured = False
    for width in WIDTHS:
        for start in range(MIN_SUPPORT + GUARD, count - width - MIN_SUPPORT - GUARD + 1):
            end = start + width
            bounds = np.array([(max(0, start - GUARD - FLANK), start - GUARD), (start, end),
                               (end + GUARD, min(count, end + GUARD + FLANK))])
            bands = []
            for band, stats in zip(speech.BANDS, statistics, strict=True):
                if reading := _band_reading(stats, bounds):
                    bands.append({"band_hz": list(band), **reading})
            if len(bands) < 2:
                continue
            measured = True
            qualifying = [band for band in bands if band["local_relative_gain"] < MAX_LOCAL_RELATIVE_GAIN
                          and band["correlation"][1] < MAX_LOCAL_CORRELATION]
            if len(qualifying) < 2:
                continue
            score = round(float(np.median([1 - band["local_relative_gain"] for band in qualifying])), 6)
            reading = {"samples": [start * HOP + (WINDOW - HOP) // 2, end * HOP + (WINDOW - HOP) // 2],
                       "strength": score, "bands": qualifying}
            if best is None or score > best["strength"]:
                best = reading
            if recover_supported and (supported is None or score > supported["strength"]) and strong_flanks(reading):
                supported = reading
    if supported is not None and not strong_flanks(best):
        return "localized_difference", {**supported, "selection": "strong_flanks_first"}
    if best:
        return "localized_difference", best
    return ("no_local_difference" if measured else "matching_flanks_unavailable"), None
