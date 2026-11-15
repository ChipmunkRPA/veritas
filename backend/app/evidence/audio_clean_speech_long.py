"""Earlier-copy speech evidence for a continuous twelve-second native pair."""

from __future__ import annotations

import numpy as np

from app.evidence import audio_clean_speech

RATE = audio_clean_speech.RATE
SAMPLES = 12 * RATE
WINDOWS = tuple(range(0, audio_clean_speech.SAMPLES + 1, 19200))
DENSE_WINDOWS = tuple(range(0, audio_clean_speech.SAMPLES + 1, 6400))
FFT, HOP = 2048, 1024
BANDS = ((300, 850), (850, 1450), (1450, 2100), (2100, 2800), (2800, 3700))


def measure_windows(first, second, starts):
    values = []
    for start in starts:
        end = start + audio_clean_speech.SAMPLES
        reason, result = audio_clean_speech.locate(first[start:end], second[start:end])
        if result:
            result = {**result, "samples": [x + start for x in result["samples"]]}
        values.append({"start": start, "reason": reason, "result": result})
    return values


def corroborate_windows(values, gap=19200):
    hits = {row["start"]: row["result"] for row in values if row["result"] is not None}
    witnessed = []
    for start, a in hits.items():
        b = hits.get(start + gap)
        if b and abs(a["samples"][0] - b["samples"][0]) <= audio_clean_speech.RANGE_SAMPLES // 2:
            strongest = a if a["relative_drop"] >= b["relative_drop"] else b
            witnessed.append((min(a["relative_drop"], b["relative_drop"]), strongest, [start, start + gap]))
    best = max(witnessed, key=lambda entry: (entry[0], -entry[1]["samples"][0]), default=None)
    return None if best is None else best[1] | {"supporting_window_starts": best[2]}


def _compare_windows(first, second):
    values = measure_windows(first, second, WINDOWS)
    matched = [r["reason"] != "reference_match_not_supported" for r in values]
    admitted = sum(matched) >= 4 and matched[0] and matched[-1]
    return {"windows": values, "admitted": admitted, "corroborated": corroborate_windows(values) if admitted else None}


def denser_windows(earlier, submitted, original_reason):
    if original_reason in {"reference_match_not_supported", "consistent_frequency_balance_change"}:
        return []
    return measure_windows(earlier, submitted, DENSE_WINDOWS)


def _spectra(signal):
    starts = np.lib.stride_tricks.sliding_window_view(signal.astype(np.float64), FFT)[::HOP]
    return np.fft.rfft(starts * np.hanning(FFT))


def _band_gain(x, y, mask):
    x, y = x[:, mask], y[:, mask]
    power = np.sum(abs(x) ** 2, axis=1)
    weights = np.minimum(1., np.quantile(power, .85) / np.maximum(power, 1e-6))
    return float(np.dot(weights, np.sum(np.real(np.conj(x) * y), axis=1)) / max(1e-6, np.dot(weights, power)))


def _profile(first, second, section_samples):
    frequencies = np.fft.rfftfreq(FFT, 1 / RATE)
    masks = [(frequencies >= low) & (frequencies < high) for low, high in BANDS]
    vectors = []
    for offset in range(0, 3 * section_samples, section_samples):
        end = offset + section_samples
        x, y = (_spectra(signal[offset:end]) for signal in (first, second))
        gains = np.array([_band_gain(x, y, mask) for mask in masks])
        if min(gains[:2]) <= .25:
            vectors.append(None)
        else:
            vectors.append(gains / np.median(gains[:2]))
    return vectors


def frequency_balance(first, second, *, section_samples=4 * RATE):
    vectors = _profile(first, second, section_samples)
    earlier, middle, later = vectors
    if earlier is None or later is None:
        return False, None
    typical = np.median([row for row in vectors if row is not None], axis=0)
    # The tilt must have the same sign and approximately the same shape across the file.
    tilt = typical[-1] - typical[0]
    direction = float(np.sign(tilt))
    consistent = all(direction * (vector[-1] - vector[0]) >= .25
                     and float(np.median(abs(vector - typical))) <= .18 for vector in (earlier, later))
    if middle is not None:
        consistent &= (direction * (middle[-1] - middle[0]) >= .15
                       and float(np.median(abs(middle - typical))) <= .22)
    explained = bool(consistent and abs(tilt) >= .35 and direction * (typical[-1] - typical[2]) >= .18)
    if not explained:
        return False, None
    trace = {"bands_hz": [list(band) for band in BANDS], "section_start_samples": [0, section_samples, 2 * section_samples],
             "relative_gains": [None if p is None else [round(float(v), 8) for v in p] for p in vectors]}
    return True, trace


def locate(earlier, submitted):
    """Return a physical local difference or a distinct coverage/processing reason."""
    observation = _compare_windows(earlier, submitted)
    if not observation["admitted"]:
        return "reference_match_not_supported", None, None
    explained, trace = frequency_balance(earlier, submitted)
    if explained:
        return "consistent_frequency_balance_change", None, trace
    if (span := observation["corroborated"]) is not None:
        return "local_change", span, None
    if all(window["reason"] in {"reference_match_not_supported", "local_support_not_available"}
           for window in observation["windows"]):
        return "local_support_not_available", None, None
    return "no_corroborated_local_change", None, None
