"""Measure concurrent harmonic structure on a fixed short mono sample grid."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks

RATE = 16_000
SAMPLES = 48_000
FRAME = 1024
HOP = 320
FFT = 4096
MIN_RUN = 3
PITCHES = 65 * np.exp2(np.arange(128) / 48)
METHOD = {
    "frame_samples": FRAME, "hop_samples": HOP, "fft_samples": FFT, "minimum_run_frames": MIN_RUN,
    "pitch_start_hz": 65, "pitch_steps_per_octave": 48, "pitch_candidates": len(PITCHES),
    "minimum_frame_rms_s16": 32, "peak_prominence_db": 8, "peak_relative_floor_db": -32,
    "peak_band_hz": [65, 4200], "maximum_harmonic": 16, "minimum_supported_harmonics": 3,
    "secondary_minimum_relative_energy": 0.16, "minimum_pitch_separation_semitones": 2,
    "maximum_frame_pitch_change_semitones": 3, "minimum_secondary_frequency_span_hz": 225,
    "relative_peak_tolerance": .018, "peak_tolerance_bounds_hz": [8, 30], "harmonic_exclusion_factor": 1.4,
    "minimum_combined_peak_weight_fraction": .55,
}


@lru_cache(maxsize=1)
def _frame_window():
    return np.hanning(FRAME)


def _peaks(frame):
    centered = frame - frame.mean()
    if np.sqrt(np.mean(centered**2)) < METHOD["minimum_frame_rms_s16"]:
        return np.empty(0), np.empty(0)
    spectrum = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(centered * _frame_window(), FFT)), 1e-6))
    peaks, properties = find_peaks(spectrum, prominence=METHOD["peak_prominence_db"], distance=3)
    frequencies = peaks * RATE / FFT
    eligible = ((frequencies >= METHOD["peak_band_hz"][0]) & (frequencies <= METHOD["peak_band_hz"][1])
                & (spectrum[peaks] >= spectrum.max() + METHOD["peak_relative_floor_db"]))
    peaks = peaks[eligible]
    if not len(peaks):
        return np.empty(0), np.empty(0)
    left, middle, right = spectrum[peaks - 1], spectrum[peaks], spectrum[peaks + 1]
    offset = np.clip(.5 * (left - right) / (left - 2 * middle + right), -.5, .5)
    frequencies = (peaks + offset) * RATE / FFT
    weights = np.exp2((middle - spectrum.max()) / 12) * np.minimum(properties["prominences"][eligible] / 16, 1)
    return frequencies, weights


def _harmonics(frequencies):
    ratio = frequencies[None, :] / PITCHES[:, None]
    orders = np.rint(ratio).astype(np.int32)
    error = np.abs(orders * PITCHES[:, None] - frequencies[None, :])
    tolerance = np.clip(METHOD["relative_peak_tolerance"] * frequencies, *METHOD["peak_tolerance_bounds_hz"])
    return orders, error, tolerance


def _supports(orders, fitted, weights):
    clipped = np.where(fitted, orders, 0)
    weighted = np.where(fitted, weights, 0.)
    strengths = np.zeros((len(PITCHES), METHOD["maximum_harmonic"] + 1))
    np.maximum.at(strengths, (np.arange(len(PITCHES))[:, None], clipped), weighted)
    # Favor candidates that explain low and high orders without rewarding missing low harmonics.
    hits = strengths[:, 1:] > 0
    required = METHOD["minimum_supported_harmonics"]
    coherent = (hits.sum(axis=1) >= required) & (hits[:, :7].sum(axis=1) >= 2)
    total = (strengths[:, 1:] / np.sqrt(np.arange(1, strengths.shape[1]))).sum(axis=1)
    return np.where(coherent, total, 0)


def pitch_pair(frame):
    frequencies, weights = _peaks(frame)
    if len(frequencies) < 6:
        return None
    orders, error, tolerance = _harmonics(frequencies)
    fitted = (orders >= 1) & (orders <= METHOD["maximum_harmonic"]) & (error <= tolerance)
    strengths = _supports(orders, fitted, weights)
    first = int(strengths.argmax())
    if strengths[first] == 0:
        return None
    residual = error[first] > METHOD["harmonic_exclusion_factor"] * tolerance
    second_scores = _supports(orders, fitted & residual, weights)
    distance = np.abs(12 * np.log2(PITCHES / PITCHES[first]))
    second_scores[distance < METHOD["minimum_pitch_separation_semitones"]] = 0.
    second = int(second_scores.argmax())
    separate = frequencies[fitted[second] & residual]
    explained = weights[fitted[first] | (fitted[second] & residual)].sum()
    if (second_scores[second] < METHOD["secondary_minimum_relative_energy"] * strengths[first]
            or len(separate) < METHOD["minimum_supported_harmonics"]
            or np.ptp(separate) < METHOD["minimum_secondary_frequency_span_hz"]
            or explained < METHOD["minimum_combined_peak_weight_fraction"] * weights.sum()):
        return None
    return tuple(sorted((float(PITCHES[first]), float(PITCHES[second]))))


def measure(raw):
    """Return half-open 20 ms grid spans. Coexisting harmonics do not count speakers."""
    if len(raw) != SAMPLES * 2:
        raise ValueError("Exact signed 16-bit three-second waveform required")
    pcm = np.frombuffer(raw, "<i2").astype(np.float64)
    spans = []
    start, previous, frames = None, None, 0
    for center in range(2 * HOP, SAMPLES - HOP, HOP):
        pair = pitch_pair(pcm[center - FRAME // 2:center + FRAME // 2])
        linked = pair is not None and previous is not None and all(
            abs(12 * np.log2(a / b)) <= METHOD["maximum_frame_pitch_change_semitones"]
            for a, b in zip(pair, previous, strict=True)
        )
        if not linked:
            if frames >= MIN_RUN:
                spans.append([start - HOP // 2, center - HOP // 2])
            frames, start = (0, None) if pair is None else (1, center)
        else:
            frames += 1
        previous = pair
    if frames >= MIN_RUN:
        spans.append([start - HOP // 2, center + HOP // 2])
    return spans
