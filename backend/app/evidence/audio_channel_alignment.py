"""Estimate small stereo offsets before looking for repeated quieter sound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.evidence import audio_similarity
from app.evidence.audio_channel_alignment_binding import (
    ALIGNMENT_WINDOW,
    GAIN_DENOMINATOR,
    MAX_SHIFT,
    MIN_GAIN_CHANGE,
    MIN_WINDOWS,
    overlap,
)
from app.evidence.audio_similarity_binding import HOP, MAX_SAMPLES, MIN_SAMPLES, WINDOW

MIN_CORRELATION = 0.86
MIN_PEAK_MARGIN = 0.015
MIN_RMS = 16
MIN_WAVEFORM_CORRELATION = 0.82


@dataclass(frozen=True)
class Alignment:
    right_offset_samples: int
    right_gain_numerator: int
    agreeing_windows: int
    usable_windows: int


def estimate(stereo: np.ndarray) -> Alignment | None:
    """Fit the common component on active short windows of the stereo recording."""
    samples = stereo.astype(np.float64)
    emphasized = samples[1:] - .9 * samples[:-1]
    left, right = emphasized.T
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    count = (len(left) - 2 * MAX_SHIFT) // ALIGNMENT_WINDOW
    starts = MAX_SHIFT + np.arange(count) * ALIGNMENT_WINDOW
    indices = starts[:, None] + np.arange(ALIGNMENT_WINDOW)
    base = left[indices]
    base_energy = np.einsum("ij,ij->i", base, base)
    scores, gains = [], []
    for shift in shifts:
        paired = right[indices + shift]
        energy = np.einsum("ij,ij->i", paired, paired)
        dot = np.einsum("ij,ij->i", base, paired)
        good = np.minimum(base_energy, energy) >= ALIGNMENT_WINDOW * MIN_RMS**2
        scores.append(np.divide(dot, np.sqrt(base_energy * energy), out=np.zeros(count), where=good))
        gains.append(np.sqrt(np.divide(energy, base_energy, out=np.zeros(count), where=good)))
    scores = np.stack(scores, axis=1)
    best = np.argmax(scores, axis=1)
    peak = scores[np.arange(count), best]
    rival = np.where(np.abs(shifts[None, :] - shifts[best, None]) > 1, scores, -1).max(axis=1)
    usable = (peak >= MIN_CORRELATION) & (peak - rival >= MIN_PEAK_MARGIN)
    if np.count_nonzero(usable) < MIN_WINDOWS:
        return None
    majority = np.bincount(best[usable], minlength=len(shifts)).argmax()
    support = usable & (best == majority)
    if support.sum() * 3 < usable.sum() * 2 or support.sum() < MIN_WINDOWS:
        return None
    supported_gains = gains[majority][support]
    gain = np.median(supported_gains)
    if not .50 <= gain <= 2.0 or np.median(np.abs(supported_gains - gain)) > .06 * gain:
        return None
    return Alignment(int(shifts[majority]), round(float(gain) * GAIN_DENOMINATOR), int(support.sum()), int(usable.sum()))


def projection(stereo: np.ndarray, alignment: Alignment) -> np.ndarray:
    """Use the left channel's sample clock and only real samples in the overlap."""
    values = stereo.astype(np.int32)
    shift, gain = alignment.right_offset_samples, alignment.right_gain_numerator
    first, end = overlap(len(values), shift)
    result = np.zeros(len(values), dtype="<i2")
    result[first:end] = ((gain * values[first:end, 0] - GAIN_DENOMINATOR * values[first + shift:end + shift, 1])
                         // (gain + GAIN_DENOMINATOR))
    return result


def waveform_correlations(values, pair):
    features = []
    for low, end in (pair.first, pair.second):
        raw = values[low:end].astype(np.float64)
        emphasized = raw[1:] - .95 * raw[:-1]
        windows = np.lib.stride_tricks.sliding_window_view(emphasized, WINDOW - 1)[::HOP]
        features.append(windows - windows.mean(axis=1, keepdims=True))
    a, b = features
    ea, eb = np.einsum("ij,ij->i", a, a), np.einsum("ij,ij->i", b, b)
    denominator = np.sqrt(ea * eb)
    return np.divide(np.einsum("ij,ij->i", a, b), denominator, out=np.zeros(len(a)), where=denominator > 0)


def _similarity_input(values):
    # Permit quiet residuals to enter the existing search without changing the audio evidence.
    peak = int(np.max(np.abs(values.astype(np.int32))))
    rms = float(values.std())
    gain = max(1, min(12, 32767 // max(peak, 1), int(128 / max(rms, 1))))
    return (values * gain).astype("<i2").tobytes()


def search(stereo: np.ndarray):
    if stereo.ndim != 2 or stereo.shape[1] != 2 or not MIN_SAMPLES <= len(stereo) <= MAX_SAMPLES:
        raise ValueError("Unsupported continuous stereo input")
    alignment = estimate(stereo)
    if alignment is None:
        return None, "insufficient_alignment", ()
    shift, gain = alignment.right_offset_samples, alignment.right_gain_numerator
    if shift == 0 and abs(gain - GAIN_DENOMINATOR) < MIN_GAIN_CHANGE:
        return alignment, "channel_difference_sufficient", ()
    values = projection(stereo, alignment)
    result = audio_similarity.find_similar_spans(_similarity_input(values))
    if result.status != "measured":
        return alignment, result.reason, ()
    spans = []
    low, high = overlap(len(stereo), shift)
    for span in result.spans:
        if span.first[0] < low or span.second[1] > high:
            continue
        if np.median(waveform_correlations(values, span)) >= MIN_WAVEFORM_CORRELATION:
            spans.append(span)
    return alignment, None, tuple(spans)
