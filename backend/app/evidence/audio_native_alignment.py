# SPDX-License-Identifier: Apache-2.0
"""Compare quieter stereo sound after aligning channels on their native clock."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve, firwin

from app.evidence import audio_channel_alignment as existing
from app.evidence import audio_similarity
from app.evidence.audio_channel_alignment_binding import MAX_SHIFT, overlap
from app.evidence.audio_similarity_binding import MAX_SAMPLES, MIN_SAMPLES, SAMPLE_RATE

FACTOR = 3
NATIVE_RATE = SAMPLE_RATE * FACTOR
FILTER = firwin(63, 7100, fs=NATIVE_RATE)


def narrowband(values: np.ndarray) -> np.ndarray:
    """Preserve sample positions while rejecting frequencies above 16 kHz Nyquist."""
    kernel = FILTER if values.ndim == 1 else FILTER[:, None]
    return convolve(values.astype(np.float64), kernel, mode="same", method="direct")


def _rank(left: np.ndarray, right: np.ndarray, shift: int) -> float:
    start, stop = overlap(len(left), shift)
    left = left[start:stop:FACTOR]
    right = right[start + shift:stop + shift:FACTOR]
    dot = np.dot(left, right)
    return float(dot / max(1., np.sqrt(np.dot(left, left) * np.dot(right, right))))


def estimate(stereo: np.ndarray) -> existing.Alignment | None:
    # Check all three native offsets within each two-channel 16 kHz alignment.
    values = narrowband(stereo)
    left, right = values.T
    best, strongest = None, -1.
    for phase in range(FACTOR):
        decimated = np.column_stack((left[::FACTOR], right[phase::FACTOR]))
        fitted = existing.estimate(decimated)
        if fitted is None:
            continue
        shift = FACTOR * fitted.right_offset_samples + phase
        if abs(shift) > FACTOR * MAX_SHIFT:
            continue
        score = _rank(left, right, shift)
        if score > strongest:
            best = existing.Alignment(shift, fitted.right_gain_numerator, fitted.agreeing_windows, fitted.usable_windows)
            strongest = score
    return best


def search(stereo: np.ndarray):
    """Return evidence intervals in native (48 kHz) sample coordinates."""
    if (stereo.dtype != np.dtype("<i2") or stereo.ndim != 2 or stereo.shape[1] != 2 or len(stereo) % FACTOR
            or not MIN_SAMPLES * FACTOR <= len(stereo) <= MAX_SAMPLES * FACTOR):
        raise ValueError("Unsupported 48 kHz continuous stereo input")
    fitted = estimate(stereo)
    if fitted is None:
        return None, "insufficient_alignment", ()
    shift = fitted.right_offset_samples
    residual = existing.projection(stereo, fitted)
    values = np.clip(np.rint(narrowband(residual)[::FACTOR]), -32768, 32767).astype("<i2")
    result = audio_similarity.find_similar_spans(existing._similarity_input(values))
    if result.status != "measured":
        return fitted, result.reason, ()
    start, stop = overlap(len(stereo), shift)
    # Avoid any samples that included padding or an unpaired native sample.
    margin = len(FILTER) // 2
    spans = []
    for pair in result.spans:
        if pair.first[0] * FACTOR < start + margin or pair.second[1] * FACTOR > stop - margin:
            continue
        if np.median(existing.waveform_correlations(values, pair)) >= existing.MIN_WAVEFORM_CORRELATION:
            spans.append(tuple((lo * FACTOR, hi * FACTOR) for lo, hi in (pair.first, pair.second)))
    return fitted, None, tuple(spans)
