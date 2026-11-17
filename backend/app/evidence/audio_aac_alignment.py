# SPDX-License-Identifier: Apache-2.0
"""Compare quieter stereo residuals after lossy audio coding."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from app.evidence import audio_channel_alignment, audio_native_alignment
from app.evidence.audio_channel_alignment_binding import AAC_EDGE_MARGIN, NATIVE_FACTOR, overlap
from app.evidence.audio_similarity import _lag_peaks, _prefix, _window_stats
from app.evidence.audio_similarity_binding import HOP, MAX_PAIRS, MIN_REPEAT_SAMPLES, SAMPLE_RATE, WINDOW

BANDS = ((160, 450), (450, 920), (920, 1600), (1600, 3000))
FILTERS = tuple(butter(3, band, fs=SAMPLE_RATE, btype="bandpass", output="sos") for band in BANDS)
MIN_CORRELATION = .76
LAGS_PER_VIEW = 8
MIN_BAND_RMS = 2
NEARBY_PAIR = 4800


def _nominate_lags(bands):
    nominees = set()
    for band in bands:
        # Equalize local energy for the lag search so that speech coding noise
        # in a loud window does not dominate the quieter portions of the clip.
        energy = np.maximum(0., uniform_filter1d(band * band, 1281))
        normalized = 128 * band / np.maximum(MIN_BAND_RMS, np.sqrt(energy))
        for view in (24 * band, normalized):
            lags, scores = _lag_peaks(view[:-1])
            nominees.update(int(lag) for lag in lags[np.lexsort((lags, -scores))[:LAGS_PER_VIEW]])
    return sorted(nominees)


def _window_scores(stats, starts, lag):
    correlations = []
    for band, total, energy in stats:
        first_sum, first_energy = _window_stats(total, energy, starts)
        second_sum, second_energy = _window_stats(total, energy, starts + lag)
        products = _prefix(band[:-lag] * band[lag:])
        covariance = products[starts + WINDOW] - products[starts] - first_sum * second_sum / WINDOW
        active = np.minimum(first_energy, second_energy) >= WINDOW * MIN_BAND_RMS**2
        correlations.append(np.divide(covariance, np.sqrt(first_energy * second_energy),
                                      out=np.zeros(len(starts)), where=active))
    # The second strongest band must agree at each supported position.
    return np.sort(np.stack(correlations), axis=0)[-2]


def _locate(values, low, high):
    bands = [sosfiltfilt(filt, values) for filt in FILTERS]
    if sum(band.std() >= MIN_BAND_RMS for band in bands) < 2:
        return "insufficient_active_bands", ()
    stats = [(band, _prefix(band), _prefix(band * band)) for band in bands]
    candidates = []
    for lag in _nominate_lags(bands):
        starts = np.arange(0, len(values) - lag - WINDOW + 1, HOP)
        scores = _window_scores(stats, starts, lag)
        edges = np.flatnonzero(np.diff(np.r_[False, scores >= MIN_CORRELATION, False]))
        for first, stop in zip(edges[::2], edges[1::2], strict=True):
            length = min(WINDOW + (int(stop - first) - 1) * HOP, lag)
            start = int(starts[first])
            pair = tuple((side * NATIVE_FACTOR, (side + length) * NATIVE_FACTOR) for side in (start, start + lag))
            if length >= MIN_REPEAT_SAMPLES and low + AAC_EDGE_MARGIN <= pair[0][0] and pair[1][1] <= high - AAC_EDGE_MARGIN:
                candidates.append((float(np.mean(scores[first:stop])), pair))

    pairs = []
    for _, pair in sorted(candidates, reverse=True):
        if any(all(abs(a[0] - b[0]) < NEARBY_PAIR for a, b in zip(pair, old, strict=True)) for old in pairs):
            continue
        pairs.append(pair)
        if len(pairs) > MAX_PAIRS:
            return "span_budget", ()
    return None, tuple(pairs)


def search(stereo: np.ndarray):
    """Return pairs of supported evidence intervals in 48 kHz samples."""
    # The fit compares three decimation phases. Keep their lengths equal
    # when an AAC edit list presents a partial final three-sample step.
    complete = len(stereo) - len(stereo) % NATIVE_FACTOR
    fitted = audio_native_alignment.estimate(stereo[:complete])
    if fitted is None:
        return None, "insufficient_alignment", ()
    residual = audio_channel_alignment.projection(stereo, fitted)
    values = audio_native_alignment.narrowband(residual)[::NATIVE_FACTOR]
    reason, pairs = _locate(values, *overlap(len(stereo), fitted.right_offset_samples))
    return fitted, reason, pairs
