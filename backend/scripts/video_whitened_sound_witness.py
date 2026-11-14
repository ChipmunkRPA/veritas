# SPDX-License-Identifier: Apache-2.0
"""Compare repeat proposals with fixed filtering and with background downweighting."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.video import lipsync
from scripts import video_adjacent_sound_witness as prior

BANDS = ((180, 950), (950, 3500), (3500, 5900))
FILTERS = tuple(butter(3, band, fs=prior.SR, btype="bandpass", output="sos") for band in BANDS)
ARMS = ("prior", "fixed", "downweighted")
FRAME = 512
ANALYSIS = np.hanning(FRAME)
MIN_MATCH = .53
MIN_BAND = .38
MIN_MARGIN = .08
SEPARATE_LAG = 384


def downweight(wave):
    """Apply one bounded filter to the entire clip using a low-energy spectral estimate."""
    chunks = np.lib.stride_tricks.sliding_window_view(wave, FRAME)[::FRAME // 2]
    power = np.abs(np.fft.rfft(chunks * ANALYSIS)) ** 2
    floor = np.quantile(power, .30, axis=0)
    rate = np.fft.rfftfreq(FRAME, d=1 / prior.SR)
    typical = max(float(np.median(floor[(rate >= 180) & (rate <= 5900)])), 1e-10)
    weights = np.clip((typical / np.maximum(floor, 1e-10)) ** .35, .15, 1.)
    fft_size = len(wave) + 2 * FRAME
    bins = np.fft.rfftfreq(fft_size, d=1 / prior.SR)
    padded = np.pad(wave, FRAME, mode="reflect")
    return np.fft.irfft(np.fft.rfft(padded) * np.interp(bins, rate, weights), n=fft_size)[FRAME:-FRAME]


def correlations(band, power, lag, starts):
    left = power[starts + prior.WINDOW] - power[starts]
    right = power[starts + lag + prior.WINDOW] - power[starts + lag]
    cross = prior._prefix(band[:-lag] * band[lag:])
    product = cross[starts + prior.WINDOW] - cross[starts]
    active = np.minimum(left, right) >= prior.WINDOW * (prior.ACTIVE_RMS * .35) ** 2
    return np.divide(product, np.sqrt(np.maximum(1e-15, left * right)), out=np.zeros_like(product), where=active)


def propose(wave):
    bands = [sosfiltfilt(kernel, wave) for kernel in FILTERS]
    powers = [prior._prefix(band**2) for band in bands]
    full = sum(bands)
    lags = sorted(set().union(*(prior._lags(band) for band in (*bands, full))))
    starts = np.arange(0, len(wave) - prior.WINDOW + 1, prior.STEP)
    profiles = np.zeros((len(lags), len(starts)))
    for i, lag in enumerate(lags):
        valid = starts[starts + lag + prior.WINDOW <= len(wave)]
        bands_at_lag = [correlations(band, power, lag, valid) for band, power in zip(bands, powers, strict=True)]
        profiles[i, :len(valid)] = np.median(bands_at_lag, axis=0)
    proposals = {}
    for lag, scores in zip(lags, profiles, strict=True):
        rivals = [i for i, other in enumerate(lags) if abs(other - lag) >= SEPARATE_LAG]
        alternative = profiles[rivals].max(axis=0) if rivals else np.zeros_like(scores)
        core = (scores >= MIN_MATCH) & (scores - alternative >= MIN_MARGIN)
        for first, stop in lipsync._runs((scores >= MIN_BAND) & (scores - alternative >= 0)):
            length = prior.WINDOW + (stop - first - 1) * prior.STEP
            if (not np.any(core[first:stop])
                    or not max(prior.MIN_WITNESS, lag - prior.ADJACENCY) <= length <= lag + prior.STEP):
                continue
            start = int(starts[first])
            width = min(length, lag)
            pair = ((start, start + width), (start + lag, start + lag + width))
            proposals[pair] = round(float(scores[first:stop].max()), 6)
    return proposals


def measure(samples, frames, patches):
    wave = samples.astype(np.float64) / 32768
    activity = lipsync.vad(wave).speech
    proposals = {"prior": dict.fromkeys(prior.adjacent_repeats(wave)),
                 "fixed": propose(wave), "downweighted": propose(downweight(wave))}
    results = {}
    for name, pairs in proposals.items():
        records = []
        ordered = sorted(pairs, key=lambda pair: (-(pairs[pair] or 0), pair[0][0] - pair[0][1], pair))
        for pair in ordered:
            if min(np.mean(activity[a // prior.PICTURE_HOP:(b + prior.PICTURE_HOP - 1) // prior.PICTURE_HOP])
                   for a, b in pair) < .4:
                continue
            if any(prior._equivalent(pair, item["ranges"]) for item in records):
                continue
            if evidence := prior.visual_support(pair, frames, patches):
                records.append({"ranges": pair, "score": pairs[pair], **evidence})
        results[name] = records
    return results
