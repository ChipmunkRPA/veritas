# SPDX-License-Identifier: Apache-2.0
"""Compare frequency-band support for short soundtrack repeats."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.video import lipsync
from scripts import video_adjacent_sound_witness as prior

BANDS = ((220, 900), (900, 2100), (2100, 4100))
FILTERS = tuple(butter(3, band, btype="bandpass", fs=prior.SR, output="sos") for band in BANDS)
ARMS = ("existing", "three_band", "same_two_bands")


def proposals(wave):
    """Require repeated waveform support in a fixed pair of bands along the run."""
    inputs = np.asarray(wave, dtype=np.float64)
    filtered = [sosfiltfilt(kernel, inputs) for kernel in FILTERS]
    powers = [prior._prefix(band**2) for band in filtered]
    lags = sorted(set().union(*(prior._lags(band) for band in filtered)))
    pairs_by_arm = {arm: set() for arm in ARMS[1:]}
    for lag in lags:
        starts = np.arange(0, len(wave) - lag - prior.WINDOW + 1, prior.STEP)
        first, second, third = (prior._band_support(band, power, lag, starts)
                                for band, power in zip(filtered, powers, strict=True))
        two = (first & second, first & third, second & third)
        for name, masks in (("three_band", (first & second & third,)), ("same_two_bands", two)):
            for mask in masks:
                for a, b in lipsync._runs(mask):
                    count = prior.WINDOW + (b - a - 1) * prior.STEP
                    if max(prior.MIN_WITNESS, lag - prior.ADJACENCY) <= count <= lag + prior.STEP:
                        start = int(starts[a])
                        width = min(count, lag)
                        pairs_by_arm[name].add(((start, start + width), (start + lag, start + lag + width)))
    return {"existing": prior.adjacent_repeats(inputs)} | {
        arm: sorted(pairs) for arm, pairs in pairs_by_arm.items()}


def measure(wave, frames, patches):
    """Use the same speech and visual rules for all three audio proposals."""
    signal = wave.astype(np.float64) / 32768
    speech = lipsync.vad(signal).speech
    result = {}
    for arm, pairs in proposals(signal).items():
        accepted = []
        for pair in sorted(pairs, key=lambda item: (item[0][0] - item[0][1], item)):
            if min(np.mean(speech[a // prior.PICTURE_HOP:(b + prior.PICTURE_HOP - 1) // prior.PICTURE_HOP])
                   for a, b in pair) < .4:
                continue
            if any(prior._equivalent(pair, entry['ranges']) for entry in accepted):
                continue
            if evidence := prior.visual_support(pair, frames, patches):
                accepted.append({'ranges': pair, **evidence})
        result[arm] = accepted
    return result
