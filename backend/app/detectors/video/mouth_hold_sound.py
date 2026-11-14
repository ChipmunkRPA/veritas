# SPDX-License-Identifier: Apache-2.0
"""Measure directly observable periodicity in the mono soundtrack of a source-bound mouth hold."""

from __future__ import annotations

import numpy as np

from app.evidence import mouth_hold_sound as binding


def measure(pcm, mapping, review, time_base):
    clock = binding.bounds(review, mapping, time_base)
    if clock is None or len(pcm) != mapping["samples"]:
        return None
    start, stop, first, last = clock
    hop = binding.HOP
    starts = binding.block_starts(start, stop)
    measured = len(starts)
    if measured < binding.MIN_BLOCKS:
        return None
    periodic = 0
    lags = np.arange(35, 221)
    for i in starts:
        values = np.asarray(pcm[i:i + 2 * hop], dtype=np.float64)
        signal = values - values.mean()
        if np.sqrt(np.mean(signal ** 2)) < .007:
            continue
        # Mean-centered periodicity for 73–457 Hz, with local peaks to reject slowly changing backgrounds.
        lagged = np.correlate(signal, signal, "full")[len(signal) - 1:][35:221]
        squared = np.r_[0., np.cumsum(signal ** 2)]
        norm = np.sqrt(squared[len(signal) - lags] * np.maximum(0., squared[-1] - squared[lags]))
        values = lagged / (1e-12 + norm)
        if np.any((values[1:-1] >= .78) & (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])):
            periodic += 1
    return {"schema": binding.SCHEMA, "samples": [start, stop], "source_pts": [first, last],
            "time_base": f"1/{binding.RATE}", "blocks": measured, "periodic_blocks": periodic,
            "sustained_periodicity": 3 * periodic >= 2 * measured}


def source_assets(pcm, mapping, review, time_base, *, time_axis):
    """Reproduce the waveform and sound clip from the committed source samples."""
    from app.detectors.audio.decode import wav_bytes
    from app.evidence.audio_packets import mouth_hold_waveform

    sound = review["sound_context"]
    if measure(pcm, mapping, review, time_base) != sound:
        raise ValueError("The soundtrack does not reproduce the measured mouth-hold interval")
    start, stop = sound["samples"]
    return {binding.WAV: wav_bytes(pcm[start:stop]),
            binding.WAVEFORM: mouth_hold_waveform(pcm, sound, time_axis=time_axis)}
