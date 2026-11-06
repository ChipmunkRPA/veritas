"""Research-only background continuity candidates; no AI-origin decisions.

This is deliberately not registered as a detector. It compares adjacent source
windows and returns exact sample coordinates for independent evaluation.
"""

from __future__ import annotations

import math

import numpy as np

RATE = 16000
SIDE_SAMPLES = 8000
STEP_SAMPLES = 4000
FFT_SAMPLES = 2048
HOP_SAMPLES = 512
BAND_EDGES = np.geomspace(60, 7800, 25)


def _js(left, right):
    middle = (left + right) / 2

    def divergence(x):
        return np.sum(
            np.where(x > 0, x * np.log2(np.maximum(x, 1e-30) / np.maximum(middle, 1e-30)), 0), axis=-1
        )

    return np.sqrt(np.clip((divergence(left) + divergence(right)) / 2, 0, 1))


def _spectra(values):
    starts = range(0, len(values) - FFT_SAMPLES + 1, HOP_SAMPLES)
    frames = np.asarray([values[a : a + FFT_SAMPLES] for a in starts], dtype=np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(FFT_SAMPLES), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(FFT_SAMPLES, 1 / RATE)
    bands = np.asarray(
        [
            power[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1)
            for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])
        ]
    ).T
    distributions = bands / np.maximum(bands.sum(axis=1, keepdims=True), 1e-30)
    profile = np.median(bands, axis=0)
    profile /= max(profile.sum(), 1e-30)
    instability = float(np.median(_js(distributions, profile)))
    return profile, instability


def candidates(pcm: np.ndarray) -> list[dict]:
    """Compare 0.5 s before/after each 0.25 s position; quieter sides abstain."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if pcm.ndim != 1 or not np.isfinite(pcm).all() or len(pcm) > RATE * 600:
        raise ValueError("Expected bounded finite mono PCM")
    rows = []
    for boundary in range(SIDE_SAMPLES, len(pcm) - SIDE_SAMPLES + 1, STEP_SAMPLES):
        sides = (pcm[boundary - SIDE_SAMPLES : boundary], pcm[boundary : boundary + SIDE_SAMPLES])
        rms = [20 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-6)) for x in sides]
        if min(rms) <= -60:
            continue
        (left, lvar), (right, rvar) = [_spectra(x) for x in sides]
        distance = float(_js(left, right))
        rows.append(
            {
                "start_sample": boundary - SIDE_SAMPLES,
                "end_sample": boundary + SIDE_SAMPLES,
                "boundary_sample": boundary,
                "boundary_s": boundary / RATE,
                "energy_change_db": abs(rms[1] - rms[0]),
                "spectral_distance": distance,
                "within_side_instability": max(lvar, rvar),
                "stationary_spectral_distance": distance if max(lvar, rvar) <= 0.20 else 0.0,
            }
        )
    return rows
