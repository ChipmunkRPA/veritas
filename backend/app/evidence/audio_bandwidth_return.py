"""Locate short, pronounced high-frequency losses in the submitted mono sound."""

from __future__ import annotations

import numpy as np

RATE = 16_000
CELL = 1280
FLANK = 5
LENGTHS = (7, 9, 11, 13, 15)
BANDS = ((500, 1900), (1900, 2700), (4100, 5500), (5500, 7400))
MIN_UPPER_DBFS = -63.
MAX_FLANK_SPREAD_DB = 8.
MAX_MID_DROP_DB = 5.
MINIMUM_DROP_DB = 12.
FLOOR = 1e-12


def _bands(pcm: np.ndarray) -> np.ndarray:
    # Nonoverlapping cells preserve the source clock. Two spectra per cell
    # leave the midpoint free of extrapolation or padding at the file edge.
    count = len(pcm) // CELL
    frames = pcm[:count * CELL].reshape(count, 2, CELL // 2).astype(np.float64)
    frames -= frames.mean(axis=-1, keepdims=True)
    window = np.hanning(CELL // 2)
    hz = np.fft.rfftfreq(CELL // 2, 1 / RATE)
    power = abs(np.fft.rfft(frames * window, axis=-1))**2 * (2 / ((CELL // 2) * np.sum(window**2)))
    energies = np.stack([power[..., (hz >= a) & (hz < b)].sum(axis=(1, 2)) / 2 for a, b in BANDS], axis=1)
    return 10 * np.log10(np.maximum(FLOOR, energies))


def _candidates(levels: np.ndarray) -> tuple[int, dict | None]:
    if len(levels) < 2 * FLANK + min(LENGTHS):
        return 0, None
    relative = levels[:, 1:] - levels[:, :1]
    flank_raw = np.lib.stride_tricks.sliding_window_view(levels, FLANK, axis=0)
    flank_rel = np.lib.stride_tricks.sliding_window_view(relative, FLANK, axis=0)
    lower = np.quantile(flank_rel, .25, axis=-1)
    typical = np.median(flank_rel, axis=-1)
    measurable = np.all(np.median(flank_raw, axis=-1)[:, -2:] >= MIN_UPPER_DBFS, axis=1)
    count, best = 0, None
    for length in LENGTHS:
        if len(levels) < 2 * FLANK + length:
            continue
        inside = np.lib.stride_tricks.sliding_window_view(relative, length, axis=0)
        starts = np.arange(FLANK, len(levels) - length - FLANK + 1)
        a, b = starts - FLANK, starts + length
        interior = np.quantile(inside[starts], .75, axis=-1)
        drops = np.minimum(lower[a], lower[b]) - interior
        agreement = np.max(abs(typical[a] - typical[b]), axis=1)
        admitted = (measurable[a] & measurable[b] & (agreement <= MAX_FLANK_SPREAD_DB)
                    & (abs(drops[:, 0]) <= MAX_MID_DROP_DB))
        for index in np.flatnonzero(admitted):
            count += 1
            drop = round(float(np.min(drops[index, 1:])), 6)
            start = int(starts[index]) * CELL
            end = start + length * CELL
            if best is None or (drop, -start, -end) > (best["drop_db"], -best["range"][0], -best["range"][1]):
                best = {"range": [start, end], "drop_db": drop,
                        "relative_band_db": np.round([lower[a[index]], interior[index], lower[b[index]]], 6).tolist()}
    return count, best


def locate(pcm: np.ndarray) -> tuple[int, dict | None]:
    samples = np.asarray(pcm)
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) > RATE * 30:
        raise ValueError("Expected at most thirty seconds of finite mono samples")
    if len(samples) < CELL * (2 * FLANK + min(LENGTHS)):
        return 0, None
    return _candidates(_bands(samples))
