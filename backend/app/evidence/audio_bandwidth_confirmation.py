"""Measure a local drop and return in two upper-frequency bands."""

from __future__ import annotations

import numpy as np

from app.evidence.audio_bandwidth_return import BANDS, CELL, FLANK, LENGTHS, RATE, _bands

POINT_DB = 16.
METHOD = {
    "rate_hz": RATE, "cell_samples": CELL, "flank_cells": FLANK, "range_cells": list(LENGTHS),
    "bands_hz": [list(band) for band in BANDS], "point_db": POINT_DB,
    "minimum_upper_flank_dbfs": -80., "maximum_upper_flank_difference_db": 12.,
    "maximum_middle_change_db": 7., "flank_percentile": 50, "interval_percentile": 65,
}


def locate(pcm: np.ndarray) -> tuple[int, dict | None]:
    samples = np.asarray(pcm)
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) > 30 * RATE:
        raise ValueError("Expected at most thirty seconds of finite mono samples")
    if len(samples) < CELL * (2 * FLANK + min(LENGTHS)):
        return 0, None
    levels = _bands(samples)
    relative = levels[:, 1:] - levels[:, :1]
    windows = np.lib.stride_tricks.sliding_window_view
    flank_upper = np.median(windows(levels[:, 2:], FLANK, axis=0), axis=-1)
    flank_balance = np.median(windows(relative, FLANK, axis=0), axis=-1)
    enough_sound = np.all(flank_upper >= METHOD["minimum_upper_flank_dbfs"], axis=1)
    count, best = 0, None
    for length in LENGTHS:
        if len(levels) < 2 * FLANK + length:
            continue
        starts = np.arange(FLANK, len(levels) - length - FLANK + 1)
        before, after = starts - FLANK, starts + length
        left, right = flank_balance[before], flank_balance[after]
        inside = np.quantile(windows(relative, length, axis=0)[starts], .65, axis=-1)
        agreement = np.max(abs(left[:, 1:] - right[:, 1:]), axis=1)
        drops = np.minimum(left, right) - inside
        keep = (enough_sound[before] & enough_sound[after] & (agreement <= METHOD["maximum_upper_flank_difference_db"])
                & (abs(drops[:, 0]) <= METHOD["maximum_middle_change_db"]))
        count += int(keep.sum())
        if not keep.any():
            continue
        values = np.where(keep, np.round(np.min(drops[:, 1:], axis=1), 6), -np.inf)
        index = int(np.argmax(values))
        drop, start = float(values[index]), int(starts[index]) * CELL
        end = start + length * CELL
        if best is None or (drop, -start, -end) > (best["drop_db"], -best["range"][0], -best["range"][1]):
            best = {"range": [start, end], "drop_db": drop,
                    "relative_band_db": np.round([left[index], inside[index], right[index]], 6).tolist()}
    return count, best
