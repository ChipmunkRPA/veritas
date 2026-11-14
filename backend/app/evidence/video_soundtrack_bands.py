# SPDX-License-Identifier: Apache-2.0
"""Corroborate soundtrack delay peaks across separate frequency bands."""
from __future__ import annotations

from itertools import combinations

from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import SAMPLE_RATE
from app.evidence import video_reference_soundtrack as timing
from app.evidence.video_soundtrack_pair_binding import BAND_CORRELATION, BANDS

MIN_MARGIN = .09
MIN_RMS = .0005
FILTERS = tuple(butter(3, band, btype="bandpass", fs=SAMPLE_RATE, output="sos") for band in BANDS)


def matches(primary, secondary):
    """A window qualifies when two band searches agree within one millisecond."""
    lanes = [timing._matches(sosfiltfilt(filt, primary), sosfiltfilt(filt, secondary), window=timing.SHORT_WINDOW,
                             step=timing.SHORT_STEP, min_rms=MIN_RMS, min_correlation=BAND_CORRELATION,
                             min_margin=MIN_MARGIN) for filt in FILTERS]
    for rows in zip(*lanes, strict=True):
        start = rows[0][0]
        agreed = [(min(a[2], b[2]), max(a[2], b[2]), a[1] if a[2] >= b[2] else b[1])
                  for a, b in combinations(rows, 2)
                  if a[1] is not None and b[1] is not None and abs(a[1]-b[1]) <= timing.STABLE]
        if agreed:
            strength, _, lag = max(agreed)
            yield start, lag, strength
        else:
            yield start, None, None
