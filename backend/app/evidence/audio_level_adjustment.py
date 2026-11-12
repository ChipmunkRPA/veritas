"""Exact compatibility with a constant volume adjustment in native mono PCM."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE

SUPPORT_SAMPLES = SAMPLE_RATE // 2
MIN_SAMPLES = 2 * SAMPLE_RATE
MIN_ACTIVE_SAMPLES = 128
ACTIVE_AMPLITUDE = 512
GAIN_LIMITS = (Fraction(1, 4), Fraction(4))


@dataclass(frozen=True)
class LevelAdjustment:
    bounds: tuple[Fraction, Fraction]
    support_bounds: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    changed: np.ndarray
    residual: np.ndarray


def _rounded(samples: np.ndarray, gain: Fraction) -> np.ndarray:
    quotient, remainder = np.divmod(samples * gain.numerator, gain.denominator)
    twice_remainder = 2 * remainder
    return quotient + ((twice_remainder > gain.denominator)
                       | ((twice_remainder == gain.denominator) & (quotient % 2 == 1)))


def _bounds(earlier: np.ndarray, submitted: np.ndarray) -> tuple[Fraction, Fraction] | None:
    nonzero = earlier != 0
    if np.any(submitted[~nonzero] != 0) or not nonzero.any():
        return None
    denominator = 2 * np.abs(earlier[nonzero])
    center = 2 * submitted[nonzero] * np.sign(earlier[nonzero])
    lower_index = int(np.argmax((center - 1) / denominator))
    upper_index = int(np.argmin((center + 1) / denominator))
    lower = max(Fraction(int(center[lower_index] - 1), int(denominator[lower_index])), GAIN_LIMITS[0])
    upper = min(Fraction(int(center[upper_index] + 1), int(denominator[upper_index])), GAIN_LIMITS[1])
    if lower > upper:
        return None
    # Verify the extrema against every integer constraint. Floating point
    # selects indices only and cannot enlarge the admitted interval.
    if (np.any(lower.numerator * denominator < (center - 1) * lower.denominator)
            or np.any(upper.numerator * denominator > (center + 1) * upper.denominator)):
        return None
    if lower == upper and not np.array_equal(_rounded(earlier, lower), submitted):
        return None
    return lower, upper


def compare(earlier: np.ndarray, submitted: np.ndarray) -> LevelAdjustment | None:
    """Find samples incompatible with every gain allowed by both end intervals."""
    if len(earlier) < MIN_SAMPLES:
        return None
    original = (earlier * 32768).astype(np.int64)
    observed = (submitted * 32768).astype(np.int64)
    maximum = int(np.max(np.abs(original)))
    if max(maximum, int(np.max(np.abs(observed)))) >= 32767:
        return None
    head = slice(0, SUPPORT_SAMPLES)
    tail = slice(-SUPPORT_SAMPLES, None)
    for interval in (head, tail):
        if np.count_nonzero(np.abs(original[interval]) >= ACTIVE_AMPLITUDE) < MIN_ACTIVE_SAMPLES:
            return None
    head_bounds = _bounds(original[head], observed[head])
    tail_bounds = _bounds(original[tail], observed[tail])
    if head_bounds is None or tail_bounds is None:
        return None
    lower = max(head_bounds[0], tail_bounds[0])
    upper = min(head_bounds[1], tail_bounds[1])
    if lower > upper or lower <= 1 <= upper or maximum * upper >= Fraction(65533, 2):
        return None

    # For a given gain, an observed integer can match any value in its
    # half-sample interval. Retain only disjoint intervals, with exact products.
    original_abs = np.abs(original)
    center = 2 * np.where(original < 0, -observed, observed)
    changed = ((2 * original_abs * upper.numerator < (center - 1) * upper.denominator)
               | (2 * original_abs * lower.numerator > (center + 1) * lower.denominator))
    if (lower == upper and any(not np.array_equal(_rounded(original[part], lower), observed[part])
                               for part in (head, tail))):
        return None
    if not changed.any() and _bounds(original, observed) is None:
        # Different samples could individually fit different gains.
        return None
    residual = (observed - _rounded(original, (lower + upper) / 2)) / 32768
    return LevelAdjustment((lower, upper), (head_bounds, tail_bounds), changed, residual)
