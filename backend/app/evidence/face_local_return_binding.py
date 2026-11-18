"""Frozen appearance comparison for a local change with matching surroundings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

METHOD = "face-short-return/23"
DIMENSIONS = (640, 360)
FRAME_COUNT = 100
STEP = Fraction(1, 25)
SCHEDULE = tuple(range(0, FRAME_COUNT, 2))
FLANK = 4
CORE_MIN, CORE_MAX = 4, 22
SCENE_POINT = .60
CENTER = (0.16736602706727632, 0.04272703764367901)
SCALE = (0.3126370017530753, 0.2776335123675416)
COEFFICIENTS = (0.7585438543936421, -0.05627124105680274)
POINT = 0.04561063845899678


@dataclass(frozen=True)
class ScanGeometry:
    step: Fraction
    frame_count: int
    flank: int
    minimum: int
    maximum: int

    @property
    def schedule(self):
        return tuple(range(0, self.frame_count, 2))

    def support(self, start, end):
        if (type(start) is not int or type(end) is not int or start % 2 or end % 2 or start < 2 * self.flank
                or end + 2 * self.flank > self.frame_count or not 2 * self.minimum <= end - start <= 2 * self.maximum):
            raise ValueError("Local face-change interval differs from the supported native grid")
        return (list(range(start - 2 * self.flank, start, 2)), list(range(start, end, 2)),
                list(range(end, end + 2 * self.flank, 2)))


LANDSCAPE = ScanGeometry(STEP, FRAME_COUNT, FLANK, CORE_MIN, CORE_MAX)
PORTRAIT = ScanGeometry(Fraction(1, 30), 120, 5, 5, 26)


def response(features):
    return math.fsum((value - center) * weight / scale for value, center, weight, scale
                     in zip(features, CENTER, COEFFICIENTS, SCALE, strict=True))


def support(start, end):
    """Validate the half-open native range and the four sampled pictures on each flank."""
    return LANDSCAPE.support(start, end)
