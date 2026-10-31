"""The one rounding rule of a number that is printed (BUILD_SPEC N-10b; code re-verification R2-06).

The server printed a score with Python's `round`, which goes to the even neighbour at a half, and the page with JavaScript's
`Math.round`, which goes up: 0.965 was "score 96 / 100" in the reason and the report and "Score 97 / 100" on the page. Both
sides now use `floor(value + 0.5)`: `half_up` here, `halfUp` in `frontend/src/lib/format.ts`. Both suites read the vectors of
`backend/tests/fixtures/rounding_vectors.json`.
"""
from __future__ import annotations

import math


def half_up(value: float) -> int:
    """The nearest whole number, a half upwards."""
    return math.floor(value + 0.5)
