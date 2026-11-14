"""Reproduce the masked-word comparison across platform math libraries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts import writing_bidirectional_transfer as study

FIT_FIELDS = ("centers", "scales", "parameters")
COEFFICIENT_TOLERANCE = 1e-14


def review(data):
    source = study.source
    source.require(data["freeze"]["runner_sha256"] == source.digest(Path(study.__file__)),
                   "Historical measuring code differs from the frozen runner")
    recorded = data["points"]
    refit = study.calibrated(data["roster"], data["development"])
    source.require(recorded.keys() == refit.keys(), "Frozen comparison arms differ")
    for arm, observed in refit.items():
        frozen = recorded[arm]
        source.require({k: v for k, v in frozen.items() if k not in FIT_FIELDS}
                       == {k: v for k, v in observed.items() if k not in FIT_FIELDS},
                       "Frozen calibration maxima, operating point or fit coverage differ")
        for key in FIT_FIELDS:
            values = frozen[key]
            source.require(isinstance(values, list) and len(values) == len(observed[key])
                           and all(type(value) in (int, float) and math.isfinite(value)
                                   and abs(value - actual) <= COEFFICIENT_TOLERANCE
                                   for value, actual in zip(values, observed[key], strict=True)),
                           "Frozen coefficients differ beyond numerical tolerance")

    original = study.evaluate(data["roster"], data["held"], data["primary"], recorded)
    source.require(original == data["result"], "The recorded fit changes held decisions")
    # The historical reader verifies every recorded position and decision with
    # the independent refit. Retain the historical data and fitted points.
    return study.replay({**data, "points": refit})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    values = review(study.source.read(args.receipt))
    print(study.source.serialize({k: v for k, v in values.items() if k != "decisions"}), end="")


if __name__ == "__main__":
    main()
