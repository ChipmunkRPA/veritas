"""Replay the dependency-transfer study from scalar measurements without models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts import writing_dependency_transfer as study

COEFFICIENTS = {"scales", "weights"}
TOLERANCE = 1e-13


def review(packet):
    source = study.source
    source.require(packet["freeze"]["runner_sha256"] == source.digest(Path(study.__file__)),
                   "Measuring code differs from the frozen run")
    result = study.replay(packet)
    refitted = result.pop("models")
    recorded = packet["result"]["models"]
    source.require(refitted.keys() == recorded.keys(), "Recorded arms differ")
    for arm, fitted in refitted.items():
        frozen = recorded[arm]
        source.require({k: v for k, v in frozen.items() if k not in COEFFICIENTS}
                       == {k: v for k, v in fitted.items() if k not in COEFFICIENTS},
                       "Calibration maxima, points or fit coverage differ")
        for key in COEFFICIENTS:
            values = frozen[key]
            source.require(len(values) == len(fitted[key]) and
                           all(type(a) in (int, float) and math.isfinite(a) and abs(a - b) <= TOLERANCE
                               for a, b in zip(values, fitted[key], strict=True)), "Refitted coefficients differ")
    frozen = study.evaluate(packet["geometry"], packet["readings"], packet["primary"], recorded)
    published = {key: value for key, value in packet["result"].items() if key != "models"}
    source.require(source.serialize(result) == source.serialize(frozen) == source.serialize(published),
                   "Frozen decision or measurements differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = review(study.source.read(args.receipt))
    print(f"Replayed {len(result['decisions'])} writing files; advancement={result['passes']}.")


if __name__ == "__main__":
    main()
