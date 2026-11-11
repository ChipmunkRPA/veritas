"""Recompute the public face-swap pooling comparison from anonymized scores.

The replay verifies arithmetic. Native input and inference verification require
the source recordings, model weights and private frame commitments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ARMS = ("reference_median", "candidate_median", "candidate_top_two", "paired_max_median")


def summarize(scores):
    if len(scores) != 6 or any(x is not None and (type(x) not in (int, float) or not math.isfinite(x)) for x in scores):
        raise ValueError("A score row needs six finite or missing values")
    measured = sorted(x for x in scores if x is not None)
    if len(measured) < 4:
        return None, None
    return median(measured), (measured[-1] + measured[-2]) / 2


def percentile90(values):
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


def scores(row):
    current, _ = summarize(row["reference"])
    candidate, high = summarize(row["candidate"])
    return current, candidate, high


def joint(current, candidate, scales):
    if current is None or candidate is None:
        return None
    return max((value - lo) / (hi - lo) for value, (lo, hi) in zip((current, candidate), scales, strict=True))


def replay(data):
    if data.get("schema") != "chipmunk.whole-clip-face-pooling-scalars/1" or len(data["development_originals"]) != 10:
        raise ValueError("Expected the published scalar comparison with ten development originals")
    controls = [scores(row) for row in data["development_originals"]]
    if any(v is None for row in controls for v in row):
        raise ValueError("Every calibration control must be assessed in both arms")
    distributions = list(zip(*controls, strict=True))
    cutoffs = [percentile90(values) for values in distributions]
    scales = [(median(values), cutoff) for values, cutoff in zip(distributions[:2], cutoffs[:2], strict=True)]
    if any(high <= low for low, high in scales):
        raise ValueError("A reference scale is constant")
    cutoffs.append(percentile90([joint(a, b, scales) for a, b, _ in controls]))
    counts = defaultdict(lambda: {name: Counter(cases=0, flagged=0, below=0, unassessed=0) for name in ARMS})
    transitions = defaultdict(Counter)
    groups = defaultdict(list)
    for row in data["evaluation"]:
        groups[row["group"]].append((row["cohort"], row["family"]))
        current, candidate, high = scores(row)
        key = f"{row['cohort']}/{row['family']}"
        outcomes = []
        for name, value, threshold in zip(ARMS, (current, candidate, high, joint(current, candidate, scales)), cutoffs, strict=True):
            outcome = "unassessed" if value is None else "flagged" if value > threshold else "below"
            outcomes.append(outcome)
            counts[key][name].update(("cases", outcome))
        transitions[key][f"{outcomes[0]} to {outcomes[-1]}"] += 1
    expected_families = {"original_videos", "inswapper", "uniface"}
    if len(groups) != 20 or any(len(rows) != 3 or len({cohort for cohort, _ in rows}) != 1
                                or {family for _, family in rows} != expected_families for rows in groups.values()):
        raise ValueError("The published source groups need one original and both matched swaps")
    return {"points": dict(zip(ARMS, cutoffs, strict=True)), "results": dict(counts), "paired_transitions": dict(transitions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalars", type=Path)
    parser.add_argument("aggregate", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.aggregate.read_bytes())
    payload = args.scalars.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected.get("portable_scalar_file_sha256"):
        parser.exit(1, "The scalar file differs from its published checksum.\n")
    actual = replay(json.loads(payload))
    if (actual["points"] != expected["frozen_points"]["points"] or actual["results"] != expected["results"]
            or actual["paired_transitions"] != expected["paired_transitions"]):
        parser.exit(1, "The independent arithmetic differs from the published counts or operating points.\n")
    print("Reproduced all four operating points, six delivery/family groups and their paired counts.")


if __name__ == "__main__":
    main()
