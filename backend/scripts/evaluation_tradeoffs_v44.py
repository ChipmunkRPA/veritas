"""Describe observed model tradeoffs without automatically rejecting candidates.

This is research comparison arithmetic, not confidence fusion or deployment
admission. Shared source groups, calibration and test roles still matter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def observed_frontier(
    models: Mapping[str, Mapping[str, float]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> dict:
    """Keep every method; annotate point-estimate dominance on explicit axes.

    An increase in one error type is a tradeoff, not an automatic failure.
    Dominance here does not establish statistical or population dominance.
    """
    axes = (*maximize, *minimize)
    if not models or not axes or len(set(axes)) != len(axes):
        raise ValueError("Nonempty models and distinct comparison axes required")
    for values in models.values():
        if not all(
            name in values
            and type(values[name]) in (int, float)
            and math.isfinite(values[name])
            for name in axes
        ):
            raise ValueError("Complete finite metrics required; unchecked is not zero")

    def dominates(a, b):
        comparisons = [a[key] >= b[key] for key in maximize] + [a[key] <= b[key] for key in minimize]
        strict = [a[key] > b[key] for key in maximize] + [a[key] < b[key] for key in minimize]
        return all(comparisons) and any(strict)

    annotations = {
        name: {
            "observed_dominated_by": sorted(
                other for other, candidate in models.items()
                if other != name and dominates(candidate, values)
            ),
            "retained_in_comparison": True,
        }
        for name, values in models.items()
    }
    return {
        "models": annotations,
        "observed_frontier": sorted(
            name for name, annotation in annotations.items() if not annotation["observed_dominated_by"]
        ),
        "maximize": list(maximize),
        "minimize": list(minimize),
        "automatic_rejection": False,
        "serving_admitted": False,
        "statistical_superiority_established": False,
        "scope": "Observed point-estimate tradeoffs only. Keep component error types, coverage and complementary cases visible. Do not select on an independent final benchmark and reuse it as validation.",
    }


def positive_hit_overlap(baseline: Mapping[str, bool | None], candidate: Mapping[str, bool | None]) -> dict:
    """Measure complementary positive-case hits, never an oracle serving result.

    None means unchecked/inconclusive. It is not a missed or original case.
    'Either useful' does not mean the union of two region masks is useful.
    """
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("The same complete positive-case support is required")
    if any(value is not None and type(value) is not bool for value in (*baseline.values(), *candidate.values())):
        raise ValueError("Explicit hit, miss or unknown required")
    keys = set(baseline)
    return {
        "positive_cases": len(keys),
        "both_useful": sum(baseline[k] is True and candidate[k] is True for k in keys),
        "candidate_useful_baseline_missed": sum(candidate[k] is True and baseline[k] is False for k in keys),
        "baseline_useful_candidate_missed": sum(baseline[k] is True and candidate[k] is False for k in keys),
        "either_method_useful": sum(baseline[k] is True or candidate[k] is True for k in keys),
        "neither_useful_both_checked": sum(baseline[k] is False and candidate[k] is False for k in keys),
        "baseline_unknown": sum(baseline[k] is None for k in keys),
        "candidate_unknown": sum(candidate[k] is None for k in keys),
        "candidate_useful_baseline_unknown": sum(candidate[k] is True and baseline[k] is None for k in keys),
        "baseline_useful_candidate_unknown": sum(baseline[k] is True and candidate[k] is None for k in keys),
        "combined_serving_accuracy_measured": False,
        "whole_file_originality_established": False,
        "scope": "Positive-case complementarity, not a deployed union/consensus rule or authorship probability. Actual combined masks, false alarms and calibration must be evaluated separately.",
    }
