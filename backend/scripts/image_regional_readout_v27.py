"""Prospective source-group regional comparison; no neural inference or serving.

Response policies here are internal research settings, not customer confidence
thresholds. A clear map never establishes that a whole file is original.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

PIXELS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
AREAS = (0.0003, 0.001, 0.003, 0.01)
KINDS = ("ai", "original", "tone", "blur")
BOOTSTRAPS = 10_000
SEED = 20261108
MAX_FALSE_GROUP_DIFFERENCE_UPPER = 0.05


def require(value, message):
    if not value:
        raise ValueError(message)


def policies():
    return {f"p{pixel:.2f}-a{area:.5f}": {"pixel": pixel, "area": area} for pixel in PIXELS for area in AREAS}


def image_gate(raw, policy):
    values = np.asarray(raw)
    require(
        values.ndim == 2
        and values.size > 0
        and np.isfinite(values).all()
        and values.min() >= 0
        and values.max() <= 1,
        "Complete finite source-size sigmoid field required",
    )
    require(policy in policies().values(), "One predeclared response policy required")
    return int(np.count_nonzero(values > policy["pixel"])) >= math.ceil(policy["area"] * values.size)


def binomial_upper(k, n, *, alpha=0.025):
    """One-sided exact Clopper–Pearson upper bound by binomial-CDF inversion."""
    require(
        type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and 0 < alpha < 1,
        "Valid binomial count, denominator and tail required",
    )
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        p = (lo + hi) / 2
        cdf = math.fsum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def binomial_lower(k, n, *, alpha=0.025):
    return 1 - binomial_upper(n - k, n, alpha=alpha)


def measurement(row, policy):
    result = row["baseline"] if policy == "baseline" else row["candidate"][policy]
    require(
        isinstance(result, dict) and result.get("status") == "ok",
        "Failed, skipped or inconclusive checks cannot become clear-file observations",
    )
    metrics = result["metrics"]
    count = metrics["predicted_positive_pixels"]
    require(type(count) is int and count >= 0, "Recorded nonnegative selected-pixel count required")
    names = (
        "intersection",
        "false_positive_pixels",
        "false_negative_pixels",
        "pixels",
        "reference_positive_pixels",
    )
    require(
        all(type(metrics[k]) is int and metrics[k] >= 0 for k in names),
        "Integer source-pixel counts required",
    )
    tp, fp, fn, pixels, reference = (metrics[k] for k in names)
    require(
        pixels > 0 and tp + fp + fn <= pixels and count == tp + fp and reference == tp + fn,
        "Pixel counts do not describe one source-bound mask pair",
    )
    if row["kind"] == "ai":
        require(
            all(
                type(metrics[k]) in (int, float) and math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1
                for k in ("iou", "recall")
            ),
            "Finite edited-region IoU and recall required",
        )
        require(
            reference > 0
            and abs(metrics["iou"] - tp / (tp + fp + fn)) < 1e-12
            and abs(metrics["recall"] - tp / reference) < 1e-12,
            "Edited metrics must reproduce pixel counts",
        )
    else:
        require(reference == 0, "Non-generative controls require an all-zero AI truth mask")
    return metrics


def coverage(rows, policy_ids):
    """Retain unknown outcomes in the denominator without inferring originality."""
    outcomes = {}
    for policy in ("baseline", *policy_ids):
        states = Counter()
        for row in rows:
            result = row.get("baseline") if policy == "baseline" else row.get("candidate", {}).get(policy)
            status = result.get("status", "missing") if isinstance(result, dict) else "missing"
            require(
                status in {"ok", "failed", "unavailable", "skipped", "inconclusive", "missing"},
                "Unknown measurement status",
            )
            states[status] += 1
        outcomes[policy] = {
            "total": len(rows),
            "statuses": dict(states),
            "not_successfully_checked": len(rows) - states["ok"],
            "not_successfully_checked_fraction": (len(rows) - states["ok"]) / len(rows) if rows else None,
        }
    return {"outcomes": outcomes, "whole_file_originality_established": False}


def validate_rows(rows, *, role, groups, policy_ids):
    require(
        rows
        and all(
            r["role"] == role
            and r["format"] == rows[0]["format"]
            and r["format"] in {"native", "jpeg75"}
            and r["kind"] in KINDS
            for r in rows
        ),
        "One complete role/format with known source kinds required",
    )
    by_photo = defaultdict(list)
    by_component = defaultdict(set)
    for row in rows:
        by_photo[row["source_id"]].append(row)
        by_component[row["component"]].add(row["source_id"])
        measurement(row, "baseline")
        require(
            set(row["candidate"]) == set(policy_ids),
            "Every frozen candidate policy must be present; no outcome-dependent omissions",
        )
        for policy in policy_ids:
            measurement(row, policy)
    require(len(by_component) == groups, "All prescribed source components required")
    for photo, values in by_photo.items():
        require(
            len(values) == 4
            and {r["kind"] for r in values} == set(KINDS)
            and len({r["component"] for r in values}) == 1,
            f"Four complete, unique source-bound views required: {photo}",
        )
    return by_component


def group_summary(rows, policy):
    components = defaultdict(list)
    for row in rows:
        components[row["component"]].append(row)
    result = {}
    for key in sorted(components):
        values = components[key]
        ai = [measurement(r, policy) for r in values if r["kind"] == "ai"]
        require(ai, "Every component needs positive source evidence")
        negatives = {
            kind: any(
                measurement(r, policy)["predicted_positive_pixels"] > 0 for r in values if r["kind"] == kind
            )
            for kind in KINDS[1:]
        }
        result[key] = {
            "useful": math.fsum(float(m["iou"] >= 0.30 and m["recall"] >= 0.50) for m in ai) / len(ai),
            "iou": math.fsum(m["iou"] for m in ai) / len(ai),
            "false_groups": negatives,
            "any_false": any(negatives.values()),
        }
    return result


def totals(summary):
    n = len(summary)
    require(n > 0, "Nonempty group summary required")
    return {
        "groups": n,
        "useful_photo_fraction_mean_over_groups": math.fsum(r["useful"] for r in summary.values()) / n,
        "mean_iou_over_groups": math.fsum(r["iou"] for r in summary.values()) / n,
        "false_groups_by_kind": {
            kind: sum(r["false_groups"][kind] for r in summary.values()) for kind in KINDS[1:]
        },
        "any_negative_false_groups": sum(r["any_false"] for r in summary.values()),
    }


def calibrate(rows, *, groups=32):
    """Select one format's map settings on calibration groups alone."""
    grid = policies()
    validate_rows(rows, role="calibration", groups=groups, policy_ids=grid)
    baseline = totals(group_summary(rows, "baseline"))
    scored = []
    for name, policy in grid.items():
        candidate = totals(group_summary(rows, name))
        eligible = all(
            candidate["false_groups_by_kind"][kind] <= baseline["false_groups_by_kind"][kind]
            for kind in KINDS[1:]
        )
        scored.append({"id": name, "policy": policy, "eligible": eligible, "summary": candidate})
    admitted = [r for r in scored if r["eligible"]]
    # Tie breaks are fixed and conservative, not learned from reserved outcomes.
    ordered = sorted(
        admitted,
        key=lambda r: (
            -r["summary"]["useful_photo_fraction_mean_over_groups"],
            -r["summary"]["mean_iou_over_groups"],
            -r["policy"]["pixel"],
            -r["policy"]["area"],
        ),
    )
    return {
        "schema": "chipmunk.regional-map-calibration/27",
        "format": rows[0]["format"],
        "selected_policy": ordered[0]["id"] if ordered else None,
        "baseline": baseline,
        "policy_results": scored,
        "serving_admitted": False,
        "probability_calibration_performed": False,
        "scope": "Internal localization policy fitting. Raw responses and selected gates are not probabilities of authenticity or pixel authorship.",
    }


def compare(rows, policy, *, role, groups):
    require(
        role in {"point", "diagnostic", "confirmation"} and policy in policies(),
        "Frozen selected policy and explicit reserved role required",
    )
    validate_rows(rows, role=role, groups=groups, policy_ids=(policy,))
    baseline, candidate = group_summary(rows, "baseline"), group_summary(rows, policy)
    a, b = totals(baseline), totals(candidate)
    keys = sorted(baseline)
    differences = np.asarray([candidate[k]["useful"] - baseline[k]["useful"] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(keys), size=(BOOTSTRAPS, len(keys)))
    interval = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975], method="linear").tolist()
    newly_false = sum(candidate[k]["any_false"] and not baseline[k]["any_false"] for k in keys)
    recovered = sum(baseline[k]["any_false"] and not candidate[k]["any_false"] for k in keys)
    false_upper = binomial_upper(newly_false, len(keys)) - binomial_lower(recovered, len(keys))
    no_kind_increase = all(
        b["false_groups_by_kind"][kind] <= a["false_groups_by_kind"][kind] for kind in KINDS[1:]
    )
    gain = float(differences.mean())
    point_pass = gain > 0 and no_kind_increase
    confirmation_pass = (
        interval[0] > 0 and no_kind_increase and false_upper <= MAX_FALSE_GROUP_DIFFERENCE_UPPER
    )
    return {
        "schema": "chipmunk.regional-reserved-comparison/27",
        "role": role,
        "format": rows[0]["format"],
        "policy": policy,
        "baseline": a,
        "candidate": b,
        "paired_useful_fraction_difference": gain,
        "paired_group_bootstrap_95pct_interval": interval,
        "newly_false_groups": newly_false,
        "recovered_false_groups": recovered,
        "false_group_rate_difference_95pct_upper_bound": false_upper,
        "no_observed_false_group_increase_in_each_negative_kind": no_kind_increase,
        "stage_passed": point_pass
        if role == "point"
        else confirmation_pass
        if role == "confirmation"
        else None,
        "serving_admitted": False,
        "scope": "Pointwise per-format research comparison. Exact marginal discordance bounds use a union bound for the paired risk difference. Group definitions remain imperfect proxies for independent people; intervals are not simultaneous across formats or evidence of unseen-generator performance.",
    }
