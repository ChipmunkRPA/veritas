"""Recount the anonymous writing position evaluation with the standard library."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from math import isclose, isfinite, sqrt
from pathlib import Path

SCHEMA = "chipmunk.writing-positional-attribution-evaluation/1"
METHODS = ("gradient", "head")


def _qualifies(first, second):
    shared = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    return (2 * shared >= second[1] - second[0]
            and 10 * shared >= 3 * (max(first[1], second[1]) - min(first[0], second[0])))


def _choice(row, method):
    reading = row[method]
    status = reading["status"]
    if status == "token_length":
        assert reading["tokens"] > 512
    if status == "completeness":
        assert method == "gradient" and reading["residual"] > .15 + .05 * abs(reading["difference"]) - 1e-8
    if status != "assessed":
        assert status in {"token_length", "completeness", "unmapped_sentence_or_token"}
        return None, 0.0

    scores = reading["scores"]
    units = row["sentences"]
    assert len(scores) == len(units) and all(map(isfinite, scores)) and reading["tokens"] <= 512
    if method == "head":
        sentence_contribution = sum(score * sqrt(unit[4]) for score, unit in zip(scores, units, strict=True))
        combined = sentence_contribution + reading["bias_entropy"] + reading["other_contribution"]
        assert abs(combined - reading["endpoint"]) <= 5.1e-5
    else:
        assert reading["residual"] <= .15 + .05 * abs(reading["difference"]) + 1e-8
    first, second, *_ = sorted(range(len(scores)), key=lambda i: -scores[i])
    gap = round(scores[first] - max(0.0, scores[second]), 8)
    return (first, gap) if gap > 0 else (None, 0.0)


def _range_hit(row, ranges):
    return any(_qualifies(bounds, truth) for bounds in ranges for truth in row["compact_truth"])


def _point(rows, method):
    maxima = defaultdict(float)
    incumbent = set()
    for row in rows:
        group = row["group"]
        _, gap = _choice(row, method)
        maxima[group] = max(maxima[group], gap)
        if row["incumbent"]:
            incumbent.add(group)
    ordered = sorted(maxima.values(), reverse=True)
    budget = len(incumbent) + 1
    return ordered[budget] if budget < len(ordered) else 0.0


def _held(rows, method, point):
    counts = Counter()
    status_counts = Counter()
    rank_positions = Counter()
    ordinary_groups, incumbent_groups, candidate_groups = set(), set(), set()
    for row in rows:
        index, gap = _choice(row, method)
        status_counts[row[method]["status"]] += 1
        shown = index is not None and gap > point
        if row["role"] == "controlled_rewrite":
            counts["rewrites"] += 1
            ranked = index is not None and _range_hit(row, [row["sentences"][index][2:4]])
            incumbent = _range_hit(row, row["incumbent"])
            counts["incumbent_exact"] += incumbent
            counts["incumbent_incorrect"] += bool(row["incumbent"]) and not incumbent
            counts["candidate_exact"] += ranked and shown
            counts["candidate_incorrect"] += shown and not ranked
            counts["added_exact"] += ranked and shown and not incumbent
            if index is None:
                rank_positions["no_rank"] += 1
            else:
                target, = row["compact_truth"]
                true_index = next(i for i, unit in enumerate(row["sentences"]) if unit[2:4] == target)
                distance = abs(index - true_index)
                rank_positions["target" if distance == 0 else "neighbor" if distance == 1 else "far"] += 1
        else:
            ordinary_groups.add(row["group"])
            counts["ordinary_files"] += 1
            if row["incumbent"]:
                incumbent_groups.add(row["group"])
            if shown:
                candidate_groups.add(row["group"])
    return {**counts, "statuses": dict(status_counts), "rank_positions": dict(rank_positions),
            "ordinary_groups": len(ordinary_groups), "incumbent_ordinary_alerts": len(incumbent_groups),
            "candidate_ordinary_alerts": len(candidate_groups), "added_ordinary_alerts": len(candidate_groups - incumbent_groups)}


def _paired(rows, method):
    starts = {row["group"]: row for row in rows if row["role"] == "original"}
    totals = defaultdict(Counter)
    for row in rows:
        if row["role"] == "original":
            continue
        start = starts[row["group"]]
        result = totals["synthetic" if row["role"] == "controlled_rewrite" else "archived_short" if row["cohort"] == "elife" else "ordinary"]
        if row[method]["status"] != "assessed" or start[method]["status"] != "assessed":
            result["unavailable"] += 1
            continue
        target, = row["compact_truth"]
        true_index = next(i for i, unit in enumerate(row["sentences"]) if unit[2:4] == target)
        differences = [after - before for after, before in zip(row[method]["scores"], start[method]["scores"], strict=True)]
        result["paired_assessed"] += 1
        result["target_rose"] += differences[true_index] > 0
        result["target_rose_most"] += differences[true_index] > max(0.0, max(
            score for i, score in enumerate(differences) if i != true_index))
    return {category: dict(result) for category, result in totals.items()}


def recount(record):
    assert record["schema"] == SCHEMA
    rows = record["rows"]
    assert len(rows) == record["files"] and len({row["id"] for row in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
        units = row["sentences"]
        assert len(units) >= 3 and all(0 <= a < b and 0 <= lo < hi and words > 0 for a, b, lo, hi, words in units)
        assert all(left[1] <= right[0] and left[3] <= right[2] for left, right in zip(units, units[1:]))
        assert row["truth"] == [[u[0], u[1]] for u in units if u[2:4] in row["compact_truth"]]
        for method in METHODS:
            _choice(row, method)
    assert len(groups) == record["groups"]
    for members in groups.values():
        assert len({(r["split"], r["cohort"]) for r in members}) == 1
        assert sum(row["role"] == "original" for row in members) == 1

    development = [r for r in rows if r["split"] == "development"]
    held = [r for r in rows if r["split"] == "held"]
    assert len(development) + len(held) == len(rows)
    assert all(row["role"] != "controlled_rewrite" for row in development)
    summary = {}
    primary = [row for row in held if row["cohort"] != "elife"]
    strata = {"primary": primary, "elife": [row for row in held if row["cohort"] == "elife"]}
    for key in ("cohort", "factual", "generator"):
        for value in sorted({r[key] for r in primary if r[key] is not None}):
            strata[f"{key}/{value}"] = [r for r in primary if r[key] == value]
    for method in METHODS:
        point = _point(development, method)
        assert isclose(point, record["internal_rank_gaps"][method], rel_tol=0, abs_tol=1e-8)
        summary[method] = {"development_status": dict(Counter(r[method]["status"] for r in development)),
                           "panel": {name: _held(cases, method, point) for name, cases in strata.items()},
                           "paired_diagnostic": _paired(held, method)}
    return {"files_recounted": len(rows), "source_groups": len(groups), "methods": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    result = recount(data)
    assert result == data["recount"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
