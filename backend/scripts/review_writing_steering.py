# SPDX-License-Identifier: Apache-2.0
"""Independent model-free numeric recount, without loading the observer."""

import argparse
import re
from collections import Counter, defaultdict
from fractions import Fraction
from gzip import decompress
from json import dumps, loads
from math import ceil
from pathlib import Path
from statistics import median

ARMS = ("unsteered", "steered")


def coverage(proposal, truth):
    a, b = proposal
    c, d = truth
    shared = max(0, min(b, d) - max(a, c))
    return 2 * shared >= d - c and 10 * shared >= 3 * (b - a + d - c - shared)


def count_locations(proposals, labels):
    pairs = {}
    graph = {
        i: [j for j, label in enumerate(labels) if coverage(r, label)]
        for i, r in enumerate(proposals)
    }

    def seek(i, seen):
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                if j not in pairs or seek(pairs[j], seen):
                    pairs[j] = i
                    return True
        return False

    return sum(seek(i, set()) for i in graph)


def reread(row, arm):
    values = row["readings"][arm]
    assert len(values) == len(row["compact_bounds"]) == len(row["source_bounds"])
    assert values[0] is None and values[-1] is None
    realized = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(realized) < 5:
        return "input_outside_scope", None
    if len(wins := [i for i, v in realized if v == max([v for _, v in realized])]) != 1:
        return "tied", None
    i = wins[0]
    difference = round(values[i] - median(v for j, v in realized if j != i))
    if any(
        round(v - median(u for k, u in realized if k != j)) == difference
        for j, v in realized
        if j != i
    ):
        return "tied", None
    return "available", {"index": i, "score": difference}


def verify_peaks(rows):
    for row in rows:
        for arm in ARMS:
            assert reread(row, arm)[1] == row["peaks"][arm]
        for index, (start, stop) in enumerate(row["source_bounds"]):
            compact = row["compact_bounds"][index]
            assert 0 < compact[1] - compact[0] <= stop - start
            if index:
                assert start >= row["source_bounds"][index - 1][1]
                assert compact[0] >= row["compact_bounds"][index - 1][1]


def recount_points(rows, points):
    verify_peaks(rows)
    for arm in ARMS:
        top = {}
        for row in rows:
            assert row["role"] != "edited"
            win = row["peaks"][arm]
            if win:
                key = row["group"]
                top[key] = max(top.get(key, -(2**64)), win["score"])
        sorted_values = sorted(top.values())
        n = len(sorted_values)
        point = sorted_values[ceil(0.95 * n) - 1] if n else None
        assert points[arm] == {
            "score": point,
            "eligible_groups": n,
            "above": sum(s > point for s in sorted_values) if n else 0,
        }


def evaluate(rows, points, baseline_available):
    by = defaultdict(Counter)
    fp_groups = {arm: set() for arm in ARMS}
    new_groups = {arm: set() for arm in ARMS}
    correct_groups = {arm: set() for arm in ARMS}
    learned_extra_groups = set()
    learned_extra_editors = set()
    comparator = "unsteered"
    experiment = "steered"
    for row in rows:
        group = row["group"]
        role = row["role"]
        label = row["truth_compact"]
        edits = bool(label)
        assert (role == "edited") == edits and len(label) == len(row["truth"])
        c = [by["edited"], by[row["generator"]]] if edits else [by[role], by["ordinary_pooled"]]
        flags = {}
        hits = {}
        for arm in ARMS:
            status, win = reread(row, arm)
            proposed = row["compact_bounds"][win["index"]] if win else None
            flags[arm] = bool(win and win["score"] > points[arm]["score"])
            ranks = count_locations([proposed], label) if proposed else 0
            hits[arm] = ranks if flags[arm] else 0
            for accumulator in c:
                accumulator.update(
                    {
                        f"{arm}_ranked": bool(win),
                        f"{arm}_flags": flags[arm],
                        f"{arm}_correct": hits[arm],
                        f"{arm}_top_correct": ranks,
                        f"{arm}_{status}": 1,
                        f"{arm}_wrong_location": bool(
                            edits and flags[arm] and not hits[arm]
                        ),
                    }
                )
            if hits[arm]:
                correct_groups[arm].add(group)
            if flags[arm] and not edits:
                fp_groups[arm].add(group)
        for accumulator in c:
            accumulator.update(files=1, labels=len(label))
            accumulator.update(
                steering_adds_correct=bool(hits[experiment] and not hits[comparator]),
                steering_loses_correct=bool(hits[comparator] and not hits[experiment]),
                steering_new_flag=bool(flags[experiment] and not flags[comparator]),
                steering_flag_cleared=bool(flags[comparator] and not flags[experiment]),
            )
        if hits[experiment] and not hits[comparator]:
            learned_extra_groups.add(group)
            learned_extra_editors.add(row["generator"])
        if not edits and flags[experiment] and not flags[comparator]:
            new_groups[experiment].add(group)
        if baseline_available:
            basematches = count_locations(row["baseline"], label)
            for accumulator in c:
                accumulator.update(
                    current_flags=bool(row["baseline"]), current_correct=basematches
                )
                for arm in ARMS:
                    region = (
                        row["compact_bounds"][row["peaks"][arm]["index"]]
                        if flags[arm]
                        else None
                    )
                    appended = bool(
                        region
                        and all(
                            region[1] <= a or b <= region[0] for a, b in row["baseline"]
                        )
                    )
                    add = row["baseline"] + [region] if appended else row["baseline"]
                    combined = count_locations(add, label)
                    accumulator.update(
                        {
                            f"{arm}_combined_correct": combined,
                            f"{arm}_new_current_correct": combined - basematches,
                            f"{arm}_combined_flags": bool(add),
                            f"{arm}_extra_wrong": bool(
                                edits and appended and combined == basematches
                            ),
                        }
                    )
    return {
        "baseline_available": baseline_available,
        "counts": {label: dict(count) for label, count in by.items()},
        "groups": {
            arm: {
                "ordinary_flagged": len(fp_groups[arm]),
                "correct": len(correct_groups[arm]),
            }
            for arm in ARMS
        },
        "steering_added_correct_groups": len(learned_extra_groups),
        "steering_editors": len(learned_extra_editors),
        "steering_added_ordinary_groups": len(new_groups[experiment]),
    }


RECEIPT_SCHEMA = "chipmunk.writing-steering-intervention/1"
ROW_KEYS = {
    "baseline",
    "compact_bounds",
    "generator",
    "group",
    "native_sha256",
    "peaks",
    "readings",
    "role",
    "source_bounds",
    "truth",
    "truth_compact",
    "type",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(rows):
    reachable, expectation = 0, Fraction()
    for row in rows:
        if row["role"] != "edited":
            continue
        available = [
            r
            for r, value in zip(
                row["compact_bounds"], row["readings"]["unsteered"], strict=True
            )
            if value is not None
        ]
        labels = row["truth_compact"]
        reachable += sum(any(coverage(r, label) for r in available) for label in labels)
        if available:
            expectation += Fraction(
                sum(any(coverage(r, label) for label in labels) for r in available),
                len(available),
            )
    return reachable, float(expectation)


def verify(receipt):
    require(receipt["schema"] == RECEIPT_SCHEMA, "Unexpected receipt schema")
    require(
        set(receipt["panels"]) == {"calibration", "fresh", "transfer"}, "Missing panel"
    )
    seen_hashes = set()
    seen_groups = set()
    for name, rows in receipt["panels"].items():
        group_ids = {row["group"] for row in rows}
        require(not seen_groups & group_ids, "A work appears in more than one panel")
        seen_groups.update(group_ids)
        require(
            len(rows) == receipt["freeze"]["files"][name]["cases"]
            and len(group_ids) == receipt["freeze"]["files"][name]["groups"],
            "Panel membership differs from freeze",
        )
        for row in rows:
            require(set(row) == ROW_KEYS, "Unexpected field in numeric panel")
            require(
                re.fullmatch(r"W[0-9]{3}", row["group"]), "Non-anonymous source label"
            )
            source_digest = row["native_sha256"]
            require(
                re.fullmatch(r"[a-f0-9]{64}", source_digest)
                and source_digest not in seen_hashes,
                "Duplicate or malformed source digest",
            )
            seen_hashes.add(source_digest)
            require(
                row["role"]
                in {"untouched", "ordinary_edit", "natural_revision", "edited"},
                "Invalid source role",
            )
            require(
                set(row["readings"]) == set(ARMS) and set(row["peaks"]) == set(ARMS),
                "Malformed comparison arms",
            )
            require(
                all(
                    type(v) is int or v is None
                    for a in ARMS
                    for v in row["readings"][a]
                ),
                "Invalid scalar measurement",
            )
        verify_peaks(rows)
    for name in ("fresh", "transfer"):
        reachable, uniform = geometry(receipt["panels"][name])
        audit = receipt["geometry_audit"]
        require(
            audit[f"{name}_reachable"] == reachable
            and audit[f"{name}_expected_uniform_top"] == uniform,
            "Eligibility geometry differs from the numeric panel",
        )
    recount_points(receipt["panels"]["calibration"], receipt["points"])
    findings = {
        name: evaluate(
            receipt["panels"][name],
            receipt["points"],
            baseline_available=name == "transfer",
        )
        for name in ("fresh", "transfer")
    }
    require(
        findings == receipt["outcomes"],
        "Claimed outcomes differ from the numeric readings",
    )
    return {"points": receipt["points"], "results": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    require(__debug__, "Run without Python optimization to enable all checks")
    receipt = loads(decompress(args.receipt.read_bytes()))
    print(dumps(verify(receipt), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
