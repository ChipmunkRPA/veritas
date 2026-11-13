"""Recompute the formant localization decisions from the public scalar receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median

ARMS = ("narrowing", "two_order")
SHA = re.compile(r"[0-9a-f]{64}")
TOTAL, HOP, EXTENT = 96_000, 240, 9_600
BACKGROUND = {"background", "background_only"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def synthetic(row):
    return row["arm"].startswith("synthetic")


def score_and_range(row, arm):
    if arm == "incumbent":
        return row[arm]
    best = row["measures"][arm]["best"]
    return None if best is None else best[:2]


def useful(found, truth):
    start, end = found
    a, b = truth
    overlap = max(0, min(end, b) - max(start, a))
    return 2 * overlap >= b - a and 10 * overlap >= 3 * (max(end, b) - min(start, a))


def selection_points(rows):
    points = {}
    for arm in ARMS:
        controls = [result[0] for row in rows if not synthetic(row) and (result := score_and_range(row, arm)) is not None]
        require(controls, "No assessed development controls")
        points[arm] = max(controls)
    return points


def _metrics(rows, arm, point):
    counts = Counter()
    useful_ids, flagged_groups, found_families = set(), set(), Counter()
    for row in rows:
        positive = synthetic(row)
        counts["synthetic" if positive else "ordinary"] += 1
        result = score_and_range(row, arm)
        if result is None:
            counts["synthetic_unassessed" if positive else "ordinary_unassessed"] += 1
            continue
        score, span = result
        hit = positive and useful(span, row["truth"])
        counts["useful_before_point"] += int(hit)
        if score <= point:
            continue
        if hit:
            counts["useful"] += 1
            useful_ids.add(tuple(row["id"]))
            found_families[row["family"]] += 1
        elif positive:
            counts["misplaced"] += 1
        else:
            counts["ordinary_flags"] += 1
            counts["isolated_background_flags"] += int(row["arm"] in BACKGROUND)
            flagged_groups.add(row["group"])
    summary = dict(counts) | {"ordinary_groups_flagged": len(flagged_groups), "useful_families": dict(found_families)}
    return summary, useful_ids, flagged_groups


def _equal_budget(rows, arm):
    groups = {}
    for row in rows:
        if not synthetic(row) and (result := score_and_range(row, arm)) is not None:
            group = row["group"]
            groups[group] = max(result[0], groups.get(group, -math.inf))
    values = sorted(groups.values(), reverse=True)
    return {str(budget): _metrics(rows, arm, values[budget])[0].get("useful", 0) for budget in range(min(5, len(values)))}


def report(held, points, incumbent_points):
    cohorts = {}
    for row in held:
        key = row["role"], str(row["level"])
        cohorts.setdefault(key, []).append(row)
    output = {}
    for (role, level), rows in cohorts.items():
        cutoffs = points | ({"incumbent": incumbent_points[role]} if role in incumbent_points else {})
        metrics = {arm: _metrics(rows, arm, point) for arm, point in cutoffs.items()}
        result = {arm: summary for arm, (summary, _, _) in metrics.items()}
        result["equal_false_group_useful"] = {arm: _equal_budget(rows, arm) for arm in cutoffs}
        if "incumbent" in metrics:
            _, old_ids, old_flags = metrics["incumbent"]
            for arm in ARMS:
                _, new_ids, new_flags = metrics[arm]
                added = new_ids - old_ids
                result[arm]["added_useful"] = len(added)
                result[arm]["lost_useful"] = len(old_ids - new_ids)
                result[arm]["added_useful_families"] = sorted({row["family"] for row in rows if tuple(row["id"]) in added})
                result[arm]["added_ordinary_groups"] = len(new_flags - old_flags)
                result[arm]["lost_ordinary_groups"] = len(old_flags - new_flags)
        output.setdefault(role, {})[level] = result
    replication = output["replication"]["-5"]
    output["gate_pass"] = {
        arm: (replication[arm]["added_useful"] >= 2 and len(replication[arm]["added_useful_families"]) >= 2
              and replication[arm]["added_ordinary_groups"] <= 1 and not replication[arm].get("isolated_background_flags", 0)
              and all(output["transfer"][level][arm].get("useful", 0) >= 2
                      and output["transfer"][level][arm]["ordinary_groups_flagged"] <= 1 for level in ("-5", "-10")))
        for arm in ARMS
    }
    return output


def valid_range(bounds):
    first, last = bounds
    return all(type(tick) is int for tick in bounds) and 0 <= first < last <= TOTAL


def review(public):
    freeze, development, held = (public[key] for key in ("freeze", "development", "held"))
    rows = development["rows"] + held
    counts = Counter(row["role"] for row in rows)
    require(counts == {"development": 54, "assessment": 54, "replication": 52, "transfer": 72}, "Delivery extent differs")
    require(len({tuple(row["id"]) for row in rows}) == len(rows) == freeze["rows"], "Duplicate or missing delivery")
    require(all(row["role"] == "development" for row in development["rows"]), "Assessment input entered fitting")
    require(freeze["schema"] == "chipmunk.audio-formant-local-transfer/1" and SHA.fullmatch(freeze["runner_sha256"])
            and all(SHA.fullmatch(pin) for pin in list(freeze["input_pins"].values()) + freeze["input_loaders"]), "Invalid method pins")
    require(hashlib.sha256(canonical([{key: value for key, value in row.items() if key != "measures"} for row in rows])).hexdigest()
            == freeze["rows_sha256"], "Delivery bindings differ")
    require(hashlib.sha256(canonical(development)).hexdigest() == public["fitting_sha256"], "Development seal differs")
    repeated = {}
    for row in rows:
        require(SHA.fullmatch(row["sha256"]), "Invalid native source commitment")
        require(not synthetic(row) or valid_range(row["truth"]), "Invalid authored edit interval")
        require(repeated.setdefault(row["sha256"], row["measures"]) == row["measures"], "Identical input produced different output")
        if incumbent := row["incumbent"]:
            require(math.isfinite(incumbent[0]) and valid_range(incumbent[1]), "Invalid incumbent source range")
        require(set(row["measures"]) == set(ARMS), "Incomplete set of measurements")
        for arm in ARMS:
            values = row["measures"][arm]
            tracked, eligible, best = (values[key] for key in ("tracked", "eligible_ranges", "best"))
            require(0 <= tracked <= 399 and 0 <= eligible <= 77 and bool(eligible) == (best is not None), "Invalid frame coverage")
            if best is None:
                continue
            score, bounds, coverage, medians = best
            first, last = bounds
            require(valid_range(bounds) and 6720 <= first <= 79680 and (first - 6720) % (4 * HOP) == 0 and last - first == EXTENT,
                    "Candidate range differs from the original clock")
            require(all(minimum <= n <= maximum for n, minimum, maximum in zip(coverage, (6, 10, 6), (24, 40, 24), strict=True))
                    and sum(coverage) <= tracked, "Range does not have enough tracked frames")
            require(len(medians) == 3 and all(len(part) == (1 if arm == ARMS[0] else 3) for part in medians), "Wrong descriptor count")
            recomputed = median(max(0., min(left - mid, right - mid)) for left, mid, right in zip(*medians, strict=True))
            require(math.isfinite(score) and abs(score - recomputed) <= 5.1e-8, "Reported score differs from its acoustic support")
    require(selection_points(development["rows"]) == development["points"], "Development points changed")
    result = report(held, development["points"], freeze["incumbent_points"])
    require(result == public["summary"], "Summary differs from scored source ranges")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = review(json.loads(args.receipt.read_bytes()))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
