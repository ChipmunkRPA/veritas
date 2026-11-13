"""Recalculate held writing locations and false alarms from the published score receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("paired", "direct", "margin")
PANELS = {"elife": {"human": 15, "splice": 15, "granite": 13, "olmo": 11},
          "plos": {"human": 10, "splice": 10, "granite": 10, "olmo": 7}}
VIEWS = ("full", "short")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(ranges, extent):
    previous = 0
    for start, end in ranges:
        if not previous <= start < end <= extent:
            return False
        previous = end
    return True


def expected_picks(values, point):
    top = sorted(((score, idx) for idx, score in enumerate(values) if score is not None), reverse=True)
    if not top or top[0][0] <= point or len(top) > 1 and top[0][0] == top[1][0]:
        return []
    return [top[0][1]]


def useful(target, location):
    if target is None or not location:
        return False
    start, end = target
    a, b = location
    shared = max(0, min(end, b) - max(start, a))
    return 10 * shared >= 3 * (end - start + b - a - shared) and 2 * shared >= end - start


def measures(row, points):
    source = row["focuses"]
    spans = [focus["nonspace"] for focus in source]
    require(spans and ordered(spans, row["nonspace_characters"])
            and ordered([focus["span"] for focus in source], row["characters"]), "Invalid source ranges")
    if row["truth"] is not None:
        require(0 <= row["target"] < len(source) and row["truth"] == source[row["target"]]["span"],
                "Edited characters differ from the known sentence position")
    target = spans[row["target"]] if row["truth"] is not None else None
    values = row["responses"]
    for arm, vector in values.items():
        require(len(vector) == len(source) and all(value is None or type(value) in (float, int) and math.isfinite(value)
                                                   for value in vector), "Invalid raw response")
        if arm != "incumbent":
            require(all((v is not None) == f["eligible"] for f, v in zip(source, vector, strict=True)),
                    "Eligibility and recorded response differ")
    native = [i for i, value in enumerate(values["incumbent"]) if value >= 0.5][:6]
    locations = [(spans[max(0, i - 1)][0], spans[min(i + 1, len(spans) - 1)][1]) for i in native]
    base = {"picked": native, "useful": any(useful(target, location) for location in locations)}
    result = {"incumbent": base}
    for arm in ARMS:
        selected = expected_picks(values[arm], points[arm])
        result[arm] = {"picked": selected, "useful": bool(selected and useful(target, spans[selected[0]]))}
    require(result == row["outcomes"], "Saved localization or direction differs")
    return result


def point_rows(receipt):
    threshold = {}
    for view in VIEWS:
        threshold[view] = {}
        rows = [item for item in receipt["point_controls"] if item["view"] == view]
        require(len(rows) == 20 and len({r["group"] for r in rows}) == 10 and
                {(r["group"], r["role"]) for r in rows} ==
                {(r["group"], role) for r in rows for role in ("human", "splice")}, "Ordinary point groups differ")
        for arm in ARMS:
            maxima = {r["group"]: -math.inf for r in rows}
            for row in rows:
                value = row["maxima"][arm]
                require(type(value) in (float, int) and math.isfinite(value), "Invalid ordinary point response")
                maxima[row["group"]] = max(maxima[row["group"]], value)
            threshold[view][arm] = sorted(maxima.values(), reverse=True)[1]
    require(threshold == receipt["points"], "Frozen operating points differ")
    return threshold


def review(receipt):
    require(receipt["schema"] == "chipmunk.writing-sentence-context-head/1", "Unknown study")
    points = point_rows(receipt)
    all_results = {}
    for view in VIEWS:
        for panel, roles in PANELS.items():
            rows = [row for row in receipt["held"] if row["view"] == view and row["panel"] == panel]
            require(len(rows) == sum(roles.values()) and len({(r["group"], r["role"]) for r in rows}) == len(rows)
                    and {role: sum(row["role"] == role for row in rows) for role in roles} == roles,
                    "Held cohort roster differs")
            measured = [(row, measures(row, points[view])) for row in rows]
            results = {}
            for arm in ("incumbent", *ARMS):
                false = {row["group"] for row, outcomes in measured
                         if row["truth"] is None and outcomes[arm]["picked"]}
                previous_false = {row["group"] for row, outcomes in measured
                                  if row["truth"] is None and outcomes["incumbent"]["picked"]}
                metrics = {"ordinary_groups_flagged": len(false), "new_ordinary_groups": len(false - previous_false),
                           "outcomes": {}}
                for role in roles:
                    group = [(row, outcomes) for row, outcomes in measured if row["role"] == role]
                    metrics["outcomes"][role] = {
                        "cases": len(group),
                        "flagged": sum(bool(outcomes[arm]["picked"]) for _, outcomes in group),
                        "useful": sum(outcomes[arm]["useful"] for _, outcomes in group),
                        "additional": sum(outcomes[arm]["useful"] and not outcomes["incumbent"]["useful"]
                                          for _, outcomes in group),
                        "without_eligible_focus": sum(not any(f["eligible"] for f in row["focuses"]) for row, _ in group),
                    }
                results[arm] = metrics
            all_results[f"{panel}_{view}"] = results
    require(len(receipt["held"]) == 182, "Total held case views differ")
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = review(json.loads(args.receipt.read_bytes()))
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Review failed. {error}") from None
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
