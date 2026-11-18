# SPDX-License-Identifier: MIT
"""Recount an anonymous replaced-token writing experiment without a model."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.independent_locations import _matches

SCHEMA = "chipmunk.writing-replaced-tokens-study/1"
ARMS = ("high_replacement", "low_replacement")
ROLES = {"ordinary", "ordinary_edit", "edited"}
ADJUDICATIONS = {"retrospective_unadjudicated", "no_prior_meaning_caveat", "meaning_drift", "copied_successor"}


def exact_float(raw):
    value = float.fromhex(raw)
    if not math.isfinite(value) or value.hex() != raw:
        raise ValueError("Invalid frozen comparison score")
    return value


def spans(value):
    if any(len(pair) != 2 or any(type(offset) is not int for offset in pair) or not 0 <= pair[0] < pair[1]
           for pair in value):
        raise ValueError("Invalid half-open writing range")
    return value


def matched(truth, choices):
    return _matches(truth, choices, minimum_iou=.30, minimum_recall=.50)


def summarize(cases, points):
    numbers = Counter()
    controls = {arm: set() for arm in ARMS}
    candidate = {arm: Counter() for arm in ARMS}
    for row in cases:
        role, group = row["role"], row["group"]
        truth, eligible, incumbent = (spans(row[key]) for key in ("truth", "eligible", "incumbent"))
        if (role not in ROLES or len(truth) != int(role == "edited")
                or (row["adjudication"] in ADJUDICATIONS) != bool(truth)
                or set(row["peaks"]) != set(ARMS)
                or any(left[1] > right[0] for left, right in zip(eligible, eligible[1:]))):
            raise ValueError("Invalid saved case roles or ranges")
        prior_hits = matched(truth, incumbent)
        numbers.update(files=1, ordinary=int(not truth), labeled_edits=len(truth),
                       incumbent_locations=prior_hits, incumbent_ordinary_flags=int(bool(incumbent) and not truth))
        for arm, raw in row["peaks"].items():
            considered = []
            selected = []
            if bool(raw is not None) != (len(eligible) >= 5):
                raise ValueError("Peak availability differs from the minimum focus coverage")
            if raw is not None:
                index, score = raw
                if type(index) is not int or not 0 <= index < len(eligible):
                    raise ValueError("Peak does not refer to a measured source range")
                considered = [eligible[index]]
                if exact_float(score) > points[arm]:
                    selected = considered
            hit = matched(truth, selected)
            added_ordinary = not truth and bool(selected) and not incumbent
            candidate[arm].update(assessed=int(bool(considered)), reachable=matched(truth, eligible),
                                  top_overlaps=matched(truth, considered), located=hit,
                                  incremental_locations=hit - prior_hits if hit > prior_hits else 0,
                                  wrong_edit_flags=int(bool(truth and selected) and not hit),
                                  ordinary_flags=int(bool(selected) and not truth),
                                  ordinary_flags_added=int(added_ordinary))
            if added_ordinary:
                controls[arm].add(group)
    return dict(numbers) | {"candidates": {arm: dict(counts) | {"ordinary_source_groups_added": len(controls[arm])}
                                         for arm, counts in candidate.items()}}


def recount(receipt):
    if receipt["schema"] != SCHEMA or set(receipt["points"]) != set(ARMS):
        raise ValueError("Unknown replaced-token receipt")
    calibration = receipt["calibration"]
    cases = receipt["cases"]
    if len(calibration) < 8 or set(calibration).intersection(row["group"] for row in cases.values()):
        raise ValueError("Calibration groups cannot overlap assessment groups")
    points = {}
    for arm in ARMS:
        values = sorted(exact_float(row[arm]) for row in calibration.values())
        points[arm] = values[-2]
        if points[arm] != exact_float(receipt["points"][arm]):
            raise ValueError("Point differs from the frozen second-highest ordinary work maximum")
    panel_names = receipt["panels"]
    if (any(len(ids) != len(set(ids)) for ids in panel_names.values())
            or set().union(*map(set, panel_names.values())) != cases.keys()):
        raise ValueError("A panel is missing or duplicates a saved assessment file")
    panels = {name: summarize([cases[alias] for alias in ids], points) for name, ids in panel_names.items()}
    by_meaning = defaultdict(list)
    for case in cases.values():
        if case["role"] == "edited":
            by_meaning[case["adjudication"]].append(case)
    return {"calibration_groups": len(calibration), "unique_assessment": summarize(list(cases.values()), points),
            "panels": panels, "edited_by_adjudication": {name: summarize(rows, points)
                                                        for name, rows in sorted(by_meaning.items())}}


def replay(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        receipt = json.load(stream)
    result = recount(receipt)
    if result != receipt["summary"]:
        raise ValueError("Recount differs from the frozen aggregate results")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(replay(parser.parse_args().receipt), indent=2, sort_keys=True, allow_nan=False))
