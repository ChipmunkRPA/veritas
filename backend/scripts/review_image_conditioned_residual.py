"""Recount the brightness- and texture-matched image residual comparison."""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path

SCHEMA = "chipmunk.image-conditioned-residual-evaluation/1"
ARMS = ("luma_drop", "luma_and_color_departure")
QUALITIES = ("75", "90")
FILES = {"development": 240, "camera": 288, "ordinary_fill": 48, "scene_text": 196}
EDIT = {"sharp", "feather"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _location(value):
    a, b, common = (value[key] for key in ("selected", "truth", "intersection"))
    _require(
        all(type(number) is int for number in (a, b, common)) and 0 <= common <= min(a, b),
        "Invalid native area",
    )
    return bool(a and b and 2 * common >= b and 10 * common >= 3 * (a + b - common))


def _summary(cases, threshold):
    counts = Counter()
    hits, ordinary = set(), set()
    for row in cases:
        positive = row["role"] in EDIT
        old_flag = any(value and value["selected"] for value in row["current"])
        old_hit = any(value and _location(value) for value in row["current"])
        current = row["probe"]
        exact = _location(current)
        flag = current["score"] > threshold
        added_hit = positive and exact and flag and not old_hit
        added_control = not positive and flag and not old_flag
        counts.update(
            edited=positive,
            ordinary=not positive,
            existing_correct=positive and old_hit,
            existing_ordinary=not positive and old_flag,
            strongest_useful=positive and exact,
            added_correct=added_hit,
            misplaced=positive and flag and not exact,
            added_ordinary=added_control,
            added_original=row["role"] == "ordinary" and added_control,
            selected_no_area=current["selected"] == 0,
        )
        if added_hit:
            hits.add(row["source_group"])
        if added_control:
            ordinary.add(row["source_group"])
    return dict(counts, added_correct_sources=len(hits), added_ordinary_sources=len(ordinary))


def recount(record):
    _require(record["schema"] == SCHEMA, "Unrecognized study")
    _require(
        set(record["commitments"]) == {"method", "development", "held", "admission", "native_recount"}
        and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in record["commitments"].values()),
        "Missing reproduced-source commitments",
    )
    banks = record["banks"]
    _require({key: len(rows) for key, rows in banks.items()} == FILES, "Incomplete plan")
    for rows in banks.values():
        for row in rows:
            for arm in ARMS:
                score = row["arms"][arm]["score"]
                _require(
                    type(score) in (int, float) and isfinite(score) and 0 <= score <= 5,
                    "Invalid residual departure",
                )

    result = defaultdict(dict)
    for quality in QUALITIES:
        for arm in ARMS:
            dev = [row for row in banks["development"] if row["quality"] == int(quality)]
            maxima = {}
            for row in dev:
                if row["role"] in EDIT or any(value and value["selected"] for value in row["current"]):
                    continue
                source, score = row["source_group"], row["arms"][arm]["score"]
                maxima[source] = max(maxima.get(source, 0), score)
            _require(len(maxima) >= 8, "Insufficient development groups")
            point = sorted(maxima.values())[-2]
            _require(record["points"][quality][arm] == point, "Frozen operating point changed")
            for scope, cohort in (
                ("development", dev),
                ("camera_and_fills", banks["camera"] + banks["ordinary_fill"]),
                ("scene_text", banks["scene_text"]),
            ):
                rows = (
                    {**row, "probe": row["arms"][arm]} for row in cohort if row["quality"] == int(quality)
                )
                result[scope].setdefault(quality, {})[arm] = _summary(rows, point)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    actual = recount(record)
    _require(actual == record["results"], "Recount differs from the published result")
    print(json.dumps(actual, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
