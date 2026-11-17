# SPDX-License-Identifier: Apache-2.0
"""Recount video lip and sound agreement with fixed points and exact clocks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-av-local-agreement/1"
ARMS = ("shared", "phonetic")
CARRIERS = ("pcm", "aac48")
CONTROLS = ("original", "gain", "same_words", "global_early", "global_late")
ROLES = ("different_words", *CONTROLS)
BUDGETS = (0, 1, 2)


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def useful(span, truth):
    if span is None:
        return False
    overlap = max(0, min(span[1], truth[1]) - max(span[0], truth[0]))
    union = span[1] - span[0] + truth[1] - truth[0] - overlap
    return 5 * overlap >= 2 * union


def measure(rows, carrier, arm, point):
    counts = {role: Counter() for role in ROLES}
    for row in rows:
        for role, case in row["cases"][carrier].items():
            edit = role == "different_words"
            current, proposal = case["baseline"], case["proposals"][arm]
            observed = proposal["candidate"]
            shown = observed is not None and observed["score"] > point and not current
            match = edit and shown and useful(observed["span"], row["truth"])
            counts[role].update(
                files=1,
                past_flags=bool(current),
                past_correct=edit and useful(current, row["truth"]),
                added_flags=shown,
                added_correct=match,
                misplaced=edit and shown and not match,
                examined=proposal["status"] != "coverage",
                top_correct=edit and observed is not None and useful(observed["span"], row["truth"]),
                whole_track_delay=proposal["status"] == "whole_track_delay",
            )
    return {role: dict(value) for role, value in counts.items()}


def freeze(rows):
    options = {}
    choices = {}
    for carrier in CARRIERS:
        considered = []
        options[carrier] = {}
        for arm in ARMS:
            maxima = dict.fromkeys({row["speaker"] for row in rows}, 0.0)
            for row in rows:
                for role in CONTROLS:
                    case = row["cases"][carrier][role]
                    proposal = case["proposals"][arm]["candidate"]
                    if not case["baseline"] and proposal is not None:
                        key = row["speaker"]
                        maxima[key] = max(maxima[key], proposal["score"])
            sorted_scores = sorted(maxima.values(), reverse=True)
            if len(sorted_scores) < 5:
                raise ValueError("Need independent calibration speakers")
            options[carrier][arm] = {}
            for budget in BUDGETS:
                point = sorted_scores[budget]
                result = measure(rows, carrier, arm, point)
                control = sum(result[role]["added_flags"] for role in CONTROLS)
                edit = result["different_words"]
                rank = (edit["added_correct"], -control, -edit["misplaced"], -budget)
                chosen = {"arm": arm, "point": point, "budget": budget}
                considered.append((rank, chosen))
                options[carrier][arm][str(budget)] = chosen
        choices[carrier] = max(considered, key=lambda item: item[0])[1]
    return options, choices


def results(rows, options, choices):
    return {
        carrier: {"selected": measure(rows, carrier, choices[carrier]["arm"], choices[carrier]["point"])}
        | {
            arm: {level: measure(rows, carrier, arm, config["point"]) for level, config in cases.items()}
            for arm, cases in options[carrier].items()
        }
        for carrier in CARRIERS
    }


def _validate_clock(row, candidate):
    if candidate is None:
        return
    frames = row["native_times"]
    begin, end = candidate["picture_span"]
    a, b = candidate["span"]
    lag = candidate["lag_samples"]
    before, during, after = candidate["bracket_frames"]
    values = candidate["agreement"]
    if (
        not 0 <= before < begin <= during < end <= after < len(frames)
        or not 0 <= a < b <= row["samples"]
        or [a, b] != [frames[begin] + lag, frames[end] + lag]
        or not -1920 <= lag <= 1920
        or lag % 320
        or len(values) != 3
        or not all(math.isfinite(value) and -1.000001 <= value <= 1.000001 for value in values)
        or not 0 < candidate["score"] <= 2
        or abs(candidate["score"] - (min(values[0], values[2]) - values[1])) > 0.0000003
    ):
        raise ValueError("A comparison disagrees with its native frame and audio clock")


def _validate_rows(rows):
    if not rows or len({r["source_sha256"] for r in rows}) != len(rows):
        raise ValueError("Each source may supply only one comparison group")
    for row in rows:
        times = row["native_times"]
        pts = row["frame_pts"]
        tb = Fraction(row["time_base"])
        start, end = row["truth"]
        if (
            times != [round((t - pts[0]) * tb * 16000) for t in pts]
            or not 0 <= start < end <= row["samples"]
            or set(row["cases"]) != set(CARRIERS)
        ):
            raise ValueError("Invalid source media timing")
        for cases in row["cases"].values():
            if set(cases) != set(ROLES):
                raise ValueError("A matched edit or control delivery is missing")
            for case in cases.values():
                if set(case["proposals"]) != set(ARMS):
                    raise ValueError("A detector comparison is missing")
                for proposal in case["proposals"].values():
                    if proposal["status"] not in {
                        "coverage",
                        "whole_track_delay",
                        "no_local_dip",
                        "measured",
                    } or (proposal["candidate"] is not None) != (proposal["status"] == "measured"):
                        raise ValueError("Unassessed comparison has a proposed location")
                    _validate_clock(row, proposal["candidate"])


def review(path, selection=None):
    record = read(path)
    rows = record["rows"]
    if record["schema"] != SCHEMA:
        raise ValueError("Expected the local audio-visual agreement study")
    _validate_rows(rows)
    if selection is None:
        if record["freeze"] is not None or freeze(rows) != (record["options"], record["choices"]):
            raise ValueError("Calibration does not reproduce its frozen points")
    else:
        review(selection)
        frozen = read(selection)
        if (
            digest(selection) != record["freeze"]
            or any(record[k] != frozen[k] for k in ("options", "choices", "implementation", "models"))
            or {row["speaker"] for row in rows} & set(frozen["development_speakers"])
        ):
            raise ValueError("Held sources or model settings cross the freeze")
    counted = results(rows, record["options"], record["choices"])
    if counted != record["outcomes"]:
        raise ValueError("The delivered clip facts do not reproduce the study summary")
    return counted


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("result", type=Path)
    cli.add_argument("--freeze", type=Path)
    args = cli.parse_args()
    print(json.dumps(review(args.result, args.freeze), sort_keys=True))
