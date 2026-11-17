# SPDX-License-Identifier: Apache-2.0
"""Freeze a cut-aware operating point from ordinary montage maxima."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.evidence.video_lip_agreement import POINTS
from scripts.video_lip_multishot_panel import CARRIERS, digest, read, recount, save


def applied(rows, points):
    return [
        row | {"matches": [reading["proposal"]["span"] for reading in (row["readings"] or ())
                          if reading["proposal"] and reading["proposal"]["score_units"] > points[row["carrier"]]]}
        for row in rows
    ]


def freeze(development, output):
    source = read(development)
    rows = source["rows"]
    ordinary = {carrier: defaultdict(int) for carrier in CARRIERS}
    for row in rows:
        if row["role"] in {"original", "gain"} and row["assessed"]:
            group = row["group"]
            measured = ordinary[row["carrier"]]
            peak = max((reading["proposal"]["score_units"] for reading in row["readings"] if reading["proposal"]), default=0)
            measured[group] = max(peak, measured[group])
    legacy = {carrier: POINTS[carrier][0] for carrier in CARRIERS}
    points = {}
    for carrier, control in ordinary.items():
        if len(control) < 20:
            raise ValueError("Too few complete ordinary montage groups for calibration")
        maxima = sorted(control.values())
        rank = (9 * len(maxima) + 9) // 10
        points[carrier] = max(legacy[carrier], maxima[rank - 1])
    result = {"development_sha256": digest(development), "points": points, "prior_points": legacy, "ordinary_maxima": ordinary,
              "development_speakers": sorted({row[name] for row in rows for name in ("speaker", "lead_speaker")}),
              "earlier_points": recount(applied(rows, legacy)), "shot_points": recount(applied(rows, points))}
    save(output, result)
    print(json.dumps(result, sort_keys=True))


def compare(held, frozen):
    source, lock = read(held), read(frozen)
    rows = source["rows"]
    if {row[name] for row in rows for name in ("speaker", "lead_speaker")} & set(lock["development_speakers"]):
        raise ValueError("A calibration speaker entered the held montage comparison")
    results = {key: recount(applied(rows, lock[field])) for key, field in (("single_shot_points", "prior_points"), ("shot_points", "points"))}
    if source["counts"] != results["shot_points"]:
        raise ValueError("The measured operating point differs from the frozen shot-specific points")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    first, later = modes.add_parser("freeze"), modes.add_parser("compare")
    for command, names in ((first, ("development", "output")), (later, ("held", "frozen"))):
        for name in names:
            command.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    (freeze if args.pop("action") == "freeze" else compare)(**args)
