# SPDX-License-Identifier: Apache-2.0
"""Freeze and count the two local observations needed for a reported interval."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from app.bench import video_av_local_agreement as bench
from app.evidence import video_lip_agreement as agreement

SCHEMA = "chipmunk.video-av-corroboration/1"


def save(path, receipt):
    with path.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(receipt, allow_nan=False, sort_keys=True).encode(), mtime=0))


def freeze(selection, held, grid, output):
    values = bench.read(selection)
    for carrier, points in agreement.POINTS.items():
        for name, point in zip(bench.ARMS, points, strict=True):
            if round(values["options"][carrier][name]["0"]["point"] * agreement.UNITS) != point:
                raise ValueError("The two selected points differ from the initial calibration")
    if values["models"] != agreement.WEIGHT_PINS:
        raise ValueError("The local review uses different models from the calibration")
    for panel in (held, grid):
        bench.review(panel, selection)
    excluded = set(values["development_speakers"])
    excluded.update(row["speaker"] for path in (held, grid) for row in bench.read(path)["rows"])
    save(
        output,
        {
            "schema": SCHEMA,
            "selection": bench.digest(selection),
            "selection_panels": [bench.digest(p) for p in (held, grid)],
            "method": bench.digest(Path(agreement.__file__)),
            "method_points": agreement.POINTS,
            "already_observed_speakers": sorted(excluded),
        },
    )


def inspect(rows):
    summary = {}
    for carrier in bench.CARRIERS:
        counts = {role: Counter() for role in bench.ROLES}
        groups = {role: set() for role in bench.ROLES}
        reports = []
        for row in rows:
            frames = agreement.source_frames(row["native_times"])
            for role, case in row["cases"][carrier].items():
                observations = [agreement.encode(case["proposals"][arm]["candidate"]) for arm in bench.ARMS]
                selected = agreement.confirmed(observations, carrier, frames, row["samples"])
                shown = selected is not None and case["baseline"] is None
                positive = role == "different_words"
                correct = shown and positive and bench.useful(selected["span"], row["truth"])
                counts[role].update(
                    files=1,
                    past_flags=bool(case["baseline"]),
                    past_correct=positive and bench.useful(case["baseline"], row["truth"]),
                    added_flags=shown,
                    added_correct=correct,
                    misplaced=shown and positive and not correct,
                    assessed=all(case["proposals"][arm]["status"] != "coverage" for arm in bench.ARMS),
                )
                if shown:
                    groups[role].add(row["speaker"])
                    reports.append(
                        {
                            "source_sha256": row["source_sha256"],
                            "role": role,
                            "speaker": row["speaker"],
                            "span": selected["span"],
                            "bracket_frames": selected["bracket_frames"],
                            "correct": correct,
                        }
                    )
        summary[carrier] = {
            "roles": {
                role: dict(entry) | {"sources_added": len(groups[role])} for role, entry in counts.items()
            },
            "reports": reports,
        }
    return summary


def assess(selection, design, inputs, output):
    lock = bench.read(design)
    if (
        lock["schema"] != SCHEMA
        or lock["selection"] != bench.digest(selection)
        or lock["method"] != bench.digest(Path(agreement.__file__))
        or lock["method_points"] != {key: list(value) for key, value in agreement.POINTS.items()}
    ):
        raise ValueError("The report confirmation differs from its documented freeze")
    result = {}
    for scope, path in inputs:
        bench.review(path, selection)
        evaluated = bench.read(path)
        speakers = {row["speaker"] for row in evaluated["rows"]}
        digest = bench.digest(path)
        if (
            scope == "confirmation"
            and speakers & set(lock["already_observed_speakers"])
            or scope != "confirmation"
            and digest not in lock["selection_panels"]
        ):
            raise ValueError("The confirmation set reuses prior observations")
        result[scope] = {"groups": len(speakers), "receipt": digest, "outcomes": inspect(evaluated["rows"])}
    save(output, {"schema": SCHEMA, "freeze": bench.digest(design), "results": result})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    lock = modes.add_parser("freeze")
    score = modes.add_parser("assess")
    for parser, args in (
        (lock, ("selection", "held", "grid", "output")),
        (score, ("selection", "design", "output")),
    ):
        for field in args:
            parser.add_argument(field, type=Path)
    score.add_argument(
        "--input", dest="inputs", nargs=2, action="append", required=True, metavar=("SCOPE", "PANEL_RESULT")
    )
    args = vars(cli.parse_args())
    action = args.pop("mode")
    if action == "assess":
        args["inputs"] = [(name, Path(path)) for name, path in args["inputs"]]
    (freeze if action == "freeze" else assess)(**args)
