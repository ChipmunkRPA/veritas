# SPDX-License-Identifier: MIT
"""Recount the short-window comparison and optionally re-decode every Opus input."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts.review_audio_two_second_opus_transfer import (
    CODEC_POINTS,
    matches,
    outputs,
    read,
    require,
    sha,
    verify_native,
)

VIEWS = ("direct", "reflected")
LENGTH = 12000
STEP = 8000


def confirmed(row, view, point):
    elevated = {
        item["start_sample"]
        for item in row["short_scores"]
        if item[view] > point
    }
    return [(a, a + LENGTH) for a in sorted(elevated) if {a - STEP, a + STEP} & elevated]


def recount(freeze, dev, held):
    source_cases = {row["id"]: row for row in freeze["cases"]}
    require(len(source_cases) == 26, "Duplicate or missing source cases")
    require(dev["freeze_sha256"] == held["freeze_sha256"], "Calibration and held point at different freezes")
    require(dev["role"] == "calibration" and held["role"] == "held", "Wrong role for a score file")
    require((freeze["window_samples"], freeze["step_samples"], freeze["speech_fraction"]) == (LENGTH, STEP, .5),
            "The declared short schedule changed")
    seen = set()
    role_sets = []
    for role in (dev, held):
        role_sets.append({k: {row[k] for row in role["rows"]} for k in ("group", "room", "pcm_sha256")})
        for row in role["rows"]:
            require(row["id"] not in seen and row["role"] == role["role"]
                    and all(row[key] == value for key, value in source_cases[row["id"]].items()),
                    "A scored row differs from the planned source")
            seen.add(row["id"])
            domain = set(range(0, row["samples"] - LENGTH + 1, STEP))
            starts = [value["start_sample"] for value in row["short_scores"]]
            require(starts == sorted(set(starts)) and set(starts) <= domain, "Unexpected or repeated short window")
            require(all(type(value["speech_samples"]) is int and 6000 <= value["speech_samples"] <= 12000
                        and all(math.isfinite(value[view]) for view in VIEWS) for value in row["short_scores"]),
                    "A stored window lacks adequate activity or has invalid scores")
    require(seen == set(source_cases) and not any(role_sets[0][k] & role_sets[1][k] for k in role_sets[0]),
            "Roles do not cover precisely the planned disjoint inputs")
    require(all(not row["truth"] for row in dev["rows"]), "An edit influenced the point")
    require(Counter(row["kind"] for row in dev["rows"]) == {"ordinary": 7, "background": 1}
            and Counter(row["kind"] for row in held["rows"]) == {"ordinary": 8, "edited": 8, "background": 2},
            "Controls or held inputs are incomplete")
    all_scores = defaultdict(set)
    for row in dev["rows"]:
        for index, first in enumerate(row["short_scores"][:-1]):
            second = row["short_scores"][index + 1]
            if second["start_sample"] != first["start_sample"] + STEP:
                continue
            for view in VIEWS:
                all_scores[view].add(min(first[view], second[view]))
    points = {view: max(all_scores[view]) for view in VIEWS}
    require(points == dev["points"] == held["points"], "Stored confidence points differ from the frozen control rule")
    require(all(not confirmed(row, view, points[view]) for row in dev["rows"] for view in VIEWS),
            "A calibration control incorrectly flags at its own maximum")

    counts = defaultdict(Counter)
    selected, times = [], defaultdict(list)
    for row in held["rows"]:
        primary, archived = outputs(row, CODEC_POINTS["aac"])
        base = primary + archived
        labels = row["truth"]
        already = matches(base, labels)
        raw = {view: confirmed(row, view, points[view]) for view in VIEWS}
        choices = {"baseline": primary, "baseline_and_two_second": base,
                   **{view: [*base, *raw[view]] for view in VIEWS}}
        geometric = [(a, a + LENGTH) for a in range(0, row["samples"] - LENGTH + 1, STEP)]
        short_indices = {i for i, (a, b) in enumerate(labels) if b - a < 9600}
        for view, candidate in choices.items():
            match = matches(candidate, labels)
            totals = counts[f"{view}/{row['kind']}"]
            totals.update(cases=1, flagged=bool(candidate), located=len(match), labels=len(labels),
                          short_labels=len(short_indices), useful_gain=len(match - already), useful_lost=len(already - match),
                          geometric_ceiling=len(matches(geometric, labels)), short_forward_windows=len(row["short_scores"]),
                          short_labels_found=len(match & short_indices))
            if view in VIEWS and raw[view]:
                selected.append({"view": view, "id": row["id"], "kind": row["kind"], "family": row["family"],
                                 "ranges": [list(pair) for pair in raw[view]],
                                 "labels_added": [labels[i] for i in sorted(match - already)],
                                 "labels_lost": [labels[i] for i in sorted(already - match)],
                                 "no_prior_flag": not base})
        for arm, elapsed in row["seconds"].items():
            times[arm].append(elapsed)
    require(held["counts"] == dict(counts), "Reconstructed outcome counts differ")
    return {"points": points, "counts": counts, "selected": selected,
            "median_held_forward_seconds": {arm: median(values) for arm, values in times.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    files = parser.add_subparsers(dest="mode", required=True)
    build = files.add_parser("build")
    for field in ("directory", "output"):
        build.add_argument(field, type=Path)
    verify = files.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--native", type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        root = args.directory
        plan_bytes = (root / "freeze.json").read_bytes()
        freeze, dev, held = [read(root / name) for name in ("freeze.json", "calibration.json.gz", "held.json.gz")]
        require(dev["freeze_sha256"] == sha(plan_bytes), "Stored calibration does not match freeze file bytes")
        outcomes = recount(freeze, dev, held)
        native = verify_native({"rows": [*dev["rows"], *held["rows"]]}, dict.fromkeys(("calibration", "held"), root))
        result = {"freeze": freeze, "calibration": dev, "held": held, "audit": {"outcomes": outcomes, "native": native}}
        body = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with args.output.open("xb") as receipt:
            receipt.write(gzip.compress(body, mtime=0))
    else:
        result = read(args.receipt)
        freeze, dev, held = (result[k] for k in ("freeze", "calibration", "held"))
        freeze_bytes = json.dumps(freeze, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        require(sha(freeze_bytes) == dev["freeze_sha256"], "Packaged freeze differs from the scored commitment")
        outcomes = recount(freeze, dev, held)
        require(outcomes == result["audit"]["outcomes"], "Reconstructed report differs from the receipt")
        if args.native is not None:
            native = verify_native({"rows": [*dev["rows"], *held["rows"]]},
                                   dict.fromkeys(("calibration", "held"), args.native))
            require(native == result["audit"]["native"], "Independent media replay differs from the receipt")
    print(json.dumps(outcomes, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
