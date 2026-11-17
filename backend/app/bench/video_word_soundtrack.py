# SPDX-License-Identifier: Apache-2.0
"""Review additional word proposals without loading the speech or video models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from app.bench.video_word_content import (
    AUDIO_MARGIN,
    AUDIO_MIN,
    CARRIERS,
    MATCH,
    ROLES,
    overlap,
    read,
    recount,
)

SCHEMA = "chipmunk.video-word-soundtrack-transfer/1"
METHODS = ("primary", "conditional", "word_crop", "independent_scan")
WEAK_MIN = 0.24
WEAK_MARGIN = 0.10
ALTERNATE_MARGIN = 0.25
LEVELS = {
    "conditional": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
    "word_crop": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
    "independent_scan": (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0),
}
DELAYS = {"global_early": -3200, "global_late": 3200}


def eligible_primary(voice):
    return bool(voice and voice["posterior"] >= AUDIO_MIN and voice["margin"] >= AUDIO_MARGIN)


def selected(case, method, point):
    voice = case["speech"]
    if case["conditional"] is not None and eligible_primary(voice):
        return voice, case["mouth"]
    if method == "independent_scan":
        proposal = case["scan_speech"]
        if proposal and proposal["score"] >= point:
            return proposal, case["scan_mouth"]
        return None, None
    if method == "primary" or voice is None or voice["posterior"] < WEAK_MIN or voice["margin"] < WEAK_MARGIN:
        return None, None
    reading = case[method]
    if (
        reading
        and reading["score"] >= point
        and reading["margin"] >= ALTERNATE_MARGIN
        and (method == "conditional" or (reading["repeat"] and reading["digit"] == voice["digit"]))
    ):
        return voice, case["mouth_unfiltered"]
    return None, None


def correct_audio(row, role, proposal):
    if proposal is None:
        return False
    a, b = row["truth"]
    shift = DELAYS.get(role, 0)
    digit = row["expected_audible"] if role == "different_words" else row["expected_visual"]
    return proposal["digit"] == digit and overlap(proposal["span"], (a + shift, b + shift)) >= MATCH


def audio_counts(rows, carrier, method, point):
    totals = defaultdict(Counter)
    for row in rows:
        for role, case in row["cases"][carrier].items():
            proposal, _ = selected(case, method, point)
            baseline, _ = selected(case, "primary", 0.0)
            right = correct_audio(row, role, proposal)
            added = proposal is not None and baseline is None
            totals[role].update(
                cases=1,
                usable=proposal is not None,
                admitted_correct=right,
                admitted_wrong=proposal is not None and not right,
                recovered=added and right,
                added_wrong=added and not right,
            )
    return dict(totals)


def select_points(rows):
    def choose(carrier, method, levels):
        def merit(level):
            counts = audio_counts(rows, carrier, method, level)
            useful = sum(case["recovered"] for case in counts.values())
            wrong = sum(case["added_wrong"] for case in counts.values())
            return useful - wrong, -wrong, counts["different_words"]["recovered"], level

        return max(levels, key=merit)

    return {
        method: {carrier: choose(carrier, method, levels) for carrier in CARRIERS}
        for method, levels in LEVELS.items()
    }


def counts(rows, points, visual_points):
    alternatives = {}
    for method in METHODS:
        levels = points.get(method, dict.fromkeys(CARRIERS, 0.0))
        projected = deepcopy(rows)
        for row in projected:
            for carrier, cases in row["cases"].items():
                for case in cases.values():
                    proposal, case["mouth"] = selected(case, method, levels[carrier])
                    if proposal is not None:
                        case["speech"] = proposal
        alternatives[method] = {
            "audio": {carrier: audio_counts(rows, carrier, method, levels[carrier]) for carrier in CARRIERS},
            "visual": recount(projected, visual_points),
        }
    return alternatives


def review(path, prior_path, freeze_path=None):
    report = read(path)
    prior = read(prior_path)
    rows = report["rows"]
    prior_digest = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    if (
        report["schema"] != SCHEMA
        or not rows
        or report["previous"] != prior_digest
        or report["visual_points"] != prior["points"]
        or report["configuration"] != prior["configuration"]
    ):
        raise ValueError("The soundtrack trial must use the original fixed mouth and frame readings")
    if len({r["source_sha256"] for r in rows}) != len(rows):
        raise ValueError("A source file was repeated")
    for row in rows:
        if set(row["cases"]) != set(CARRIERS) or any(
            set(cases) != set(ROLES) for cases in row["cases"].values()
        ):
            raise ValueError("The matched carrier or control roster is incomplete")
        for cases in row["cases"].values():
            for case in cases.values():
                for reading in (case["speech"], case["scan_speech"]):
                    if reading and not 0 <= reading["span"][0] < reading["span"][1] <= row["samples"]:
                        raise ValueError("An audible interval is outside its source clock")
    if freeze_path is None:
        if (
            report["phase"] != "calibration"
            or report["freeze"] is not None
            or report["points"] != select_points(rows)
        ):
            raise ValueError("The development readings do not reproduce the acoustic points")
    else:
        control = read(freeze_path)
        review(freeze_path, prior_path)
        stable = ("points", "visual_points", "configuration", "weight", "method_pins")
        if (
            report["phase"] == "calibration"
            or report["freeze"] != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            or any(report[key] != control[key] for key in stable)
            or {r["speaker"] for r in rows} & {r["speaker"] for r in control["rows"]}
        ):
            raise ValueError("The evaluation crossed the frozen method or development split")
    result = counts(rows, report["points"], report["visual_points"])
    if result != report["results"]:
        raise ValueError("The measured cases do not reproduce the reported totals")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.receipt, args.prior, args.freeze), sort_keys=True, indent=2))
