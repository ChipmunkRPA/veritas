# SPDX-License-Identifier: Apache-2.0
"""Recount and verify the frozen single-video word comparison without models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("one_window", "nearby_time_check")
CARRIERS = ("pcm", "aac48")
ROLES = ("original", "gain", "same_words", "different_words", "global_early", "global_late")
MATCH = 0.40


def overlap(first, second):
    a, b = first
    x, y = second
    union = max(b, y) - min(a, x)
    return max(0, min(b, y) - max(a, x)) / union


def recount(rows, points):
    total = defaultdict(Counter)
    for row in rows:
        for enc, cases in row["cases"].items():
            for role, case in cases.items():
                category = (
                    "ordinary"
                    if role in ("original", "gain")
                    else "whole_delay"
                    if role.startswith("global")
                    else role
                )
                truth = row["truth"]
                old = case["baseline"]
                old_correct = bool(old and overlap(old, truth) >= MATCH)
                speech = case["speech"]
                for arm in ARMS:
                    outcome = total[enc + "/" + arm + "/" + category]
                    mouth = case["mouth"]
                    passed = bool(mouth and mouth[arm] > points[arm][enc])
                    matches = False
                    correct = False
                    if role == "different_words" and passed:
                        matches = overlap(speech["span"], truth) >= MATCH
                        correct = (
                            matches
                            and speech["digit"] == row["expected_audible"]
                            and mouth["visible_digit"] == row["expected_visual"]
                        )
                    outcome.update(
                        {
                            "files": 1,
                            "assessed": bool(mouth),
                            "candidate_flags": passed,
                            "old_flags": bool(old),
                            "union_flags": bool(old or passed),
                            "added_flag": bool(passed and not old),
                        }
                    )
                    if role == "different_words":
                        outcome.update(
                            {
                                "correct_audio": bool(
                                    speech
                                    and speech["digit"] == row["expected_audible"]
                                    and overlap(speech["span"], truth) >= MATCH
                                ),
                                "old_correct_region": old_correct,
                                "candidate_overlap": matches,
                                "correct_content_and_region": correct,
                                "additional_correct": bool(correct and not old_correct),
                                "unsupported_candidate": bool(passed and not correct),
                            }
                        )
    return dict(sorted(total.items()))


def cutoffs(rows):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for enc in CARRIERS:
            by_group = {}
            for row in rows:
                for case in ("original", "gain", "same_words", "global_early", "global_late"):
                    report = row["cases"][enc][case]["mouth"]
                    if report:
                        group = row["speaker"]
                        by_group[group] = max(by_group.get(group, 0.0), report[arm])
            ranking = sorted(by_group.values(), reverse=True)
            result[arm][enc] = max(
                0.10, ranking[1] if len(ranking) >= 2 else (ranking[0] if ranking else 0.10)
            )
    return result


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def review(path, frozen=None):
    report = read(path)
    if report.get("schema") != "chipmunk.video-word-content/1" or not report.get("rows"):
        raise ValueError("Expected the whole-word comparison receipt")
    rows = report["rows"]
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("The receipt reuses a source video")
    for row in rows:
        if set(row["cases"]) != set(CARRIERS) or any(
            set(cases) != set(ROLES) for cases in row["cases"].values()
        ):
            raise ValueError("A source omitted one of its matched delivery conditions")
        if any(
            cases[role]["speech"]
            and not 0 <= cases[role]["speech"]["span"][0] < cases[role]["speech"]["span"][1] <= row["samples"]
            for cases in row["cases"].values()
            for role in ROLES
        ):
            raise ValueError("An audible interval lies outside the presented soundtrack")
    if report["freeze"] is None:
        if frozen is not None or cutoffs(rows) != report["points"]:
            raise ValueError("The receipt does not match its development operating points")
    else:
        if frozen is None or hashlib.sha256(frozen.read_bytes()).hexdigest() != report["freeze"]:
            raise ValueError("The held result has a different development receipt")
        control = read(frozen)
        review(frozen)
        if control["points"] != report["points"] or control["configuration"] != report["configuration"]:
            raise ValueError("The held result changed a frozen operating point or model")
        if {row["speaker"] for row in rows} & {row["speaker"] for row in control["rows"]}:
            raise ValueError("A held group also set the operating point")
    totals = recount(rows, report["points"])
    if totals != report["outcomes"]:
        raise ValueError("The recorded counts differ from the individual cases")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.receipt, args.freeze), indent=2, sort_keys=True))
