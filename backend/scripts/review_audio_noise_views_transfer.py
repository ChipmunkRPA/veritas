# SPDX-License-Identifier: Apache-2.0
"""Recount noisy-speech decisions and operating-point tradeoffs without model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-noise-views-public/1"
HEADS = ("enhanced", "joint")
SCENES = {"traffic", "cafe"}
LABELS = {"bonafide", "spoof", "isolated"}
BASELINES = {None, "unassessed", "inconclusive", "original", "ai"}


def require(check, message):
    if not check:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(trials, head, point, rule):
    grouped, alarms, gains, gain_scenes = defaultdict(Counter), set(), set(), set()
    for row in trials:
        scene, label = row["scene"], row["label"]
        count = grouped["isolated" if label == "isolated" else f"{scene}/{label}"]
        incumbent = row["baseline"] == "ai"
        additional = (row["admitted"] and row["scores"][head] > point
                      and row["baseline"] in {None, "unassessed", "inconclusive"})
        count["files"] += 1
        count["incumbent_ai"] += incumbent
        count["enhanced_admitted"] += row["admitted"]
        count["additional_ai"] += additional
        count["combined_ai"] += incumbent or additional
        if additional and scene in SCENES:
            if label == "spoof":
                gains.add(row["group"])
                gain_scenes.add(scene)
            else:
                alarms.add(row["group"])
    advances = (len(gains) >= rule["min_added_synthetic_groups"] and len(gain_scenes) >= rule["min_scenes_with_additions"]
                and len(alarms) <= rule["max_added_genuine_groups"]
                and grouped["gain_control/bonafide"]["additional_ai"] <= rule["max_added_gain_control"]
                and grouped["isolated"]["additional_ai"] <= rule["max_isolated_flags"])
    return {"cells": dict(grouped), "added_synthetic_groups": len(gains), "scenes_with_additions": sorted(gain_scenes),
            "added_genuine_groups": len(alarms), "advances": advances}


def frontier(trials, head, rule):
    points = sorted({row["scores"][head] for row in trials if row["admitted"]}, reverse=True)
    results = [summarize(trials, head, point, rule) for point in [*points, -math.inf]]
    limits = {}
    for budget in (0, 1, 2, 3):
        allowed = [row for row in results if row["added_genuine_groups"] <= budget
                   and row["cells"]["gain_control/bonafide"]["additional_ai"] == row["cells"]["isolated"]["additional_ai"] == 0]
        limits[str(budget)] = max((row["added_synthetic_groups"] for row in allowed), default=0)
    return limits


def review(receipt):
    require(receipt["schema"] == SCHEMA, "Unsupported public review format")
    trials, rule, points = (receipt[name] for name in ("trials", "rule", "points"))
    require(len(trials) == 104 and len({row["id"] for row in trials}) == len(trials), "Incomplete held trials")
    require(all(type(row["admitted"]) is bool and row["label"] in LABELS and row["baseline"] in BASELINES
                and set(row["scores"]) == set(HEADS) and all(math.isfinite(value) for value in row["scores"].values())
                for row in trials), "Invalid score rows")
    require(set(points) == set(HEADS) and all(math.isfinite(value) for value in points.values()), "Invalid frozen operating points")
    counts = {head: summarize(trials, head, points[head], rule) for head in HEADS}
    require(counts == receipt["results"], "Recorded aggregate does not match the exact decisions")
    envelope = {head: frontier(trials, head, rule) for head in HEADS}
    require(envelope == receipt["diagnostic_frontier"], "Diagnostic tradeoff table differs")
    return {"results": counts, "diagnostic_frontier": envelope}


def export(run):
    summary, scores, heads, roster = (read(run / name) for name in ("result.json", "held.json", "heads.json", "roster.json"))
    require(summary["held_sha256"] == sha(run / "held.json") and summary["heads"] == heads
            and scores["heads_sha256"] == sha(run / "heads.json")
            and scores["freeze_sha256"] == sha(run / "freeze.json"), "Private score or operating-point commitments differ")
    for row in roster:
        path = run / "pcm" / f"{row['id']:03}.pcm"
        require(path.stat().st_size == row["samples"] * 2 and sha(path) == row["pcm_sha256"], "Frozen PCM differs")
    parameters = heads["joint_parameters"]
    trials = []
    for row, source in zip(scores["rows"], (row for row in roster if row["role"] == "held"), strict=True):
        require(all(row[key] == value for key, value in source.items()), "Measured rows differ from the frozen roster")
        values = [row[key] for key in ("raw_logit", "enhanced_logit", "suppression_db")]
        weighted = [(value - offset) * weight / scale
                    for value, offset, scale, weight in zip(values, parameters["center"], parameters["scale"],
                                                           parameters["coef"][:-1], strict=True)]
        trials.append({"id": row["id"], "label": row["label"], "group": row["group"], "scene": row["scene"],
                       "baseline": row["baseline"], "admitted": row["enhanced_admitted"], "pcm_sha256": row["pcm_sha256"],
                       "scores": {"enhanced": values[1], "joint": math.fsum([*weighted, parameters["coef"][-1]])}})
    rule = summary["freeze"]["rule"]
    receipt = {"schema": SCHEMA, "freeze": summary["freeze"], "rule": rule, "points": heads["points"], "trials": trials,
               "development_sha256": heads["development_sha256"], "held_sha256": summary["held_sha256"],
               "model_free_pcm_inputs_verified": len(roster), "results": summary["results"],
               "diagnostic_frontier": {head: frontier(trials, head, rule) for head in HEADS}}
    review(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("export", "review"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "export":
        if not args.out:
            parser.error("export requires --out")
        receipt = export(args.input)
        with args.out.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    else:
        print(json.dumps(review(read(args.input)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
