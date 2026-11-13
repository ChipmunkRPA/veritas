"""Recompute the Bayer prediction evaluation from its public scalar measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PANELS = {
    "development": "image-native-stencil-interpolation",
    "confirmation": "image-native-generated-grid",
}
POSITIVE = {"generated_feathered", "generated_sharp", "generated_linear_promote"}
OUT_OF_SCOPE = {"resized", "eight_bit"}


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def encoded(document):
    return (json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def area(box):
    if box is None:
        return 0
    x, y, x1, y1 = box
    return (x1 - x) * (y1 - y)


def validate_rows(rows, prior, positive, split):
    remaining = {(row["key"], row["role"]): row for row in prior}
    maxima = dict.fromkeys({key for key, _ in remaining}, 0.)
    for row in rows:
        identity = row["camera"], row["role"]
        check(identity in remaining, "Unplanned or duplicate observation")
        witness = remaining.pop(identity)
        check(all(row[field] == witness[field] for field in ("input_sha256", "truth_sha256", "truth_pixels")),
              "Native pixels or masks differ from the public source record")
        status, box, score, hit = (row[key] for key in ("status", "box", "score", "intersection"))
        check(status in {"unsupported", "no_repeatable_stencil", "assessed"}
              and (status == "assessed") == (score is not None)
              and (status == "unsupported") == (identity[1] == "eight_bit"), "Invalid assessment status")
        check(score is None or (math.isfinite(score) and 0 <= score <= 12 and bool(score) == bool(box)), "Invalid score")
        if box is not None:
            x, y, x1, y1 = box
            check(all(type(n) is int and n % 16 == 0 for n in box) and 0 <= x < x1 <= 2048 and 0 <= y < y1 <= 1536
                  and all(n in (64, 96, 128, 160, 192) for n in (x1-x, y1-y)) and max(x1-x, y1-y) <= 2*min(x1-x, y1-y),
                  "Proposed region lies outside the native search grid")
        check(type(hit) is int and 0 <= hit <= min(row["truth_pixels"], area(box)), "Invalid pixel intersection")
        if identity[1] not in positive and score is not None:
            maxima[identity[0]] = max(maxima[identity[0]], score)
    check(not remaining, f"Missing {split} observations")
    return maxima


def outcome(rows, prior, point):
    baseline = {(row["key"], row["role"]): row["shifted"] for row in prior}
    totals = {}
    for row in rows:
        role, target, hit, box = (row[key] for key in ("role", "truth_pixels", "intersection", "box"))
        old = baseline[row["camera"], role]
        prior_flag = bool(old and old["box"] and old["score"] > 2.4346873354224834
                          and not any(all(value > .70 for value in axis) for axis in old["parity_fit"]))
        current_flag = row["score"] is not None and row["score"] > point
        prior_useful = False
        if prior_flag and role in POSITIVE:
            prior_useful = old["pixels"]["iou"] >= .30 and old["pixels"]["recall"] >= .50
        top_useful = role in POSITIVE and bool(box) and hit / target >= .50 and hit / (area(box) + target - hit) >= .30
        current_useful = current_flag and top_useful
        totals.setdefault(role, Counter()).update(files=1, assessed=row["status"] == "assessed",
                baseline=prior_flag, candidate=current_flag, baseline_useful=prior_useful, candidate_useful=current_useful,
                kept=prior_useful and current_useful, gained=current_useful and not prior_useful,
                lost=prior_useful and not current_useful, top_useful=top_useful)
    gains = {role: totals[role]["gained"] - totals[role]["lost"] for role in POSITIVE}
    flagged = {role: totals[role]["candidate"] for role in totals}
    ordinary_delta = sum(flagged[role] - totals[role]["baseline"] for role in totals.keys() - POSITIVE)
    advances = (gains["generated_feathered"] >= 2 and gains["generated_sharp"] + gains["generated_linear_promote"] >= 2
                and all(totals[role]["lost"] * 3 <= totals[role]["baseline_useful"] for role in POSITIVE)
                and ordinary_delta <= 2 and not any(flagged[role] for role in OUT_OF_SCOPE | {"original"}))
    return totals, advances


def review(path):
    ledger = json.loads(path.read_bytes())
    freeze, development = (ledger[key] for key in ("freeze", "development"))
    check(freeze["schema"] == "chipmunk.image-bayer-prediction-transfer/1" and
          development["freeze_sha256"] == digest(encoded(freeze)) and
          ledger["development_sha256"] == digest(encoded(development)), "Freeze or operating point was changed")
    sources = {}
    models = {}
    for split, prefix in PANELS.items():
        measurements = path.with_name(f"{prefix}-measurements-2026-11-13.json").read_bytes()
        roster = path.with_name(f"{prefix}-cc0-roster-2026-11-13.json").read_bytes()
        binding = freeze["sources"][split]
        check((binding["measurements_sha256"], binding["roster_sha256"]) == (digest(measurements), digest(roster)),
              "Public source commitments differ")
        records = json.loads(measurements)
        check(records["freeze_sha256"] == binding["source_freeze_sha256"], "Retained source freeze differs")
        roster = json.loads(roster)
        check(all(row["license"] == "CC0-1.0" for row in roster)
              and {row["id"] for row in roster} == {row["key"] for row in records["rows"]}, "Source rights roster differs")
        models[split] = {row["model"].casefold() for row in roster}
        check(len(models[split]) == len(roster) == (13 if split == "development" else 8), "Camera groups differ")
        sources[split] = records["rows"]
    check(not models["development"] & models["confirmation"], "Camera groups overlap")
    maxima = validate_rows(development["rows"], sources["development"], POSITIVE, "development")
    validate_rows(ledger["confirmation"], sources["confirmation"], POSITIVE, "confirmation")
    point = sorted(maxima.values())[-2]
    check(point == development["point"], "Operating point differs from ordinary source maxima")
    totals, advances = outcome(ledger["confirmation"], sources["confirmation"], point)
    check(totals == ledger["counts"] and advances is ledger["advances"], "Counts or decision differ")
    return {"counts": totals, "advances": advances, "development_rows": len(development["rows"]), "point": point}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    print(json.dumps(review(parser.parse_args().ledger), sort_keys=True))
