"""Audit paired audio nuisance fits, points, and saved decisions without model inference."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_paired_nuisance_native as native
from scripts import audio_paired_nuisance_transfer as study

ARMS = ("raw_point", "paired_correction")
ROLES = study.ROLES[1:]


def original_logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p) - math.log1p(-p)


def archived_role(role, archive, lock, metadata):
    index = study.ROLES.index(role)
    rows = []
    for number in range(index * 12, (index + 1) * 12):
        path = archive / f"chunk-{number:02d}-full.jsonl"
        study.require(study.digest(path) == lock["archive_sha256"][str(number)], "Archive changed")
        with path.open() as stream:
            for line, original in zip(stream, metadata[number * study.CHUNK:(number + 1) * study.CHUNK], strict=True):
                row = json.loads(line)
                study.require(row["id"] == original["id"] and row["pcm_sha256"] == original["pcm_sha256"],
                              "Archived row/source binding differs")
                rows.append((row, original["background"]))
    study.require(len(rows) == 1044, "Partial archived role")
    return rows


def review_fit(rows, fit, weight):
    grouped = defaultdict(dict)
    for row, _ in rows:
        if row["embedding"] is not None:
            grouped[row["base_id"]][row["condition"]] = np.asarray(row["embedding"])
    clean = np.array([views["clean"] for views in grouped.values() if "clean" in views])
    center = np.asarray(fit["center"])
    correction = np.asarray(fit["correction"])
    study.require(np.max(np.abs(clean.mean(axis=0) - center)) <= 1e-12, "Clean center differs")
    clean_gradient = ((clean - center) * ((clean - center) @ correction)[:, None]).mean(axis=0)
    shifts = [[views[c] - views["clean"] for c in study.METHOD["fit_conditions"] if c in views]
              for views in grouped.values() if "clean" in views]
    shifts = [np.asarray(group) for group in shifts if group]
    mixed_gradient = np.mean([np.mean(x * (x @ (correction - weight))[:, None], axis=0) for x in shifts], axis=0)
    ridge = .25 * float(np.mean(np.square(clean - center)))
    gradient = clean_gradient + 4 * mixed_gradient + ridge * correction
    study.require(np.linalg.norm(gradient) <= 1e-8 and math.isclose(fit["ridge"], ridge, abs_tol=1e-12)
                  and fit["clean_anchors"] == len(clean) and fit["paired_foregrounds"] == len(shifts)
                  and fit["matched_shifts"] == sum(len(group) for group in shifts), "Fit normal-equation residual differs")
    return float(np.linalg.norm(gradient))


def chosen(old, score, point):
    if old == "original" or old == "unassessed":
        return old
    return "ai" if score > point else "inconclusive"


def pair_auc(positives, negatives):
    if not positives or not negatives:
        return None
    pairs = sum((1 if a > b else .5 if a == b else 0) for a in positives for b in negatives)
    return round(pairs / (len(positives) * len(negatives)), 6)


def verify_points(ledger, points):
    for arm in ARMS:
        expected = {}
        for condition in study.CONDITIONS:
            control = [r for r in ledger if r["condition"] == condition and not r["label"]
                       and r["old"] in ("ai", "inconclusive")]
            incumbent_false = sum(r["old"] == "ai" for r in control)
            bound = sorted((r[arm] for r in control), reverse=True)[incumbent_false]
            expected[condition] = {"controls": len(control), "incumbent_false_budget": incumbent_false, "point": bound}
        study.require(points[arm] == {"cells": expected, "point": max(p["point"] for p in expected.values())},
                      "Replayed calibration points differ")


def verify_counts(ledger, points, results, conditions, *, archived=None):
    for condition in conditions:
        for arm in ARMS:
            tally = Counter()
            pos, neg = [], []
            gained_families, gained_backgrounds = set(), set()
            for i, row in enumerate(ledger):
                if row["condition"] != condition:
                    continue
                old, score = row["old"], row[arm]
                family = row["kind"] if archived else row["family"]
                label = ("synthetic" if row["label"] else "genuine") if archived else (
                    family if family in ("genuine", "background") else "synthetic")
                direction = chosen(old, score, points[arm]["point"])
                tally[f"{label}_current_{old}"] += 1
                tally[f"{label}_candidate_{direction}"] += 1
                if old == "ai" and direction != "ai":
                    tally[f"{label}_lost"] += 1
                if old != "ai" and direction == "ai":
                    tally[f"{label}_added"] += 1
                    if archived and label == "synthetic":
                        gained_families.add(family)
                        bg = archived[i][1]
                        if bg is not None:
                            gained_backgrounds.add(bg["raw_sha256"])
                if score is not None and label != "background":
                    (pos if label == "synthetic" else neg).append(score)
            expected = {"counts": dict(tally), "auc": pair_auc(pos, neg)}
            if archived:
                expected.update(synthetic_recovered_families=sorted(gained_families),
                                background_sources_with_recoveries=len(gained_backgrounds))
                observed = results[arm][condition]
            else:
                observed = results[condition][arm]
            study.require(expected == observed, "Independent rank or decision count differs")


def verify_advancement(result):
    conditions = result["conditions"]
    corrected = [conditions["paired_correction"][name] for name in study.STRONG]
    plain = [conditions["raw_point"][name] for name in study.STRONG]
    clean = conditions["paired_correction"]["clean"]["counts"]
    gained = {family for cell in corrected for family in cell["synthetic_recovered_families"]}
    true = sum(cell["counts"].get("synthetic_candidate_ai", 0) for cell in corrected)
    original = sum(cell["counts"].get("synthetic_current_ai", 0) for cell in corrected)
    recalibrated = sum(cell["counts"].get("synthetic_candidate_ai", 0) for cell in plain)
    checks = {"four_net_strong_detections": true >= original + 4,
              "two_more_than_score_only": true >= recalibrated + 2,
              "no_extra_false_ai_in_strong_cells": all(cell["counts"].get("genuine_candidate_ai", 0)
                                                       <= cell["counts"].get("genuine_current_ai", 0) for cell in corrected),
              "no_extra_clean_false_ai": clean.get("genuine_candidate_ai", 0) <= clean.get("genuine_current_ai", 0),
              "all_existing_clean_ai_flags_preserved": clean.get("synthetic_lost", 0) == 0,
              "recoveries_two_families": len(gained) >= 2}
    study.require(result["advance"] == {"checks": checks, "passes": all(checks.values())}, "Advancement criteria differ")
    return all(checks.values())


def verify_archive(folder, archive):
    freeze, files, weight, _ = study.checked(folder)
    metadata = study.read(files["variants"])["variants"]
    fit_file, point_file = folder / "fitted-private.json", folder / "points.json"
    aggregate = study.read(folder / "aggregate.json")
    fit = study.read(fit_file)
    points = study.read(point_file)["points"]
    study.require(aggregate["fitted_sha256"] == study.digest(fit_file)
                  and aggregate["points_sha256"] == study.digest(point_file), "Fit/point file commitment differs")
    gradient = review_fit(archived_role("fusion_fit", archive, freeze, metadata), fit, weight)
    maximum = 0.
    views, passage = 0, {}
    for role in ROLES:
        archived = archived_role(role, archive, freeze, metadata)
        ledger = study.read(folder / f"{role}-scalar-ledger.json")
        for (recorded, _), row in zip(archived, ledger, strict=True):
            study.require(recorded["label"] == row["label"] and recorded["kind"] == row["kind"]
                          and recorded["condition"] == row["condition"]
                          and recorded["decisions"]["current"] == row["old"], "Ledger role changed")
            if recorded["raw"] is None:
                study.require(row["raw_point"] is None and row["paired_correction"] is None,
                              "Missing incumbent response cannot produce correction")
                continue
            raw = original_logit(recorded["raw"])
            removed = math.fsum((x - c) * w for x, c, w in zip(
                recorded["embedding"], fit["center"], fit["correction"], strict=True))
            discrepancy = max(abs(raw - row["raw_point"]), abs(raw - removed - row["paired_correction"]))
            study.require(discrepancy <= 2e-12, "Independently reconstructed ledger score differs")
            maximum = max(maximum, discrepancy)
        if role == "confidence_cal":
            verify_points(ledger, points)
        result = aggregate["roles"][role]
        verify_counts(ledger, points, result["conditions"], study.CONDITIONS, archived=archived)
        passage[role] = verify_advancement(result)
        views += len(ledger)
    study.require(aggregate["advance"] == (passage["selection"] and passage["confirmation"]), "Release decision differs")
    return {"verified_archive_views": views + 1044, "scored_decision_replays": views, "rank_cells": len(ROLES) * 9 * 2,
            "maximum_scalar_error": maximum, "fit_gradient_norm": gradient, "aggregate_sha256": study.digest(folder / "aggregate.json")}


def verify_native(folder, source, native_run):
    lock, exact, _, points = native.bind(folder, source)
    ledger = native_run / "scalar-ledger-private.json"
    result = study.read(native_run / "aggregate.json")
    study.require(study.read(native_run / "freeze.json") == lock and result["ledger_sha256"] == study.digest(ledger),
                  "Native frozen source or output ledger changed")
    entries = study.read(ledger)
    for case, row in zip(exact, entries, strict=True):
        native.pcm(source, case)
        study.require(all(case[key] == row[key] for key in ("id", "condition", "pcm_sha256", "family"))
                      and case["baseline"]["decision"] == row["old"], "Native ledger does not bind the source")
        raw = case["baseline"]["raw"]
        if raw is None:
            study.require(row["raw_point"] is None and row["paired_correction"] is None, "Native unassessed response")
        else:
            study.require(math.isfinite(row["paired_correction"]) and abs(original_logit(raw) - row["raw_point"]) <= 2e-12,
                          "Native raw response changed")
    verify_counts(entries, points, result["conditions"], native.CONDITIONS)
    return {"verified_pcm_views": len(entries), "native_conditions": len(native.CONDITIONS), "new_neural_forwards": 0,
            "aggregate_sha256": study.digest(native_run / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "archive", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--native-source", type=Path)
    args = parser.parse_args()
    result = verify_archive(args.folder, args.archive)
    if args.native_run is not None:
        result["native"] = verify_native(args.folder, args.native_source, args.native_run)
    study.write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
