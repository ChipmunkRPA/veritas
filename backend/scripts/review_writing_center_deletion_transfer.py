"""Independently replay exact context inputs and decisions from saved PEP scores."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from app.detectors.text.extract import text_hash
from app.evidence.document_matched_context import one_complete_window
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_center_deletion_transfer import ARMS, CALIBRATION, EXACT_ARMS, confirmation, development
from scripts.writing_focus_v3 import digest
from scripts.writing_paired_revision_replication import SEEDS


def log_odds(value):
    require(type(value) is float and math.isfinite(value) and 0 <= value <= 1, "Invalid score")
    clipped = min(1 - 1e-7, max(1e-7, value))
    return math.log(clipped) - math.log1p(-clipped)


def geometry(rows, scores):
    bound, seen = 0, set()
    for row in rows:
        text = row["text"]
        require(row["text_sha256"] == text_hash(text), "Source text hash differs")
        context = row["native"]
        centers = row["centers"]
        require(len(context) == 5 and {entry["position"] for entry in centers} <= {1, 2, 3}, "Invalid native positions")
        for j, center in enumerate(context):
            spans = center["spans"]
            require(len(spans) == (2 if j in {0, 4} else 3), "Wrong native window length")
            if j in {1, 2, 3}:
                focal = next((item for item in centers if item["position"] == j), None)
                if focal is not None:
                    require(focal["span"] == spans[1] and len(focal["probes"]) == 3
                            and [item["spans"] for item in focal["probes"]]
                            == [spans[:2], spans[1:], [spans[0], spans[2]]], "Delete-one native geometry differs")
        entries = context + [item for center in centers for item in center["probes"]]
        for entry in entries:
            prev = 0
            pieces = []
            for start, end in entry["spans"]:
                require(prev <= start < end <= len(text), "Invalid native source span")
                pieces.append(text[start:end])
                prev = end
            require(entry["sha256"] == text_hash(" ".join(pieces)), "Exact input hash differs")
            reading = scores[entry["sha256"]]
            require(one_complete_window(reading["coverage"]), "Saved score is missing complete token coverage")
            seen.add(entry["sha256"])
            bound += 1
    require(seen == scores.keys(), "Score inventory differs from source geometry")
    return bound


def analyze(row, scores):
    native = [scores[item["sha256"]]["score"] for item in row["native"]]
    arms = dict(zip(EXACT_ARMS, ({}, {}), strict=True))
    for center in row["centers"]:
        position = center["position"]
        left, right, no_center = (log_odds(scores[probe["sha256"]]["score"]) for probe in center["probes"])
        arms["matched_context"][position] = log_odds(native[position])
        arms["center_deletion"][position] = min(left - no_center, right - no_center)
    return arms


def position(values, cutoff):
    if not values or max(values.values()) <= cutoff:
        return None
    high = max(values.values())
    leaders = [item for item, value in values.items() if value == high]
    return leaders[0] if len(leaders) == 1 else None


def measured(row, spans):
    start, end = row["focus"]
    true = {i for i in range(start, end) if not row["text"][i].isspace()}
    area = {i for a, b in spans for i in range(a, b) if not row["text"][i].isspace()}
    overlap = len(area.intersection(true))
    return 10 * overlap >= 3 * (len(area) + len(true) - overlap) and 2 * overlap >= len(true)


def replay(run, source):
    freeze, fixed, saved = (read(run / filename) for filename in ("freeze.json", "holdout-freeze.json", "aggregate.json"))
    require(freeze["calibration"] == {name: digest(path) for name, path in CALIBRATION.items()},
            "Calibration source commitments differ")
    cutoffs, control_scores, holdout_scores = (read(run / name) for name in ("operating-points.json", "calibration.json", "holdout.json"))
    require(cutoffs["calibration_sha256"] == digest(run / "calibration.json")
            and control_scores["binding"] == digest(run / "freeze.json")
            and holdout_scores["binding"] == digest(run / "holdout-freeze.json")
            and fixed["point_sha256"] == digest(run / "operating-points.json")
            and fixed["case_sha256"] == digest(run / "holdout-cases.json")
            and saved["holdout_sha256"] == digest(run / "holdout.json"), "Trial commitment differs")
    calibration, _ = development()
    rows, values, admission = confirmation(run, source)
    require(rows == read(run / "holdout-cases.json") and sorted(values) == fixed["input_hashes"],
            "Rebuilt fixture set differs from frozen")
    checked = geometry(calibration, control_scores["responses"]) + geometry(rows, holdout_scores["responses"])
    maxima = {arm: {} for arm in EXACT_ARMS}
    for row in calibration:
        for arm, scores in analyze(row, control_scores["responses"]).items():
            if scores:
                group = row["group"]
                maxima[arm][group] = max(*scores.values(), maxima[arm].get(group, -math.inf))
    for arm, groups in maxima.items():
        require(all(math.isclose(value, cutoffs["group_maxima"][arm][name], abs_tol=1e-12)
                    for name, value in groups.items())
                and math.isclose(sorted(groups.values())[-2], cutoffs["cutoff"][arm], abs_tol=1e-12),
                "Grouped control point differs")
    decisions = []
    for row in rows:
        scores = holdout_scores["responses"]
        current = [entry["spans"] for entry in row["native"] if scores[entry["sha256"]]["score"] >= .5][:6]
        exact = {}
        for arm, readings in analyze(row, scores).items():
            focus = position(readings, cutoffs["cutoff"][arm])
            exact[arm] = [[item["span"]] for item in row["centers"] if item["position"] == focus]
        inventory = {"current": current, **exact, "combined": current + exact["center_deletion"]}
        decisions.append({"group": row["group"], "role": row["role"], "arms": {
            arm: {"flag": bool(windows), "useful": any(measured(row, candidate) for candidate in windows)}
            for arm, windows in inventory.items()},
            "matched_plus_current": any(measured(row, candidate) for candidate in current + exact["matched_context"])})
    require(decisions == read(run / "outcomes.json") and saved["outcome_sha256"] == digest(run / "outcomes.json"),
            "Independent localization differs")
    counts = {}
    for role in ("before", "after", *SEEDS):
        subset = [row for row in decisions if row["role"] == role]
        counts[role] = {arm: {key: sum(row["arms"][arm][key] for row in subset) for key in ("flag", "useful")}
                        for arm in ARMS} | {"files": len(subset)}
        for arm in ARMS:
            counts[role][arm]["flags"] = counts[role][arm].pop("flag")
    require(counts == saved["counts"] and admission == saved["generator_admission"], "Held totals differ")
    gains = {name: {row["group"] for row in decisions if row["role"] == name and row["arms"]["combined"]["useful"]
                    and not row["arms"]["current"]["useful"]} for name in SEEDS}
    false_groups = {row["group"] for row in decisions if row["role"] == "after" and row["arms"]["combined"]["flag"]
                    and not row["arms"]["current"]["flag"]}
    extras = {row["group"] for row in decisions if row["role"] in SEEDS and row["arms"]["combined"]["useful"]
              and not row["matched_plus_current"]}
    decision = saved["decision"]
    require(decision["unique_gain_groups"] == len(set.union(*gains.values()))
            and decision["gain_groups_by_generator"] == {name: len(groups) for name, groups in gains.items()}
            and decision["added_later_false_groups"] == len(false_groups) and decision["added_over_matched_groups"] == len(extras)
            and decision["advances"] == (all(admission[name].get("admitted", 0) >= 6 and gains[name] for name in SEEDS)
                                         and len(set.union(*gains.values())) >= 3 and len(false_groups) <= 1 and len(extras) >= 2),
            "Advancement arithmetic differs")
    return {"checked_fixture_files": len(calibration) + len(rows), "checked_input_bindings": checked,
            "held_decision_records": len(decisions), "aggregate_sha256": digest(run / "aggregate.json"),
            "verified": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("source", type=Path)
    args = cli.parse_args()
    result = replay(args.run, args.source)
    write(args.run / "audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
