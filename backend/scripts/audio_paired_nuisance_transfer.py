"""Test paired acoustic nuisance correction on fixed audio detector embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
BASE = ROOT / "backend/data/independent-evaluation"
WIDTH = 1024
CHUNK = 87
ROLES = ("fusion_fit", "confidence_cal", "selection", "confirmation")
STRONG = ("speech_0", "speech_5")
CONDITIONS = ("clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15", "speech_0", "speech_5", "speech_15")
METHOD = {
    "correction": "Fit one label-free linear correction on paired changes of the same foreground in competing speech.",
    "fit_conditions": ["speech_0", "speech_5", "speech_15"],
    "fit_loss": "Clean correction squared plus four times residual paired score drift squared plus trace-scaled ridge.",
    "weights": "Equal fitted clean foregrounds. Equal noisy variants within each paired foreground.",
    "speech_weight": 4.0,
    "ridge_clean_mean_variance": .25,
    "readout": "Current raw log odds minus the correction centered on the fitted clean embedding mean.",
    "point": "One score per method. Enforce the incumbent false AI count in each calibration condition using strict excess.",
    "scope": "Use only original pooled features and incumbent raw responses. No acoustic condition or second file at prediction.",
    "policy": "Existing original and unassessed outcomes are retained. Other outcomes are AI when the score exceeds the point.",
    "advance": "Both selection and confirmation need at least four net extra strong-speech detections against current, "
               "at least two more than score-only, no additional false AI in either strong-speech cell, no additional "
               "clean false AI, no lost clean synthetic AI flags, and recoveries from two synthetic generator families.",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def paths():
    return {
        "original_freeze": PANEL / "audio-combined-confidence-v84-freeze.json",
        "published_fit": PANEL / "audio-combined-confidence-v84-training.json",
        "published_selection": PANEL / "audio-combined-confidence-v84-selection.json",
        "published_confirmation": PANEL / "audio-combined-confidence-v84-result.json",
        "variants": BASE / "audio-confidence-mixtures-v82/variants.json",
        "original_raw_head": BASE / "audio-v9/development/representations.jsonl",
        "original_raw_head_receipt": PANEL / "audio-shared-confirmation-v62-freeze.json",
    }


def metadata(files):
    frozen, variants = read(files["original_freeze"]), read(files["variants"])
    require(digest(files["variants"]) == frozen["inputs_sha256"] and frozen["roles"] == list(ROLES)
            and frozen["chunks"] == 48 and frozen["chunk_size"] == CHUNK,
            "The original committed four-role input grid is required")
    prior_head = read(files["original_raw_head_receipt"])["artifacts_sha256"]
    require(digest(files["original_raw_head"]) == prior_head[str(files["original_raw_head"].relative_to(ROOT))],
            "Original baseline head commitment differs")
    rows = variants["variants"]
    require(len(rows) == len({row["id"] for row in rows}) == 48 * CHUNK
            and [row["role"] for row in rows] == [r for r in ROLES for _ in range(12 * CHUNK)],
            "Exact prior role coverage required")
    owners = {}
    role_counts = {}
    for role in ROLES:
        part = [row for row in rows if row["role"] == role]
        sets = {"groups": {row["group"] for row in part},
                "foregrounds": {row["source_sha256"] for row in part},
                "pcm": {row["pcm_sha256"] for row in part},
                "backgrounds": {row["background"]["raw_sha256"] for row in part if row["background"]}}
        role_counts[role] = {key: len(values) for key, values in sets.items()}
        for key, values in sets.items():
            require(not (owners.get(key, set()) & values), f"Cross-role source overlap in {key}")
            owners.setdefault(key, set()).update(values)
    published_fit = read(files["published_fit"])
    published = {**published_fit["fit_stages"], **published_fit["calibration_stages"],
                 **read(files["published_selection"])["source_stages"],
                 **read(files["published_confirmation"])["source_stages"]}
    commitments = {int(index): part["full_sha256"] for index, part in published.items() if part["completed"] == CHUNK}
    require(set(commitments) == set(range(48)), "Missing published archive chunk")
    return rows, role_counts, commitments


def freeze(folder, archive):
    files = paths()
    _, counts, commitments = metadata(files)
    for index, expected in commitments.items():
        require(digest(archive / f"chunk-{index:02d}-full.jsonl") == expected, "An archive chunk differs from its published digest")
    folder.mkdir(mode=0o700)
    write(folder / "freeze.json", {"schema": "chipmunk.audio-paired-nuisance/1", "method": METHOD,
          "code_sha256": digest(Path(__file__)), "input_sha256": {name: digest(path) for name, path in files.items()},
          "archive_sha256": commitments, "known_disjoint_role_counts": counts, "neural_forwards": 0,
          "versions": {"numpy": np.__version__}})


def checked(folder):
    frozen = read(folder / "freeze.json")
    files = paths()
    require(frozen["method"] == METHOD and frozen["code_sha256"] == digest(Path(__file__))
            and frozen["input_sha256"] == {name: digest(path) for name, path in files.items()},
            "Frozen protocol or inputs changed")
    with files["original_raw_head"].open() as stream:
        reference = json.loads(next(stream))
    weight, bias = np.asarray(reference["head_weight"], np.float64), reference["head_bias"]
    require(weight.shape == (WIDTH,) and np.isfinite(weight).all() and math.isfinite(bias), "Malformed head")
    return frozen, files, weight, bias


def records(role, archive, frozen, files, weight, bias):
    index = ROLES.index(role)
    parts = read(files["variants"])["variants"]
    rows, vectors = [], []
    discrepancy = 0.0
    for number in range(12 * index, 12 * (index + 1)):
        path = archive / f"chunk-{number:02d}-full.jsonl"
        require(digest(path) == frozen["archive_sha256"][str(number)], "Exact source-bound archive required")
        with path.open() as stream:
            for line, expected in zip(stream, parts[CHUNK * number:CHUNK * (number + 1)], strict=True):
                row = json.loads(line)
                require(all(row[key] == expected[key] for key in ("id", "base_id", "role", "group", "label", "kind",
                                                                 "condition", "pcm_sha256", "sha256")),
                        "Archived score and variant source differ")
                old = row["decisions"]["current"]
                require(old in ("ai", "original", "inconclusive", "unassessed"), "Invalid incumbent decision")
                if old != "unassessed":
                    require(old == row["original_decision"], "Assessed decision differs from the original")
                if row["status"] == "ok":
                    embedding = np.asarray(row["embedding"], "<f4")
                    require(embedding.shape == (WIDTH,) and np.isfinite(embedding).all()
                            and row["embedding"] == embedding.tolist()
                            and hashlib.sha256(embedding.tobytes()).hexdigest() == row["embedding_sha256"]
                            and row["input_pcm_sha256"] == row["pcm_sha256"]
                            and row["windows"] == [{"start_sample": 0, "end_sample": 64000}]
                            and row["feature_frames"] == row["valid_feature_frames"] == 199
                            and old != "unassessed" and type(row["raw"]) is float and 0. <= row["raw"] <= 1.,
                            "Archived readout differs from the assessed native window")
                    delta = abs(float(expit(embedding @ weight + bias)) - row["raw"])
                    require(delta < 1e-6, "Original head did not reconstruct its stored score")
                    discrepancy = max(delta, discrepancy)
                else:
                    require(row["status"] == "not_applicable" and old == "unassessed"
                            and row["embedding"] is None and row["raw"] is None, "Unassessed input has a readout")
                    embedding = np.zeros(WIDTH, dtype="<f4")
                rows.append({key: row[key] for key in ("id", "base_id", "group", "kind", "label", "condition", "raw")}
                            | {"original_decision": old, "background": expected["background"]["raw_sha256"]
                               if expected["background"] else None})
                vectors.append(embedding)
    require(len(rows) == 1044, "Every fixed view must be retained")
    return rows, np.asarray(vectors, np.float64), discrepancy


def fit(rows, embeddings, weight):
    clean = np.array([i for i, r in enumerate(rows) if r["condition"] == "clean" and r["raw"] is not None])
    by_source = {rows[i]["base_id"]: i for i in clean}
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        if row["raw"] is not None and row["condition"] in METHOD["fit_conditions"] and row["base_id"] in by_source:
            grouped[row["base_id"]].append(i)
    require(len(by_source) == len(clean) >= 100 and len(grouped) >= 95, "Insufficient paired fit support")
    center = embeddings[clean].mean(axis=0)
    x = (embeddings[clean] - center) / math.sqrt(len(clean))
    shifts = np.asarray([(embeddings[i] - embeddings[by_source[source]]) / math.sqrt(len(grouped) * len(indices))
                        for source, indices in grouped.items() for i in indices])
    clean_gram = x.T @ x
    noise_gram = shifts.T @ shifts
    strength = METHOD["speech_weight"]
    ridge = METHOD["ridge_clean_mean_variance"] * np.trace(clean_gram) / WIDTH
    lhs = clean_gram + strength * noise_gram
    lhs.flat[::WIDTH + 1] += ridge
    target = strength * (noise_gram @ weight)
    correction = np.linalg.solve(lhs, target)
    require(np.isfinite(correction).all() and np.linalg.norm(lhs @ correction - target) < 1e-8,
            "Linear normal-equation verification failed")
    return {"center": center.tolist(), "correction": correction.tolist(), "clean_anchors": len(clean),
            "paired_foregrounds": len(grouped), "matched_shifts": len(shifts), "ridge": float(ridge),
            "clean_rms_logit_change": float(np.linalg.norm(x @ correction)),
            "fit_rms_paired_shift_raw": float(np.linalg.norm(shifts @ weight)),
            "fit_rms_paired_shift_corrected": float(np.linalg.norm(shifts @ (weight - correction)))}


def responses(rows, embeddings, fitted):
    raw = [None if row["raw"] is None else float(logit(np.clip(row["raw"], 1e-6, 1 - 1e-6))) for row in rows]
    corrections = (embeddings - np.asarray(fitted["center"])) @ np.asarray(fitted["correction"])
    return {"raw_point": raw, "paired_correction": [None if score is None else score - float(delta)
                                                   for score, delta in zip(raw, corrections, strict=True)]}


def thresholds(rows, scores):
    result = {}
    for cond in CONDITIONS:
        controls = [(row, score) for row, score in zip(rows, scores, strict=True)
                    if row["label"] == 0 and row["condition"] == cond and row["original_decision"] in ("ai", "inconclusive")]
        budget = sum(row["original_decision"] == "ai" for row, _ in controls)
        require(len(controls) > budget, "Insufficient assessed control headroom")
        scores_sorted = sorted(score for _, score in controls)
        result[cond] = {"controls": len(controls), "incumbent_false_budget": budget, "point": scores_sorted[-budget - 1]}
    return {"cells": result, "point": max(row["point"] for row in result.values())}


def decision(old, response, point):
    return old if old in ("original", "unassessed") else ("ai" if response > point else "inconclusive")


def auc(labels, scores):
    genuine, synthetic = labels.count(0), labels.count(1)
    if not genuine or not synthetic:
        return None
    ranks = rankdata(scores)
    return round(float((sum(r for label, r in zip(labels, ranks, strict=True) if label) - synthetic * (synthetic + 1) / 2)
                       / (genuine * synthetic)), 6)


def metrics(rows, scores, point):
    conditions = {}
    for cond in CONDITIONS:
        tally = Counter()
        labels, values = [], []
        gained = set()
        backgrounds = set()
        for row, response in zip(rows, scores, strict=True):
            if row["condition"] != cond:
                continue
            original = row["original_decision"]
            predicted = decision(original, response, point)
            label = "synthetic" if row["label"] else "genuine"
            tally.update((f"{label}_current_{original}", f"{label}_candidate_{predicted}"))
            if response is not None:
                labels.append(row["label"])
                values.append(response)
            if original != "ai" and predicted == "ai":
                tally[f"{label}_added"] += 1
                if label == "synthetic":
                    gained.add(row["kind"])
                    backgrounds.add(row["background"])
            if original == "ai" and predicted != "ai":
                tally[f"{label}_lost"] += 1
        conditions[cond] = {"counts": dict(tally), "auc": auc(labels, values), "synthetic_recovered_families": sorted(gained),
                            "background_sources_with_recoveries": len(backgrounds - {None})}
    return conditions


def eligible(results):
    projected = results["paired_correction"]
    raw = results["raw_point"]

    def strong(arm, metric):
        return sum(arm[cond]["counts"].get(metric, 0) for cond in STRONG)

    old = strong(projected, "synthetic_current_ai")
    combined = strong(projected, "synthetic_candidate_ai")
    clean = projected["clean"]["counts"]
    checks = {
        "four_net_strong_detections": combined - old >= 4,
        "two_more_than_score_only": combined - strong(raw, "synthetic_candidate_ai") >= 2,
        "no_extra_false_ai_in_strong_cells": all(projected[c]["counts"].get("genuine_candidate_ai", 0)
                                                 <= projected[c]["counts"].get("genuine_current_ai", 0) for c in STRONG),
        "no_extra_clean_false_ai": clean.get("genuine_candidate_ai", 0) <= clean.get("genuine_current_ai", 0),
        "all_existing_clean_ai_flags_preserved": not clean.get("synthetic_lost", 0),
        "recoveries_two_families": len({k for c in STRONG for k in projected[c]["synthetic_recovered_families"]}) >= 2,
    }
    return {"passes": all(checks.values()), "checks": checks}


def run(folder, archive):
    frozen, files, weight, bias = checked(folder)
    started = time.monotonic()
    selected_points = {}
    stages = {}
    for role in ROLES:
        rows, embeddings, error = records(role, archive, frozen, files, weight, bias)
        if role == "fusion_fit":
            fitted = fit(rows, embeddings, weight)
            write(folder / "fitted-private.json", {"freeze_sha256": digest(folder / "freeze.json"), **fitted})
            continue
        values = responses(rows, embeddings, fitted)
        if role == "confidence_cal":
            selected_points = {key: thresholds(rows, value) for key, value in values.items()}
            write(folder / "points.json", {"fitted_sha256": digest(folder / "fitted-private.json"),
                                          "points": selected_points})
        decision_points = {key: selected_points[key]["point"] for key in values}
        measured = {key: metrics(rows, value, decision_points[key]) for key, value in values.items()}
        stages[role] = {"conditions": measured, "advance": eligible(measured), "cases": len(rows),
                        "max_original_raw_reconstruction_error": error}
        ledger = [{"label": row["label"], "kind": row["kind"], "condition": row["condition"],
                   "old": row["original_decision"], "raw_point": values["raw_point"][i],
                   "paired_correction": values["paired_correction"][i]} for i, row in enumerate(rows)]
        write(folder / f"{role}-scalar-ledger.json", ledger)
        write(folder / f"{role}.json", stages[role])
        print(json.dumps({"role": role, "cases": len(rows), **stages[role]["advance"]}), flush=True)
    result = {"schema": "chipmunk.audio-paired-nuisance-result/1", "freeze_sha256": digest(folder / "freeze.json"),
              "fit": {k: v for k, v in fitted.items() if k not in ("center", "correction")},
              "fitted_sha256": digest(folder / "fitted-private.json"), "points_sha256": digest(folder / "points.json"),
              "advance": all(stages[role]["advance"]["passes"] for role in ROLES[2:]),
              "roles": stages, "elapsed_s": time.monotonic() - started, "new_neural_forwards": 0}
    write(folder / "aggregate.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("output", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else run)(args.output, args.archive)


if __name__ == "__main__":
    main()
