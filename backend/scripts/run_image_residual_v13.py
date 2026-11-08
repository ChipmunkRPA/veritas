"""Freeze and execute a consumed-source image-feature development comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import image_residual_localizer_v13 as method
from scripts import run_image_confirmation_v12 as v12
from scripts import run_image_supported_v11 as v11

ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "backend/scripts/image_residual_localizer_v13.py",
    "backend/scripts/run_image_residual_v13.py",
    "backend/scripts/image_residual_runtime_v13.py",
    "backend/scripts/specialist_runtime_v4.py",
    "backend/tests/test_image_residual_v13.py",
    "docs/image-residual-v13-protocol.md",
    "backend/scripts/run_image_supported_v11.py",
    "backend/scripts/run_image_confirmation_v12.py",
    "backend/scripts/prepare_image_supported_v11.py",
)
POLICY = {
    "version": 13,
    "source_scope": "Previously consumed project sources only; no fresh accuracy claim.",
    "fit": "120 v11 source groups (specialist_cal and fusion_fit); exact source RGB masks, original/lighting/jpeg55 variants and genuine ordinary-patch controls.",
    "point": "60 source-disjoint v11 fusion_cal groups; at most5% genuine groups with original or ordinary-patch response strictly above point; no AI outcomes used.",
    "diagnostic": "60 v11 confirmation plus48 consumed v12 groups; no role changes or case omission.",
    "arms": ["residual_features", "residual_features_projected"],
    "projection": "Remove top2 fit-genuine standardized intervention-delta covariance directions. Same augmented fit samples, L2=.01, logistic ranking head in both arms.",
    "localization": "Source-resolution measurements pooled to64x64. Median3x3 response,8-connected components with>=16 cells. Edge-aligned source coordinate projection.",
    "limits": "No pretrained weights or neural forwards. No confidence calibration, full-image authorship, fresh-source, unseen-generator, external reviewer or serving admission claim.",
    "continuation": "Require higher mean edited IoU and more IoU>=.3 cases than current, no extra genuine flags, and ordinary-control flags<=5%; passing needs real baseline ordinary controls and fresh confirmation before any promotion.",
    "serving_admitted": False,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def records(path):
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    method.require(len(rows) == header["planned"], "Complete native reference journal required")
    return {r["id"]: r for r in rows}


def prepare(main, trial):
    trial.mkdir(parents=True, exist_ok=False)
    parent = main / "backend/data/independent-evaluation/image-supported-v11"
    recent = main / "backend/data/independent-evaluation/image-fresh-v12"
    old_plan, recent_plan = read(parent / "source-plan.json"), read(recent / "input-plan.json")
    bound = [
        parent / "source-plan.json",
        parent / "freeze.json",
        recent / "input-plan.json",
        recent / "result.json",
        recent / "source-arithmetic-audit.json",
        recent / "native.jsonl",
    ]
    baseline, entries = {}, []
    rolemap = {
        "specialist_cal": "fit",
        "fusion_fit": "fit",
        "fusion_cal": "point",
        "confirmation": "diagnostic",
    }
    families = {g["id"]: g["items"][0]["item"]["family"] for g in old_plan["groups"]}
    for role in rolemap:
        path = parent / f"{role}-jpeg.jsonl"
        bound.append(path)
        baseline.update(records(path))
    recent_baseline = records(recent / "native.jsonl")
    for cohort, rows, references in [
        ("v11", [r for r in old_plan["files"] if r["format"] == "jpeg"], baseline),
        ("consumed-v12", recent_plan["files"], recent_baseline),
    ]:
        for row in rows:
            ref = references[row["id"]]
            method.require(
                ref["input_sha256"] == row["input_sha256"]
                and ref["rgb_sha256"] == row["rgb_sha256"]
                and ref["status"] == "ok"
                and ref["current_reliable_cells"] > 0,
                "Assessed bound current reference required",
            )
            entries.append(
                {
                    "key": cohort + "-" + row["id"],
                    "cohort": cohort,
                    "role": rolemap[row["role"]] if cohort == "v11" else "diagnostic",
                    "generator": families[row["group"]] if cohort == "v11" else row["generator"],
                    "source": row,
                    "current": {"flag": ref["current_region_flag"], **ref["current_mask_metrics"]},
                }
            )
    groups = defaultdict(set)
    for entry in entries:
        groups[entry["role"]].add(entry["source"]["group"])
    method.require(
        {k: len(v) for k, v in groups.items()} == {"fit": 120, "point": 60, "diagnostic": 108}
        and sum(map(len, groups.values())) == len(set.union(*groups.values())),
        "Disjoint complete source roles required",
    )
    planned = sum(
        3 + (e["source"]["label"] == 0) if e["role"] == "fit" else 1 + (e["source"]["label"] == 0)
        for e in entries
    )
    write(
        trial / "plan.json",
        {
            "policy": POLICY,
            "entries": entries,
            "feature_cases_planned": planned,
            "source_group_counts": {k: len(v) for k, v in groups.items()},
            "source_plans": {"v11": str(parent / "source-plan.json")},
            "bound_prior_files": {str(p): digest(p) for p in bound},
        },
    )
    write(
        trial / "freeze.json",
        {
            "policy": POLICY,
            "plan_sha256": digest(trial / "plan.json"),
            "code_sha256": {p: digest(ROOT / p) for p in CODE},
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow")},
            "new_feature_extractions_before_freeze": 0,
            "new_neural_forwards": 0,
            "serving_admitted": False,
        },
    )
    print(
        json.dumps(
            {
                "prepared_source_groups": {k: len(v) for k, v in groups.items()},
                "planned_feature_cases": planned,
            }
        ),
        flush=True,
    )


def check(trial):
    plan, lock = read(trial / "plan.json"), read(trial / "freeze.json")
    method.require(
        lock["policy"] == POLICY == plan["policy"]
        and lock["plan_sha256"] == digest(trial / "plan.json")
        and lock["code_sha256"] == {p: digest(ROOT / p) for p in CODE}
        and lock["packages"] == {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow")}
        and all(digest(p) == sha for p, sha in plan["bound_prior_files"].items()),
        "Frozen source/code/runtime changed",
    )
    return plan


def load(plan, entry):
    row = entry["source"]
    data, rgb, truth = (
        v11.input_for(read(plan["source_plans"]["v11"]), row)
        if entry["cohort"] == "v11"
        else v12.input_for(row)
    )
    method.require(
        bool(truth.any()) == bool(row["label"])
        and truth.shape == rgb.shape[:2]
        and hashlib.sha256(data).hexdigest() == row["input_sha256"],
        "Exact bound source truth required",
    )
    return rgb, truth


def run(trial):
    plan = check(trial)
    cache = trial / "features"
    cache.mkdir(exist_ok=False)
    x, y, deltas, cases, done = [], [], [], [], 0
    journal = (trial / "features.jsonl").open("x")
    try:
        for entry in plan["entries"]:
            rgb, truth = load(plan, entry)
            soft = method.grid(truth.astype(np.float32))
            row = entry["source"]
            seed = int(hashlib.sha256(entry["key"].encode()).hexdigest()[:8], 16)
            indices = method.sample_indices(soft, seed)
            variants = ["original", "lighting", "jpeg55"] if entry["role"] == "fit" else ["original"]
            if row["label"] == 0:
                variants += ["ordinary_patch"]
            original_samples = None
            for variant in variants:
                pixels = method.intervene(rgb, variant)
                values = method.features(pixels)
                record = {
                    "key": entry["key"],
                    "variant": variant,
                    "role": entry["role"],
                    "group": row["group"],
                    "cohort": entry["cohort"],
                    "generator": entry["generator"],
                    "label": row["label"],
                    "source_input_sha256": row["input_sha256"],
                    "rgb_sha256": method.sha_array(pixels),
                    "truth_sha256": method.sha_array(truth),
                    "features_sha256": method.sha_array(values),
                    "width": rgb.shape[1],
                    "height": rgb.shape[0],
                }
                if entry["role"] == "fit":
                    samples = values.reshape(-1, len(method.FEATURES))[indices]
                    x.append(samples)
                    y.append(soft.reshape(-1)[indices])
                    if variant == "original":
                        original_samples = samples
                    elif variant in ("lighting", "jpeg55") and row["label"] == 0:
                        deltas.append(samples - original_samples)
                else:
                    name = hashlib.sha256((entry["key"] + variant).encode()).hexdigest() + ".npz"
                    path = cache / name
                    np.savez_compressed(path, features=values)
                    record.update(features_path=str(path), features_file_sha256=digest(path))
                    cases.append(record)
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                done += 1
                if done % 40 == 0:
                    print(
                        json.dumps(
                            {"feature_cases_complete": done, "planned": plan["feature_cases_planned"]}
                        ),
                        flush=True,
                    )
    finally:
        journal.close()
    method.require(done == plan["feature_cases_planned"], "All predeclared feature cases required")
    x, y = np.concatenate(x), np.concatenate(y)
    np.savez_compressed(trial / "fit-samples-private.npz", x=x, y=y, deltas=np.concatenate(deltas))
    heads = {name: method.fit(x, y, deltas, rank) for name, rank in zip(POLICY["arms"], (0, 2), strict=True)}
    for name, head in heads.items():
        maxima = defaultdict(list)
        for case in cases:
            if case["role"] == "point" and case["label"] == 0:
                with np.load(case["features_path"], allow_pickle=False) as data:
                    maxima[case["group"]].append(float(method.response(data["features"], head).max()))
        head["point"] = method.point([max(v) for v in maxima.values()])
        write(trial / (name + "-head-private.json"), head)
    entries = {e["key"]: e for e in plan["entries"]}
    outcomes = []
    for case in cases:
        entry = entries[case["key"]]
        _, truth = load(plan, entry)
        method.require(
            method.sha_array(truth) == case["truth_sha256"]
            and digest(case["features_path"]) == case["features_file_sha256"],
            "Source/cache binding changed",
        )
        with np.load(case["features_path"], allow_pickle=False) as data:
            values = data["features"]
        method.require(method.sha_array(values) == case["features_sha256"], "Feature array changed")
        for name, head in heads.items():
            scores = method.response(values, head)
            prediction = method.mask(scores, head["point"]["value"], case["width"], case["height"])
            score_path = cache / (
                hashlib.sha256((case["key"] + case["variant"] + name).encode()).hexdigest() + ".scores.npz"
            )
            np.savez_compressed(score_path, scores=scores)
            outcomes.append(
                {
                    **{
                        k: case[k]
                        for k in (
                            "key",
                            "variant",
                            "role",
                            "group",
                            "cohort",
                            "generator",
                            "label",
                            "truth_sha256",
                        )
                    },
                    "arm": name,
                    "scores_path": str(score_path),
                    "scores_file_sha256": digest(score_path),
                    "mask_sha256": method.sha_array(prediction),
                    "metrics": method.metrics(prediction, truth),
                }
            )
    write(trial / "outcomes-private.json", outcomes)
    report = {
        "policy": POLICY,
        "freeze_sha256": digest(trial / "freeze.json"),
        "features_journal_sha256": digest(trial / "features.jsonl"),
        "feature_cases_completed": done,
        "fit_sample_cells": len(y),
        "genuine_intervention_delta_cells": sum(map(len, deltas)),
        "heads": {
            name: {k: head[k] for k in ("objective", "iterations", "removed_dimensions", "point")}
            for name, head in heads.items()
        },
        "diagnostic": {},
        "new_neural_forwards": 0,
        "outside_human_reviews": 0,
        "serving_admitted": False,
    }
    for cohort in ("all", "v11", "consumed-v12"):
        selected = [
            e
            for e in entries.values()
            if e["role"] == "diagnostic" and (cohort == "all" or e["cohort"] == cohort)
        ]
        current = summary(
            [
                {
                    "label": e["source"]["label"],
                    "variant": "original",
                    "metrics": {
                        **e["current"],
                        "selected_area_fraction": e["current"]["predicted_positive_pixels"]
                        / e["current"]["pixels"],
                    },
                }
                for e in selected
            ]
        )
        arms = {
            name: summary(
                [
                    r
                    for r in outcomes
                    if r["role"] == "diagnostic"
                    and r["arm"] == name
                    and (cohort == "all" or r["cohort"] == cohort)
                ]
            )
            for name in heads
        }
        report["diagnostic"][cohort] = {"current_originals_only": current, "arms": arms}
    reference = report["diagnostic"]["all"]["current_originals_only"]
    for name, row in report["diagnostic"]["all"]["arms"].items():
        gates = {
            "higher_mean_iou": row["edited_mean_iou"] > reference["edited_mean_iou"],
            "more_iou_0_3": row["edited_iou_at_least_0_3"] > reference["edited_iou_at_least_0_3"],
            "no_extra_genuine_flags": row["genuine_flags"] <= reference["genuine_flags"],
            "ordinary_controls_at_most_5pct": row["ordinary_flags"] <= 0.05 * row["ordinary_inputs"],
        }
        row.update(continuation_gates=gates, development_continuation_eligible=all(gates.values()))
    check(trial)
    write(trial / "result.json", report)
    print(
        json.dumps({"completed": done, "diagnostic": report["diagnostic"]["all"], "serving_admitted": False}),
        flush=True,
    )


def summary(rows):
    ai = [r["metrics"] for r in rows if r["label"] == 1 and r["variant"] == "original"]
    genuine = [r["metrics"] for r in rows if r["label"] == 0 and r["variant"] == "original"]
    ordinary = [r["metrics"] for r in rows if r["variant"] == "ordinary_patch"]
    return {
        "edited_inputs": len(ai),
        "edited_flagged": sum(r["flag"] for r in ai),
        "edited_iou_at_least_0_3": sum(r["iou"] >= 0.3 for r in ai),
        "edited_mean_iou": float(np.mean([r["iou"] for r in ai])),
        "edited_mean_pixel_f1": float(np.mean([r["pixel_f1"] for r in ai])),
        "genuine_inputs": len(genuine),
        "genuine_flags": sum(r["flag"] for r in genuine),
        "ordinary_inputs": len(ordinary),
        "ordinary_flags": sum(r["flag"] for r in ordinary),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("main", type=Path)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.main.resolve(), args.trial.resolve())
    else:
        run(args.trial.resolve())
