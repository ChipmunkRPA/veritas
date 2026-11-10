"""Retain video-check identity in a jointly fitted dependent-evidence combiner.

Post-result development on already public scalars; no new inference or default.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import psutil

from app.bench import specialists as engine

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
TRIAL = ROOT / "backend/data/independent-evaluation/video-joint-scalars-v106"
PUBLIC = PANEL / "video-joint-scalar-freeze-v106.json"
VIEWS = ("face-crop", "face-appearance", "face-feature-change")
ARMS = {"appearance_only": ("face-appearance",), "joint_views": VIEWS}
FILES = (
    "video-face-specialists-v20-release.json",
    "video-face-specialists-v20-manifests.json.gz",
    "video-face-specialists-v20-all-artifact.json",
    "video-face-specialists-v20-all-points.json",
    "video-face-specialists-v20-result.json",
)
CODE = (
    "backend/scripts/video_joint_scalar_v106.py",
    "backend/tests/test_video_joint_scalar_v106.py",
    "backend/app/bench/specialists.py",
    "docs/video-joint-scalars-v106-protocol.md",
)
RAW_APPEARANCE_POINT = 0.11243900750879454


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def features(file, heads, views):
    result = []
    for view in views:
        units = [u for u in file["units"] if u["specialist"] == view]
        head = heads.get(view + "|single-face")
        if not units or not head or head["status"] != "fitted" or any(u["status"] != "ok" for u in units):
            return None
        values = [engine._logit(head, [u["raw"]]) for u in units]
        result.append(max(0.0, max(values)))
    return result


def probability(model, values):
    return 1 / (1 + math.exp(-max(-700.0, min(700.0, engine._logit(model, values)))))


def fit(files, heads, views, minimum=25):
    def available(role):
        selected = [(f, features(f, heads, views)) for f in files if f["role"] == role]
        return [(f, x) for f, x in selected if x is not None]

    training = available("fusion_fit")
    model = engine._fit(
        [x for f, x in training],
        [f["label"] for f, x in training],
        [f["group"] for f, x in training],
        minimum=minimum,
        l2=0.001,
    )
    require(model["status"] == "fitted", "Joint response fitting lacks complete source support")
    calibration = available("fusion_cal")
    calibrated = engine._fit(
        [[engine._logit(model, x)] for f, x in calibration],
        [f["label"] for f, x in calibration],
        [f["group"] for f, x in calibration],
        minimum=minimum,
        l2=0.001,
    )
    require(calibrated["status"] == "fitted", "Separate file calibration lacks complete source support")
    return {
        "views": list(views),
        "response": model,
        "calibration": calibrated,
        "fitting_files": len(training),
        "calibration_files": len(calibration),
        "reference_prior": 0.5,
    }


def predict(file, heads, model):
    x = features(file, heads, model["views"])
    if x is None:
        return None
    return probability(model["calibration"], [engine._logit(model["response"], x)])


def old_predict(file, artifact):
    x = engine._features(file, artifact)
    if x is None:
        return None
    model = artifact["combiners"][engine.pattern(file)]
    return probability(model["calibration"], [engine._logit(model["response"], x)])


def complete_genuine_groups(files, predictions):
    grouped = defaultdict(list)
    for file in files:
        if file["role"] == "fusion_cal" and file["label"] == 0:
            grouped[file["group"]].append(predictions[file["id"]])
    return {g: max(values) for g, values in grouped.items() if values and all(v is not None for v in values)}


def point(values, common, budget=3, minimum=25):
    require(
        len(common) >= minimum and type(budget) is int and 0 <= budget < len(common),
        "Common complete genuine calibration groups required",
    )
    require(
        all(engine.finite(values.get(g)) and 0 <= values[g] <= 1 for g in common),
        "Finite observed calibration probabilities required",
    )
    cut = sorted(values[g] for g in common)[len(common) - budget - 1]
    return {
        "threshold": cut,
        "common_genuine_groups": len(common),
        "budget": budget,
        "groups_flagged": sum(values[g] > cut for g in common),
        "comparison": "strict_greater_than",
    }


def summarize(files, predictions, cut):
    result = Counter(
        dict.fromkeys(
            (
                "assessed",
                "inconclusive",
                "ai",
                "genuine",
                "ai_flagged",
                "missed_ai_including_inconclusive",
                "false_ai",
            ),
            0,
        )
    )
    result["planned"] = len(files)
    groups = set()
    loss = {0: defaultdict(list), 1: defaultdict(list)}
    for file in files:
        y = file["label"]
        p = predictions[file["id"]]
        flag = p is not None and p > cut
        result["ai" if y else "genuine"] += 1
        result["inconclusive" if p is None else "assessed"] += 1
        if y:
            result["ai_flagged" if flag else "missed_ai_including_inconclusive"] += 1
        elif flag:
            result["false_ai"] += 1
            groups.add(file["group"])
        if p is not None:
            loss[y][file["group"]].append((p - y) ** 2)
    means = [
        sum(sum(v) / len(v) for v in cells.values()) / len(cells) if cells else None
        for cells in loss.values()
    ]
    return {
        **result,
        "false_ai_groups": len(groups),
        "source_class_balanced_brier": sum(means) / 2 if all(v is not None for v in means) else None,
    }


def raw_appearance(file):
    units = [u for u in file["units"] if u["specialist"] == "face-appearance"]
    if not units or any(u["status"] != "ok" for u in units):
        return None
    require(all(engine.finite(u["raw"]) for u in units), "Finite assessed appearance responses required")
    return float(any(u["raw"] > RAW_APPEARANCE_POINT for u in units))


def identity():
    return {
        "schema": "chipmunk.video-joint-scalar-freeze/106",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_sha256": {p: sha(PANEL / p) for p in FILES},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "psutil")},
        "arms": {k: list(v) for k, v in ARMS.items()},
        "genuine_calibration_group_budget": 3,
        "minimum_groups_per_class": 25,
        "new_neural_calls": 0,
        "scope": "Post-result development on consumed public v20 scalar observations. Existing local heads unchanged; new per-view file response and separate calibration. Not fresh confirmation, localization validation, full-app accuracy or customer probability.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def run():
    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged comparison freeze required")
    output = TRIAL / "evaluation"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    observations = []

    def observe(initial=False):
        row = {
            "elapsed_s": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "disk_free_bytes": shutil.disk_usage(output).free,
        }
        observations.append(row)
        minimum = 3 * 1024**3 if initial else 1536 * 1024**2
        require(
            row["elapsed_s"] <= 900
            and row["rss_bytes"] <= 1536 * 1024**2
            and min(row["available_ram_bytes"], row["disk_free_bytes"]) >= minimum,
            "Data-only comparison bound exceeded",
        )

    observe(initial=True)
    released = read(PANEL / "video-face-specialists-v20-release.json")
    require(
        all(sha(ROOT / p) == released["files_sha256"][p] for p in released["files_sha256"]),
        "Original public release changed",
    )
    with gzip.open(PANEL / "video-face-specialists-v20-manifests.json.gz", "rt") as stream:
        manifest = json.load(stream)["all"]
    audit = engine.audit(manifest)
    require(audit["files"] == 648, "Complete public scalar panel required")
    files = manifest["files"]
    artifact = read(PANEL / "video-face-specialists-v20-all-artifact.json")
    heads = artifact["specialists"]
    models = {name: fit(files, heads, views) for name, views in ARMS.items()}
    observe()
    calibration = [f for f in files if f["role"] == "fusion_cal"]
    cal_predictions = {
        name: {f["id"]: predict(f, heads, model) for f in calibration} for name, model in models.items()
    }
    cal_predictions["old_family_max"] = {f["id"]: old_predict(f, artifact) for f in calibration}
    genuine = {name: complete_genuine_groups(calibration, values) for name, values in cal_predictions.items()}
    common = set.intersection(*(set(rows) for rows in genuine.values()))
    points = {name: point(genuine[name], common) for name in ARMS}
    write(
        output / "fitted-models.json",
        {"models": models, "points": points, "source_local_heads_sha256": engine.digest(heads)},
    )
    write(
        output / "fit-seal.json",
        {
            "fitted_models_sha256": sha(output / "fitted-models.json"),
            "freeze_sha256": sha(TRIAL / "freeze.json"),
        },
    )
    observe()
    selected = [f for f in files if f["role"] == "confirmation"]
    require(len(selected) == 160, "All reserved development cases required")
    predictions = {
        name: {f["id"]: predict(f, heads, model) for f in selected} for name, model in models.items()
    }
    predictions["old_family_max"] = {f["id"]: old_predict(f, artifact) for f in selected}
    predictions["raw_appearance"] = {f["id"]: raw_appearance(f) for f in selected}
    old_point = read(PANEL / "video-face-specialists-v20-all-points.json")["file"]["threshold"]
    cuts = {
        **{name: p["threshold"] for name, p in points.items()},
        "old_family_max": old_point,
        "raw_appearance": 0.5,
    }
    results = {name: summarize(selected, values, cuts[name]) for name, values in predictions.items()}
    results["raw_appearance"].pop("source_class_balanced_brier")
    previous = read(PANEL / "video-face-specialists-v20-result.json")
    for name, reference in [
        ("old_family_max", previous["arms"]["all"]["all"]),
        ("raw_appearance", previous["raw_cue_references"]["face-appearance"]["all"]),
    ]:
        require(
            all(
                results[name].get(k, 0) == reference.get(k, 0)
                for k in (
                    "ai",
                    "genuine",
                    "ai_flagged",
                    "false_ai",
                    "false_ai_groups",
                    "inconclusive",
                    "missed_ai_including_inconclusive",
                )
            ),
            "Historical reference counts do not reproduce",
        )
    write(output / "predictions.json", predictions)
    require(
        sha(output / "fitted-models.json") == read(output / "fit-seal.json")["fitted_models_sha256"],
        "Fitted model changed after seal",
    )
    require(not {"torch", "transformers"}.intersection(sys.modules), "No neural runtime allowed")
    observe()
    result = {
        "schema": "chipmunk.video-joint-scalar-result/106",
        "status": "complete",
        "source_files": 648,
        "evaluated_files": 160,
        "known_evaluation_source_groups": len({f["group"] for f in selected}),
        "results": results,
        "common_genuine_calibration_groups": len(common),
        "new_points": points,
        "fitting_support": {
            name: {k: model[k] for k in ("fitting_files", "calibration_files")}
            for name, model in models.items()
        },
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "fitted_model_sha256": sha(output / "fitted-models.json"),
        "prediction_journal_sha256": sha(output / "predictions.json"),
        "duration_s": time.monotonic() - started,
        "maximum_observed_self_rss_bytes": max(r["rss_bytes"] for r in observations),
        "minimum_available_ram_bytes": min(r["available_ram_bytes"] for r in observations),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": lock["scope"],
    }
    write(output / "result.json", result)
    write(PANEL / "video-joint-scalar-result-v106.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    freeze() if parser.parse_args().command == "freeze" else run()
