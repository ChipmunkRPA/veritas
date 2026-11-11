"""Fit equal-size speech layer heads and replay fixed points on separate inputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from scripts.audio_layer_public_codec_v12 import unpack
from scripts.audio_multisource_v8 import balanced_weights

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
BASE = "audio-v12-development-review.json.gz"
SOURCE = "audio-v9-identity-plan.json.gz"
ARMS = ("raw_point", "concatenated", "relation", "relation_unpaired")
DIRECTIONS = ("ai", "original", "inconclusive", "unassessed")
PENALTY = 0.1
DIMS = 64
LAYERS = [8, 16, 24]
FIT_RECIPE = {
    "layers": [8, 24], "projection_seed": 61221, "projected_dimensions_per_layer": DIMS,
    "concatenated": "Standardized early and late projections",
    "relation": "Absolute difference and product of the same standardized projections",
    "features": "Each head takes baseline raw logit, 128 standardized features, and intercept",
    "regularization": {"raw": .001, "features": .05, "consistency": PENALTY},
    "consistency": "Mean squared logit differences from clean to other conditions within each fitting source, equally weighted by source",
    "point": "Strictly above the finite-sample 95% rank of maxima over assessed genuine calibration groups",
    "decision": "Only an assessed incumbent-inconclusive input may become AI",
    "advance": "At least two distinct fresh noisy synthetic groups recovered, at least one per scene, with at most one additional genuine flag, at least six assessed genuine views per scene, and more groups recovered than raw point and concatenation.",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def projection() -> np.ndarray:
    rng = np.random.default_rng(FIT_RECIPE["projection_seed"])
    return rng.choice([-1., 1.], (1024, DIMS)) / 32


def project(layers, matrix) -> np.ndarray:
    value = np.asarray(layers, np.float64)
    if value.shape != (3, 1024) or not np.isfinite(value).all():
        raise ValueError("Exact finite 8/16/24 pooled state vectors required")
    return value[[0, 2]] @ matrix


def _columns(values: np.ndarray, arm: str) -> np.ndarray:
    a, b = values[:, 0], values[:, 1]
    return np.hstack([a, b] if arm == "concatenated" else [np.abs(a-b), a*b])


def _stats(x: np.ndarray, weights: np.ndarray):
    center = np.tensordot(weights, x, axes=1)
    spread = np.sqrt(np.tensordot(weights, (x-center)**2, axes=1))
    return center, np.maximum(spread, 1e-6)


def _raw(rows):
    return logit(np.clip([row["raw"] for row in rows], 1e-6, 1-1e-6))


def _clean_pair_weights(rows):
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        variants = groups[row["base_id"]]
        if row["condition"] in variants:
            raise ValueError("Repeated fitted condition")
        variants[row["condition"]] = i
    pairs = [[(variants["clean"], i) for name, i in variants.items() if name != "clean"]
             for variants in groups.values() if "clean" in variants and len(variants) > 1]
    if not pairs:
        raise ValueError("Paired clean and transformed training inputs required")
    indices = np.asarray([pair for source in pairs for pair in source])
    weights = np.asarray([1 / (len(pairs)*len(source)) for source in pairs for _ in source])
    return indices, weights


def fit(rows, arm):
    if arm not in ARMS[1:] or any(row["role"] != "fit" for row in rows):
        raise ValueError("Only the original fit partition may learn feature parameters")
    rows = [row for row in rows if row["raw"] is not None]
    weights = balanced_weights(rows)
    layers = np.array([row["projected"] for row in rows])
    lmean, lstd = _stats(layers, weights)
    features = _columns((layers - lmean) / lstd, arm)
    mean, std = _stats(features, weights)
    standardized = (features-mean) / std * math.sqrt(16 / features.shape[1])
    x = np.column_stack((_raw(rows), standardized, np.ones(len(rows))))
    labels = np.array([row["label"] for row in rows])
    pairs, pweights = _clean_pair_weights(rows)
    difference = x[pairs[:, 1]] - x[pairs[:, 0]]
    penalty = 0 if arm == "relation_unpaired" else PENALTY
    ridge = np.array([.001, *([.05]*features.shape[1]), 0.])

    def objective(coef):
        z, d = x @ coef, difference @ coef
        loss = weights @ (np.logaddexp(0, z) - labels*z)
        grad = x.T @ (weights * (expit(z) - labels))
        value = loss + .5 * (ridge @ coef**2 + penalty * (pweights @ d**2))
        return value, grad + ridge*coef + penalty * (difference.T @ (pweights*d))

    initial = np.zeros(x.shape[1])
    initial[0] = 1
    solved = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(0, None)] + [(None, None)]*(len(initial)-1),
                      options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000})
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError(f"Head fit failed for {arm}: {solved.message}")
    return {"layer_center": lmean.tolist(), "layer_scale": lstd.tolist(),
            "center": mean.tolist(), "scale": std.tolist(), "coefficients": solved.x.tolist(),
            "fit_rows": len(rows), "fit_groups": len({row["group"] for row in rows}),
            "paired_conditions": len(pairs), "loss": float(solved.fun)}


def score(rows, arm, model=None):
    admitted = [r for r in rows if r["raw"] is not None]
    if not admitted:
        return [None] * len(rows)
    raw = _raw(admitted)
    if arm != "raw_point":
        layers = np.array([r["projected"] for r in admitted])
        values = _columns((layers-model["layer_center"])/model["layer_scale"], arm)
        x = (values-model["center"])/model["scale"] * math.sqrt(16 / values.shape[1])
        raw = np.column_stack((raw, x, np.ones(len(x)))) @ model["coefficients"]
    values = iter(raw.tolist())
    return [None if row["raw"] is None else next(values) for row in rows]


def point(rows, scores):
    if len(rows) != len(scores) or any(row["role"] != "cal" for row in rows):
        raise ValueError("Only the separate calibration partition may set operating points")
    maxima = {}
    for row, response in zip(rows, scores, strict=True):
        if row["label"] == 0 and response is not None:
            group = row["group"]
            maxima[group] = max(response, maxima.get(group, -math.inf))
    if len(maxima) < 20:
        raise ValueError("At least 20 assessed genuine calibration groups required")
    rank = min(len(maxima), math.ceil((len(maxima)+1)*.95))
    return {"value": sorted(maxima.values())[rank-1], "genuine_groups": len(maxima), "rank": rank}


def _auc(rows, values):
    measured = [(r["label"], v) for r, v in zip(rows, values, strict=True) if v is not None]
    n1 = sum(label for label, _ in measured)
    n0 = len(measured)-n1
    if not n1 or not n0:
        return None
    ranks = rankdata([v for _, v in measured])
    rank_sum = sum(rank for rank, (label, _) in zip(ranks, measured, strict=True) if label)
    return round(float((rank_sum - n1*(n1+1)/2)/(n0*n1)), 6)


def summarize(rows, values, limit):
    counts = defaultdict(Counter)
    rescued = set()
    added_false = set()
    for r, response in zip(rows, values, strict=True):
        old = r["decision"]
        if old not in DIRECTIONS or (response is None) != (old == "unassessed"):
            raise ValueError("Measured coverage and existing decision disagree")
        promoted = old == "inconclusive" and response > limit
        chosen = "ai" if promoted else old
        result = counts[r["condition"]]
        label = "synthetic" if r["label"] else "genuine"
        result[f"{label}_planned"] += 1
        result[f"{label}_current_{old}"] += 1
        result[f"{label}_candidate_{chosen}"] += 1
        if promoted:
            result[f"added_{label}"] += 1
            (rescued if r["label"] else added_false).add(r["group"])
    for condition, result in counts.items():
        positions = [i for i, row in enumerate(rows) if row["condition"] == condition]
        result["auc"] = _auc([rows[i] for i in positions], [values[i] for i in positions])
    return {"conditions": dict(counts), "rescued_synthetic_groups": len(rescued),
            "added_genuine_groups": len(added_false)}


def features(role, review, catalogue, matrix):
    packet = review["feature_packets"][role]
    path = PANEL / packet["name"]
    if digest(path) != packet["sha256"]:
        raise ValueError("The exact public role packet is required")
    metadata = {r["id"]: r for r in catalogue["variants"]["development"] if r["role"] == role}
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header["role"] != role or header["records"] != len(metadata):
            raise ValueError("Public source role or record count differs")
        for line in stream:
            packed = json.loads(line)["record"]
            row = metadata.pop(packed["id"])
            if any(packed[k] != row[k] for k in ("id", "role", "sha256", "pcm_sha256")):
                raise ValueError("Source and feature identities differ")
            projected = None
            if packed["raw"] is not None:
                if packed["layer_ids"] != LAYERS:
                    raise ValueError("Unexpected native layer order")
                projected = project(unpack(packed["layer_embeddings"]), matrix).tolist()
            records.append({**row, "base_id": row["base_id_sha256"], "raw": packed["raw"], "projected": projected})
    if metadata or len(records) != header["records"]:
        raise ValueError("Public role coverage differs")
    return records


def execute(args):
    review, catalogue, matrix = read(PANEL / BASE), read(PANEL / SOURCE), projection()
    roles = ("fit", "cal", "development")
    commitments = {name: digest(PANEL / name) for name in (BASE, SOURCE, *(review["feature_packets"][r]["name"] for r in roles))}
    lock = {"schema": "chipmunk.audio-cross-layer-experiment/1", "source_digests": commitments,
            "code_sha256": digest(Path(__file__)), "recipe": FIT_RECIPE}
    output = args.output
    if args.action == "fit":
        write(output / "freeze.json", lock)
        fit_rows, cal_rows = [features(role, review, catalogue, matrix) for role in roles[:2]]
        models = {arm: fit(fit_rows, arm) for arm in ARMS[1:]}
        points = {arm: point(cal_rows, score(cal_rows, arm, models.get(arm))) for arm in ARMS}
        write(output / "fitted.json", {"lock": lock, "heads": models, "points": points})
        return
    fitted = read(output / "fitted.json")
    if fitted["lock"] != lock:
        raise ValueError("Fit receipt or method differs")
    held = features("development", review, catalogue, matrix)
    decisions = {row["id"]: row["original_decision"] for row in review["outcomes"]}
    for r in held:
        observed = decisions.pop(r["id"])
        r["decision"] = "unassessed" if r["raw"] is None else observed
    if decisions:
        raise ValueError("Historical outcomes do not match every held row")
    results = {arm: summarize(held, score(held, arm, fitted["heads"].get(arm)), fitted["points"][arm]["value"])
               for arm in ARMS}
    write(output / "public-readout.json", {"fitted_sha256": digest(output / "fitted.json"),
          "planned": len(held), "arms": results, "elapsed_s": time.monotonic()-args.started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "readout"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.started = time.monotonic()
    execute(args)


if __name__ == "__main__":
    main()
