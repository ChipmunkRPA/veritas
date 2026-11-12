"""Compare frozen temporal voice heads on exact partial edits and competing speech."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.prepare_audio_specialists_v15 import ASSETS

RATE = 16_000
SAMPLES = RATE * 4
CELL = 3_200
CELLS = SAMPLES // CELL
RANGE_CELLS = 3
WIDTH = 1_024
ROLES = ("fusion_fit", "confidence_cal", "selection", "confirmation")
ARMS = ("incumbent_clip", "incumbent_frames", "clean_head", "mixture_head", "paired_head")
HUMAN = ("ordinary_same_host", "ordinary_other_voice")
SYNTHETIC = "synthetic"
SCHEMA = "chipmunk.audio-partial-residual-transfer/1"
METHOD = {
    "model": "The exact installed float32 voice model and its learned softmax mixture of all frozen encoder layers. One native 4 s forward per distinct PCM.",
    "geometry": "First paired fbank frame center is sample 280, then every 320 samples. Average frames by their centers in 20 successive 3,200-sample cells. Cells describe encoder timestamps; each feature may depend on the wider recording.",
    "baseline": "Original pooled clip logit and the existing affine head applied to the unrefitted cell vectors. The latter is a local inspection comparator, with no new learned parameters.",
    "features": "Use the per-cell hidden vector plus its difference from the recording-wide cell mean. Fit-only feature means and standard deviations, with a 1e-4 scale floor.",
    "heads": "Three L2-regularized linear heads (0.035). Balance positive and negative sample overlap mass. Fit on only matched splice triples, dry only for clean_head and all noise conditions for mixture_head and paired_head.",
    "paired_loss": "The paired head adds weight 0.20 to logistic edited-over-genuine ordering inside the exact splice, and weight 0.03 to squared response difference more than one full cell outside it. Compare both ordinary controls.",
    "local_decision": "Select the highest mean across three consecutive cells, with earlier start for equal means. Each arm strictly exceeds its own maximum response across all genuine confidence_cal views, including unedited and background-only controls. No synthetic cases or selection outcomes set the points.",
    "useful": "At least 0.30 time IoU and 0.50 recall against the exact edit support. Retain threshold-free best range, oracle interval mean, matched-pair ordering, false flags, and inconclusive results.",
    "gate": "Future transfer only: at least four additional useful noisy held views on at least three pairs and two families versus the unrefitted frame arm; at most one extra noisy false pair, no lost dry useful pair, and zero held background-only flags.",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def ordered_rows(manifest):
    return sorted(manifest["rows"], key=lambda row: row["id"])


def frozen(run, weights):
    manifest_file = run / "inputs.json"
    manifest = read(manifest_file)
    if manifest["sample_rate"] != RATE or manifest["samples"] != SAMPLES:
        raise ValueError("Expected the fixed four-second PCM scope")
    if manifest["freeze_sha256"] != sha(run / "inputs-freeze.json"):
        raise ValueError("Input construction freeze differs")
    rows = ordered_rows(manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows) or {row["role"] for row in rows} != set(ROLES):
        raise ValueError("Expected distinct IDs and all four source roles")
    if any(row["family"] == "expressive" and row["role"] in ROLES[:2] and row["arm"] == SYNTHETIC for row in rows):
        raise ValueError("Expressive examples are excluded from training and threshold setting")
    if {name: sha(weights / name) for name in ASSETS} != ASSETS:
        raise ValueError("Frozen encoder files differ")
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), root / "app/detectors/audio/upgraded.py", root / "app/third_party/kaldi/fbank.py"]
    return {"schema": SCHEMA, "method": METHOD, "input_manifest_sha256": sha(manifest_file),
            "inputs_freeze_sha256": manifest["freeze_sha256"], "assets": ASSETS,
            "source_views": len(rows), "role_counts": dict(Counter(row["role"] for row in rows)),
            "code_sha256": {str(p.relative_to(root)): sha(p) for p in code},
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers", "pyloudnorm")}}, rows


def pcm_bytes(run, row):
    path = run / row["path"]
    if not path.is_relative_to(run) or not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("PCM path is outside the input folder")
    data = path.read_bytes()
    if len(data) != SAMPLES * 2 or hashlib.sha256(data).hexdigest() != row["pcm_sha256"]:
        raise ValueError("PCM byte count or commitment differs")
    return data


class TemporalCapture:
    """Retain cell means from a single unchanged backbone call."""

    def __init__(self, model):
        self.model = model
        self.record = None
        self.hook = model.backbone.register_forward_hook(self.capture, with_kwargs=True)

    def capture(self, _module, _args, kwargs, output):
        import torch

        states = output.hidden_states
        mask = kwargs["attention_mask"]
        if self.record is not None or len(states) != 25 or mask.shape != (1, 199) or not bool(torch.all(mask == 1)):
            raise ValueError("Expected one exact four-second native frame clock")
        mix = torch.stack(states).mul(self.model.layer_weights.softmax(dim=0)[:, None, None, None]).sum(0)[0]
        cells = (280 + np.arange(199) * 320) // CELL
        if mix.shape != (199, WIDTH) or not bool(torch.isfinite(mix).all()) or set(cells) != set(range(CELLS)):
            raise ValueError("Unexpected temporal representation or frame positions")
        vectors = torch.stack([mix[cells == i].mean(0) for i in range(CELLS)])
        scores = self.model.head(vectors).squeeze(-1)
        self.record = (vectors.cpu().numpy(), scores.cpu().numpy())

    def close(self):
        self.hook.remove()


def encode(run, role, rows, weights):
    import torch

    from app.detectors.audio.upgraded import prepare_voice_window
    from scripts.run_audio_specialists_v15 import load_voice

    output_file = run / f"{role}-features.npz"
    if output_file.exists() or (run / f"{role}-features.json").exists():
        raise ValueError("This role already has an encoder ledger")
    if role in ROLES[2:] and not (run / "fit.json").is_file():
        raise ValueError("Fit and seal operating points before scoring held roles")
    model = load_voice({"weights": str(weights), "assets": ASSETS})
    capture = TemporalCapture(model)
    bound, vectors, local, pooled, cache = [], [], [], [], {}
    started = time.monotonic()
    try:
        for row in (r for r in rows if r["role"] == role):
            data = pcm_bytes(run, row)
            key = row["pcm_sha256"]
            if key not in cache:
                prepared = prepare_voice_window(np.frombuffer(data, "<i2").astype(np.float32) / 32_768)
                if not np.any(prepared[-320:]):
                    raise ValueError("The four-second source lacks an exact unpadded ending")
                input_hash = hashlib.sha256(prepared.astype("<f4", copy=False).tobytes()).hexdigest()
                capture.record = None
                with torch.inference_mode():
                    logit = float(model(torch.from_numpy(prepared)[None]).item())
                if capture.record is None:
                    raise ValueError("No native frames were measured")
                cache[key] = (*capture.record, logit, input_hash)
            hidden, scores, logit, input_hash = cache[key]
            vectors.append(hidden)
            local.append(scores)
            pooled.append(logit)
            bound.append({"id": row["id"], "pcm_sha256": key, "model_input_sha256": input_hash})
            if len(bound) % 20 == 0:
                print(f"{role}: {len(bound)} views encoded", flush=True)
    finally:
        capture.close()
    with output_file.open("xb") as stream:
        np.savez_compressed(stream, vectors=np.stack(vectors), local=np.stack(local), pooled=np.asarray(pooled, dtype=np.float32))
    write(run / f"{role}-features.json", {"freeze_sha256": sha(run / "model-freeze.json"), "rows": bound,
          "arrays_sha256": sha(output_file), "forwards": len(cache), "seconds": round(time.monotonic() - started, 3)})


def load_role(run, role, rows):
    entries = [row for row in rows if row["role"] == role]
    file = run / f"{role}-features.npz"
    binding = read(run / f"{role}-features.json")
    if (binding["freeze_sha256"] != sha(run / "model-freeze.json") or binding["arrays_sha256"] != sha(file)
            or [(r["id"], r["pcm_sha256"]) for r in entries] != [(r["id"], r["pcm_sha256"]) for r in binding["rows"]]):
        raise ValueError("Feature ledger commitment differs")
    with np.load(file, allow_pickle=False) as data:
        vectors, local, pooled = (data[key].astype(np.float64) for key in ("vectors", "local", "pooled"))
    n = len(entries)
    if (vectors.shape, local.shape, pooled.shape) != ((n, CELLS, WIDTH), (n, CELLS), (n,)):
        raise ValueError("Invalid role representation shape")
    if not all(np.isfinite(a).all() for a in (vectors, local, pooled)):
        raise ValueError("Non-finite role representation")
    return entries, vectors, local, pooled


def representation(vectors):
    return np.concatenate((vectors, vectors - vectors.mean(axis=1, keepdims=True)), axis=2)


def overlap_weights(truth):
    if truth is None:
        return np.zeros(CELLS)
    a, b = truth
    if not 0 <= a < b <= SAMPLES:
        raise ValueError("Known voice-edit bounds exceed the scored track")
    starts = np.arange(CELLS) * CELL
    return np.maximum(0, np.minimum(b, starts + CELL) - np.maximum(a, starts)) / CELL


def matched_groups(entries):
    groups = defaultdict(dict)
    for i, row in enumerate(entries):
        if row["arm"] in (*HUMAN, SYNTHETIC):
            group = groups[row["pair_id"], row["condition"]]
            if row["arm"] in group:
                raise ValueError("Duplicate arm in a matched comparison")
            group[row["arm"]] = i
    if any(set(row) != {*HUMAN, SYNTHETIC} for row in groups.values()):
        raise ValueError("Every matched comparison requires both ordinary controls and the edited view")
    return groups


def objective_inputs(entries, scaled, labels):
    pair, off = [], []
    for group in matched_groups(entries).values():
        positive = group[SYNTHETIC]
        weights = labels[positive]
        touched = np.flatnonzero(weights)
        off_bins = np.r_[0:max(0, touched[0] - 1), min(CELLS, touched[-1] + 2):CELLS]
        for arm in HUMAN:
            difference = scaled[positive] - scaled[group[arm]]
            pair.append(np.average(difference, weights=weights, axis=0))
            off.extend(difference[off_bins])
    return np.stack(pair), np.stack(off)


def train(entries, x, arm):
    indices = [i for i, r in enumerate(entries) if r["arm"] in (*HUMAN, SYNTHETIC)
               and (arm != "clean_head" or r["condition"] == "dry")]
    entries = [entries[i] for i in indices]
    x = x[indices]
    center, scale = x.mean(axis=(0, 1)), np.maximum(x.std(axis=(0, 1)), 1e-4)
    scaled = (x - center) / scale
    labels = np.stack([overlap_weights(r["truth"]) for r in entries])
    y = labels.reshape(-1)
    data = scaled.reshape(len(y), -1)
    positive = y / (2 * y.sum())
    mass = positive + (1 - y) / (2 * (1 - y).sum())
    if arm == "paired_head":
        pair, off = objective_inputs(entries, scaled, labels)

    def objective(parameters):
        w, bias = parameters[:-1], parameters[-1]
        logits = data @ w + bias
        errors = mass * expit(logits) - positive
        value = float(mass @ np.logaddexp(0, logits) - positive @ logits + .035 * (w @ w) / 2)
        grad = errors @ data + .035 * w
        if arm == "paired_head":
            relative, untouched = pair @ w, off @ w
            value += .20 * np.logaddexp(0, -relative).mean() + .03 * (untouched @ untouched) / len(off)
            grad += (-.20 / len(pair)) * (expit(-relative) @ pair) + (.06 / len(off)) * (untouched @ off)
        return value, np.r_[grad, errors.sum()]

    fitted = minimize(objective, np.zeros(data.shape[1] + 1), method="L-BFGS-B", jac=True,
                      options={"maxiter": 350, "ftol": 1e-11, "gtol": 1e-7})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Frozen optimization failed for {arm}: {fitted.message}")
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": fitted.x[:-1].tolist(),
            "bias": float(fitted.x[-1]), "fit_views": len(entries), "iterations": fitted.nit, "objective": float(fitted.fun)}


def cell_responses(represented, local, pooled, heads):
    scores = {"incumbent_clip": pooled, "incumbent_frames": local}
    for arm, head in heads.items():
        scores[arm] = ((represented - head["center"]) / head["scale"]) @ head["weights"] + head["bias"]
    return scores


def select(values):
    if np.ndim(values) == 0:
        return float(values), 0, SAMPLES
    windows = np.lib.stride_tricks.sliding_window_view(values, RANGE_CELLS).mean(axis=1)
    start = int(np.argmax(windows))
    return float(windows[start]), start * CELL, (start + RANGE_CELLS) * CELL


def fit(run, rows):
    fit_rows, vectors, _, _ = load_role(run, ROLES[0], rows)
    cal_rows, cal, local, pooled = load_role(run, ROLES[1], rows)
    inputs = representation(vectors)
    heads = {arm: train(fit_rows, inputs, arm) for arm in ARMS[2:]}
    measured = cell_responses(representation(cal), local, pooled, heads)
    points = {}
    for arm, values in measured.items():
        point, witness = max((select(values[i])[0], row["id"]) for i, row in enumerate(cal_rows) if row["label"] == 0)
        points[arm] = {"point": point, "witness": witness}
    write(run / "fit.json", {"freeze_sha256": sha(run / "model-freeze.json"),
          "fit_inputs": {role: sha(run / f"{role}-features.json") for role in ROLES[:2]}, "heads": heads, "points": points})
    print("Three temporal heads and five genuine-only points sealed", flush=True)


def useful_range(start, end, truth):
    if truth is None:
        return False
    a, b = truth
    intersect = max(0, min(end, b) - max(start, a))
    union = end - start + b - a - intersect
    return 10 * intersect >= 3 * union and 2 * intersect >= b - a


def measures(run, role, rows, sealed):
    entries, vectors, local, pooled = load_role(run, role, rows)
    scores = cell_responses(representation(vectors), local, pooled, sealed["heads"])
    output = []
    for i, row in enumerate(entries):
        obs = {}
        for arm in ARMS:
            values = scores[arm][i]
            value, start, end = select(values)
            flag = value > sealed["points"][arm]["point"]
            oracle = overlap_weights(row["truth"])
            oracle_mean = None if not oracle.any() or np.ndim(values) == 0 else float(np.average(values, weights=oracle))
            obs[arm] = {"score": value, "range": [start, end], "flag": flag, "oracle": oracle_mean,
                        "best_range_useful": useful_range(start, end, row["truth"])}
        output.append({"id": row["id"], "pair": row["pair_id"], "role": role, "condition": row["condition"],
                       "family": row["family"], "arm": row["arm"], "label": row["label"], "truth": row["truth"], "checks": obs})
    for group in matched_groups(entries).values():
        positive = output[group[SYNTHETIC]]
        for arm in ARMS:
            record = positive["checks"][arm]
            controls = [group[key] for key in HUMAN]
            record["beats_both"] = all(record["score"] > output[c]["checks"][arm]["score"] for c in controls)
            oracle = record["oracle"]
            record["oracle_beats_both"] = oracle is not None and all(
                oracle > np.average(scores[arm][c], weights=overlap_weights(positive["truth"])) for c in controls)
    return output


def summarize(ledger):
    counts = defaultdict(Counter)
    held = [r for r in ledger if r["role"] == "confirmation"]
    for r in ledger:
        for arm, o in r["checks"].items():
            label = r["label"]
            counts[(r["role"], r["condition"], arm)].update(
                positive=label, controls=1 - label, true_flag=int(bool(label and o["flag"])),
                false_flag=int(bool(not label and o["flag"])), useful=int(bool(o["flag"] and o["best_range_useful"])),
                wrong_location=int(bool(label and o["flag"] and not o["best_range_useful"])),
                inconclusive_positive=int(bool(label and not o["flag"])), inconclusive_control=int(bool(not label and not o["flag"])),
                threshold_free_useful=int(o["best_range_useful"]), beats_both=int(o.get("beats_both", False)),
                oracle_beats_both=int(o.get("oracle_beats_both", False)))
    baseline, candidate = "incumbent_frames", "paired_head"
    def location(row, arm):
        o = row["checks"][arm]
        return o["flag"] and o["best_range_useful"]
    noisy = [r for r in held if r["condition"] != "dry"]
    gains = [r for r in noisy if location(r, candidate) and not location(r, baseline)]
    losses = [r for r in noisy if location(r, baseline) and not location(r, candidate)]
    falses = {arm: {r["pair"] for r in noisy if not r["label"] and r["checks"][arm]["flag"]} for arm in (baseline, candidate)}
    details = {"added_noisy_views": len(gains), "lost_noisy_views": len(losses), "added_noisy_pairs": len({r["pair"] for r in gains}),
               "added_noisy_families": len({r["family"] for r in gains}),
               "added_noisy_false_pairs": len(falses[candidate] - falses[baseline]),
               "lost_dry_useful_pairs": len({r["pair"] for r in held if r["condition"] == "dry" and location(r, baseline) and not location(r, candidate)}),
               "background_flags": sum(r["arm"] == "background_only" and r["checks"][candidate]["flag"] for r in held)}
    qualifies = (details["added_noisy_views"] - details["lost_noisy_views"] >= 4 and details["added_noisy_pairs"] >= 3
                 and details["added_noisy_families"] >= 2 and details["added_noisy_false_pairs"] <= 1
                 and details["lost_dry_useful_pairs"] == details["background_flags"] == 0)
    by_family = []
    for family in sorted({r["family"] for r in noisy if r["label"]}):
        members = [r for r in noisy if r["family"] == family and r["label"]]
        by_family.append({"family": family, "positive": len(members), "useful": {
            arm: sum(location(r, arm) for r in members) for arm in ARMS}})
    return {"counts": [{"role": role, "condition": cond, "arm": arm, **values} for (role, cond, arm), values in sorted(counts.items())],
            "held_noisy_by_family": by_family, "gate": details, "qualifies": qualifies}


def report(run, rows):
    sealed = read(run / "fit.json")
    if (sealed["freeze_sha256"] != sha(run / "model-freeze.json") or
            sealed["fit_inputs"] != {role: sha(run / f"{role}-features.json") for role in ROLES[:2]}):
        raise ValueError("Fitted heads or operating points were not sealed on the fitting roles")
    ledger = [row for role in ROLES[2:] for row in measures(run, role, rows, sealed)]
    write(run / "decisions.json", ledger)
    summary = summarize(ledger)
    write(run / "result.json", {"schema": SCHEMA, "fit_sha256": sha(run / "fit.json"),
          "decisions_sha256": sha(run / "decisions.json"), **summary})
    print({key: summary[key] for key in ("gate", "qualifies")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "encode", "fit", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args()
    freeze, rows = frozen(args.run, args.weights)
    lock = args.run / "model-freeze.json"
    if args.phase == "freeze":
        write(lock, freeze)
        print({"role_counts": freeze["role_counts"], "forwards": 0}, flush=True)
        return
    if read(lock) != freeze:
        raise ValueError("Frozen encoder, inputs, or scoring procedure differs")
    if args.phase == "encode":
        if args.role is None:
            parser.error("encode requires --role")
        encode(args.run, args.role, rows, args.weights)
    elif args.phase == "fit":
        fit(args.run, rows)
    else:
        report(args.run, rows)


if __name__ == "__main__":
    main()
