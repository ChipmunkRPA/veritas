"""Independently replay the fitted temporal decisions from committed CPU features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RATE = 16_000
CELL = 3_200
N = 20
ROLES = ("fusion_fit", "confidence_cal", "selection", "confirmation")
HUMAN = ("ordinary_same_host", "ordinary_other_voice")
HEADS = ("clean_head", "mixture_head", "paired_head")
ARMS = ("incumbent_clip", "incumbent_frames", *HEADS)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def support(truth):
    if truth is None:
        return [0.0] * N
    start, stop = truth
    return [len(range(max(start, i * CELL), min(stop, (i + 1) * CELL))) / CELL for i in range(N)]


def select(values):
    if np.ndim(values) == 0:
        return float(values), [0, N * CELL]
    starts = [(math.fsum(values[i:i + 3]) / 3, -i) for i in range(N - 2)]
    value, neg_start = max(starts)
    return value, [-neg_start * CELL, (3 - neg_start) * CELL]


def useful(bounds, truth):
    if truth is None:
        return False
    start, end = bounds
    a, b = truth
    intersect = len(range(max(start, a), min(end, b)))
    if not intersect:
        return False
    union = len(range(start, end)) + len(range(a, b)) - intersect
    return intersect / union >= .30 and intersect / (b - a) >= .50


def load(run, rows, role, model_sha):
    from app.detectors.audio.upgraded import prepare_voice_window

    file = run / f"{role}-features.npz"
    binding = read(run / f"{role}-features.json")
    actual = [r for r in rows if r["role"] == role]
    require(binding["freeze_sha256"] == model_sha and binding["arrays_sha256"] == sha(file)
            and len(actual) == len(binding["rows"]), "Native role ledger differs")
    for row, recorded in zip(actual, binding["rows"], strict=True):
        data = (run / row["path"]).read_bytes()
        require(len(data) == 2 * 4 * RATE and hashlib.sha256(data).hexdigest() == row["pcm_sha256"], "Original audio bytes differ")
        native = np.asarray(prepare_voice_window(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768), dtype="<f4")
        require((row["id"], row["pcm_sha256"], hashlib.sha256(native.tobytes()).hexdigest()) ==
                (recorded["id"], recorded["pcm_sha256"], recorded["model_input_sha256"]), "Exact model input differs")
    require(binding["forwards"] == len({r["pcm_sha256"] for r in actual}), "Forward count differs")
    with np.load(file, allow_pickle=False) as f:
        hidden, local, clip = (f[key].astype(np.float64) for key in ("vectors", "local", "pooled"))
    represented = np.concatenate((hidden, hidden - hidden.sum(axis=1, keepdims=True) / N), axis=-1)
    return actual, represented, {"incumbent_frames": local, "incumbent_clip": clip}, binding["forwards"]


def groups(rows):
    lookup = defaultdict(dict)
    for i, row in enumerate(rows):
        if row["arm"] in (*HUMAN, "synthetic"):
            lookup[row["pair_id"], row["condition"]][row["arm"]] = i
    require(all(set(row) == {*HUMAN, "synthetic"} for row in lookup.values()), "Unmatched waveform pair")
    return lookup


def stationarity(data, role_rows, heads):
    check = {}
    for arm, parameters in heads.items():
        indices = [i for i, r in enumerate(role_rows) if r["arm"] in (*HUMAN, "synthetic")
                   and (arm != "clean_head" or r["condition"] == "dry")]
        rows = [role_rows[i] for i in indices]
        selected = data[indices]
        w = np.array(parameters["weights"])
        centering = np.mean(selected, axis=(0, 1))
        variance = np.mean((selected - centering) ** 2, axis=(0, 1))
        scale = np.maximum(np.sqrt(variance), .0001)
        require(np.allclose(centering, parameters["center"], atol=1e-12) and np.allclose(scale, parameters["scale"], atol=1e-12)
                and len(rows) == parameters["fit_views"], "Fitting statistics or source rows differ")
        x = (selected - centering) / scale
        y = np.array([support(r["truth"]) for r in rows])
        z = x @ w + parameters["bias"]
        pos = y / (2 * y.sum())
        neg = (1 - y) / (2 * (1 - y).sum())
        weighted_residual = (pos + neg) * np.exp(-np.logaddexp(0, -z)) - pos
        objective = np.sum(pos * np.logaddexp(0, -z) + neg * np.logaddexp(0, z)) + .0175 * np.dot(w, w)
        gradient = np.einsum("ij,ijk->k", weighted_residual, x) + .035 * w
        if arm == "paired_head":
            pairs, distant = [], []
            for group in groups(rows).values():
                i = group["synthetic"]
                present = np.flatnonzero(y[i])
                outside = [k for k in range(N) if k < present[0] - 1 or k > present[-1] + 1]
                for human in HUMAN:
                    contrast = x[i] - x[group[human]]
                    pairs.append(y[i] @ contrast / y[i].sum())
                    distant.extend(contrast[outside])
            delta = np.stack(pairs)
            far = np.stack(distant)
            relative, invariant = delta @ w, far @ w
            objective += .20 * np.logaddexp(0, -relative).mean() + .03 * np.mean(invariant ** 2)
            gradient -= .20 * np.mean(delta * np.exp(-np.logaddexp(0, relative))[:, None], axis=0)
            gradient += .06 * np.mean(far * invariant[:, None], axis=0)
        norm = max(abs(float(weighted_residual.sum())), float(np.max(np.abs(gradient))))
        require(same(objective, parameters["objective"]) and norm < .001, "Fitted convex objective or gradient differs")
        check[arm] = {"fit_views": len(rows), "max_gradient": norm, "objective": float(objective)}
    return check


def replay(run):
    model = read(run / "model-freeze.json")
    manifest = read(run / "inputs.json")
    input_audit = read(run / "inputs-verified.json")
    result = read(run / "result.json")
    seal = read(run / "fit.json")
    rows = sorted(manifest["rows"], key=lambda r: r["id"])
    model_sha = sha(run / "model-freeze.json")
    require(model["input_manifest_sha256"] == input_audit["inputs_sha256"] == sha(run / "inputs.json")
            and input_audit["rows"] == model["source_views"] == len(rows), "The verified source intake differs")
    require(result["fit_sha256"] == sha(run / "fit.json") and result["decisions_sha256"] == sha(run / "decisions.json")
            and seal["freeze_sha256"] == model_sha and seal["fit_inputs"] == {r: sha(run / f"{r}-features.json") for r in ROLES[:2]},
            "Fitting or decision commitment differs")
    saved = {r["id"]: r for r in read(run / "decisions.json")}
    by_cell, forwards = defaultdict(Counter), 0
    for role in ROLES:
        cases, data, scores, count = load(run, rows, role, model_sha)
        forwards += count
        if role == ROLES[0]:
            fitted = stationarity(data, cases, seal["heads"])
            continue
        for arm, head in seal["heads"].items():
            scores[arm] = np.einsum("ijk,k->ij", (data - head["center"]) / head["scale"], head["weights"]) + head["bias"]
        if role == ROLES[1]:
            for arm in ARMS:
                value, witness = max((select(scores[arm][i])[0], row["id"]) for i, row in enumerate(cases) if row["label"] == 0)
                require(same(value, seal["points"][arm]["point"]) and witness == seal["points"][arm]["witness"], "Point or control witness differs")
            continue
        group = groups(cases)
        for i, row in enumerate(cases):
            expected = saved[row["id"]]
            require((row["role"], row["condition"], row["truth"], row["arm"], row["pair_id"], row["family"], row["label"]) ==
                    tuple(expected[k] for k in ("role", "condition", "truth", "arm", "pair", "family", "label")), "Saved case differs")
            for arm in ARMS:
                values = scores[arm][i]
                saved_arm = expected["checks"][arm]
                value, bounds = select(values)
                flag = value > seal["points"][arm]["point"]
                located = useful(bounds, row["truth"])
                require(same(value, saved_arm["score"]) and bounds == saved_arm["range"] and flag == saved_arm["flag"]
                        and located == saved_arm["best_range_useful"], "Saved response, location, or decision differs")
                paired_order, oracle_order = False, False
                if row["label"]:
                    ordinary = [group[row["pair_id"], row["condition"]][human] for human in HUMAN]
                    paired_order = all(value > select(scores[arm][j])[0] for j in ordinary)
                    if arm != "incumbent_clip":
                        fractions = support(row["truth"])
                        oracle = math.fsum(w * x for w, x in zip(fractions, values, strict=True)) / sum(fractions)
                        require(same(oracle, saved_arm["oracle"]), "Oracle interval differs")
                        oracle_order = all(oracle > math.fsum(w * x for w, x in zip(fractions, scores[arm][j], strict=True)) / sum(fractions)
                                           for j in ordinary)
                    require(paired_order == saved_arm["beats_both"] and oracle_order == saved_arm["oracle_beats_both"], "Pair ordering differs")
                label = row["label"]
                counts = by_cell[role, row["condition"], arm]
                counts.update(positive=label, controls=1-label, true_flag=int(label and flag), false_flag=int(not label and flag),
                              useful=int(flag and located), wrong_location=int(label and flag and not located),
                              inconclusive_positive=int(label and not flag), inconclusive_control=int(not label and not flag),
                              threshold_free_useful=int(located), beats_both=int(paired_order), oracle_beats_both=int(oracle_order))
    recorded = {(r["role"], r["condition"], r["arm"]): {k: v for k, v in r.items() if k not in ("role", "condition", "arm")}
                for r in result["counts"]}
    require(by_cell == recorded and set(saved) == {r["id"] for r in rows if r["role"] in ROLES[2:]}, "Aggregate outcome cells differ")
    gate_replay(list(saved.values()), result)
    return {"bound_pcm_views": len(rows), "native_forwards": forwards, "source_linked_decisions": len(saved) * len(ARMS),
            "aggregate_cells": len(by_cell), "optimizer_checks": fitted, "report_sha256": sha(run / "result.json"),
            "reviewer_sha256": sha(Path(__file__))}


def gate_replay(rows, result):
    held = [r for r in rows if r["role"] == "confirmation"]
    noisy = [r for r in held if r["condition"] != "dry"]
    reference, candidate = "incumbent_frames", "paired_head"
    located, false_groups = defaultdict(set), defaultdict(set)
    for r in held:
        for arm in ARMS:
            values = r["checks"][arm]
            if values["flag"] and useful(values["range"], r["truth"]):
                located[arm].add(r["id"])
            if values["flag"] and not r["label"] and r["condition"] != "dry":
                false_groups[arm].add(r["pair"])
    additions = located[candidate] - located[reference]
    losses = located[reference] - located[candidate]
    added = [r for r in noisy if r["id"] in additions]
    gate = {"added_noisy_views": len(added), "lost_noisy_views": sum(r["id"] in losses for r in noisy),
            "added_noisy_pairs": len({r["pair"] for r in added}), "added_noisy_families": len({r["family"] for r in added}),
            "added_noisy_false_pairs": len(false_groups[candidate] - false_groups[reference]),
            "lost_dry_useful_pairs": len({r["pair"] for r in held if r["condition"] == "dry" and r["id"] in losses}),
            "background_flags": sum(r["arm"] == "background_only" and r["checks"][candidate]["flag"] for r in held)}
    qualifies = all((gate["added_noisy_views"] - gate["lost_noisy_views"] >= 4, gate["added_noisy_pairs"] >= 3,
                     gate["added_noisy_families"] >= 2, gate["added_noisy_false_pairs"] <= 1,
                     gate["lost_dry_useful_pairs"] == 0, gate["background_flags"] == 0))
    families = []
    for family in sorted({r["family"] for r in noisy if r["label"]}):
        ids = {r["id"] for r in noisy if r["label"] and r["family"] == family}
        families.append({"family": family, "positive": len(ids), "useful": {arm: len(ids & located[arm]) for arm in ARMS}})
    require((gate, qualifies, families) == (result["gate"], result["qualifies"], result["held_noisy_by_family"]),
            "Held advancement or family counts differ")


def meeting_replay(run, parent):
    from app.detectors.audio.upgraded import prepare_voice_window
    from scripts.audio_room_reference_transfer import parent_inputs

    freeze = read(run / "meeting-replay-freeze.json")
    result = read(run / "meeting-replay.json")
    require(result["freeze_sha256"] == sha(run / "meeting-replay-freeze.json")
            and freeze["fit_sha256"] == sha(run / "fit.json")
            and freeze["parent_roster_sha256"] == sha(parent / "roster.json"), "Meeting replay commitment differs")
    sources = {row["id"]: (row, audio) for pair in parent_inputs(parent) for row, audio, _ in pair.values()}
    by_cell = defaultdict(Counter)
    for lock, scored in zip(freeze["inputs"], result["decisions"], strict=True):
        identity, digest, windows = lock
        row, audio = sources.pop(identity)
        require((identity, row["source_pcm_sha256"]) == (scored["id"], digest), "Parent waveform witness differs")
        require(all(row[key] == scored[key] for key in ("panel", "condition", "truth", "kind")), "Meeting source labels differ")
        for offset, bound, prepared in zip((0, 32_000), windows, scored["model_inputs"], strict=True):
            selected = audio[2*offset:2*(offset + 64_000)]
            require(hashlib.sha256(selected).hexdigest() == bound, "Meeting input window differs")
            processed = prepare_voice_window(np.frombuffer(selected, dtype="<i2").astype(np.float32) / 32768)
            require(hashlib.sha256(processed.astype("<f4", copy=False).tobytes()).hexdigest() == prepared, "Meeting model input differs")
        for arm, observation in scored["checks"].items():
            a, b = observation["range"]
            scope = (a, b) in {(0, 64_000), (32_000, 96_000)} if arm == "incumbent_clip" else a % CELL == 0 and b-a == 3*CELL
            require(0 <= a < b <= 96_000 and scope, "Meeting range is off the source grid")
            located = useful((a, b), row["truth"])
            require(located == observation["location"], "Meeting localization decision differs")
            positive, flag = row["kind"] == "partial_ai", observation["flag"]
            by_cell[row["panel"], row["condition"], arm].update(positive=int(positive), genuine=int(not positive),
                true_flag=int(positive and flag), false_flag=int(not positive and flag), useful=int(flag and located),
                threshold_free_location=int(located))
    expected = [{"role": r, "condition": c, "arm": a, **values} for (r, c, a), values in sorted(by_cell.items())]
    require(not sources and expected == result["cells"] and result["forwards"] == 96, "Meeting aggregate cells differ")
    return {"prepared_input_hashes": 96, "aggregate_cells": len(expected), "report_sha256": sha(run / "meeting-replay.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--meeting-parent", type=Path)
    args = parser.parse_args()
    audit = replay(args.run)
    if args.meeting_parent:
        audit["meeting_replay"] = meeting_replay(args.run, args.meeting_parent)
    write(args.output, audit)
    print(audit, flush=True)


if __name__ == "__main__":
    main()
