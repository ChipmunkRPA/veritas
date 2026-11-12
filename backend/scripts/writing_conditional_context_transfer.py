"""Test exact sentence likelihood under nearby, distant and empty contexts."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np

from scripts.writing_token_curvature import (
    file_hash,
    read,
    target_token_indices,
    text_hash,
    verified_model,
    write,
)

SCHEMA = "chipmunk.writing-conditional-context-transfer/1"
RUNTIME = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.15.1"}
CONDITIONS = ("near", "far", "empty")
SOURCE_FILES = ("inputs.json", "outcomes.json", "primary.json")
MODELS = ("absolute", "conditional", "primary_rank", "combined")
POINT_GROUP_BUDGET = 2
PLAN = {
    "scope": "Read the exact interior sentences of the retained plain-text article cases. Use twelve older works for fitting and the ten source-disjoint previously evaluated articles only for held readout.",
    "near": "Preceding sentence in native order; require a following sentence and 12–65 focus words. Require 8–80 words in both context sentences.",
    "far": "Same-article sentence separated by at least three sentence positions, closest in word count to the actual preceding sentence. Resolve ties by source order.",
    "empty": "Same prompt scaffold with no context. Exact target tokens must be identical in every arm and have complete native source coverage.",
    "fit": "Fixed StandardScaler and L2 logistic regression C=1, balanced labels. Train on eligible generated focus sentences and eligible original human sentences. No held labels or scores enter fitting or cutoffs.",
    "features": ["empty mean log probability", "near minus empty", "near minus far", "existing primary context log odds"],
    "point": "Require a strict score above the third-highest original article maximum in selection; report held equal false-group counts separately.",
    "advance": "A proposed addition must recover at least three held edits in different works, across both generators, at no more than one additional false human group versus the native incumbent. Retain existing useful locations.",
}


def source_cases(parent):
    incoming = read(parent / "inputs.json")
    outcome = {(row["group"], row["kind"]): row for row in read(parent / "outcomes.json") if row["format"] == "txt"}
    primary = read(parent / "primary.json")["responses"]
    cases, prompts = [], {}
    for row in incoming["rows"]:
        if row["format"] != "txt":
            continue
        path = parent / "fixtures" / row["file"]
        body = path.read_bytes().decode()
        assert file_hash(path) == row["file_sha256"] and text_hash(body) == row["text_sha256"]
        spans = [window["focus"] for window in row["windows"]]
        assert spans == sorted(spans) and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        sentences = [body[a:b] for a, b in spans]
        counts = [len(value.split()) for value in sentences]
        eligible = []
        for i, window in enumerate(row["windows"]):
            native = " ".join(body[a:b] for a, b in window["spans"])
            assert text_hash(native) == window["sha256"] and primary[window["sha256"]]["coverage"]["token_windows"] == 1
            if not 0 < i < len(spans) - 1 or not 12 <= counts[i] <= 65 or not 8 <= counts[i-1] <= 80:
                continue
            far = [j for j, count in enumerate(counts) if abs(j-i) >= 3 and 8 <= count <= 80]
            if not far:
                continue
            j = min(far, key=lambda k: (abs(counts[k]-counts[i-1]), k))
            conditions = {}
            for kind, context in zip(CONDITIONS, (sentences[i-1], sentences[j], ""), strict=True):
                prefix = "Context:\n" + context + "\n\nSentence:\n"
                digest = text_hash(prefix + sentences[i])
                prompts[digest] = [prefix, sentences[i]]
                conditions[kind] = digest
            eligible.append({"focus": spans[i], "near": spans[i-1], "far": spans[j], "conditions": conditions,
                             "primary": window["sha256"], "score": primary[window["sha256"]]["score"]})
        previous = outcome[row["group"], row["kind"]]
        assert previous["file_sha256"] == row["file_sha256"] and previous["truth"] == row["truth"]
        cases.append({key: row[key] for key in ("file", "file_sha256", "group", "scope", "kind", "truth")}
                     | {"length": len(body), "eligible": eligible, "incumbent": previous["arms"]["current"]})
    groups = {scope: {case["group"] for case in cases if case["scope"] == scope}
              for scope in ("fresh", "exposed_rewrite_transfer")}
    assert len(cases) == 55 and len(groups["fresh"]) == 10 and len(groups["exposed_rewrite_transfer"]) == 12
    assert not groups["fresh"] & groups["exposed_rewrite_transfer"]
    return cases, prompts


def freeze(parent, run, snapshot):
    verified_model(snapshot)
    cases, prompts = source_cases(parent)
    run.mkdir(mode=0o700)
    write(run / "cases.json", cases)
    write(run / "prompts.json", prompts)
    write(run / "freeze.json", {"schema": SCHEMA, "plan": PLAN, "runner": file_hash(Path(__file__)),
          "parent": {name: file_hash(parent / name) for name in SOURCE_FILES},
          "runtime": RUNTIME, "model_revision": snapshot.name,
          "cases_sha256": file_hash(run / "cases.json"), "prompts_sha256": file_hash(run / "prompts.json")})
    print(f"Froze {len(cases)} articles and {len(prompts)} exact prompts", flush=True)


def checked(parent, run):
    lock = read(run / "freeze.json")
    assert lock["schema"] == SCHEMA and lock["plan"] == PLAN and lock["runner"] == file_hash(Path(__file__))
    assert lock["parent"] == {name: file_hash(parent / name) for name in SOURCE_FILES}
    assert lock["cases_sha256"] == file_hash(run / "cases.json")
    assert lock["prompts_sha256"] == file_hash(run / "prompts.json")
    return read(run / "cases.json"), read(run / "prompts.json")


def score(parent, run, snapshot, limit):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    assert all(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
    assert all(importlib.metadata.version(name) == version for name, version in RUNTIME.items())
    _, prompts = checked(parent, run)
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024 * 1024)
    commitment = file_hash(run / "freeze.json")
    checkpoint = run / "readings-checkpoint.json"
    saved, elapsed = {}, 0
    if checkpoint.is_file():
        previous = read(checkpoint)
        assert previous["freeze_sha256"] == commitment
        saved, elapsed = previous["measurements"], previous["seconds"]
    entries = list(prompts.items())
    assert list(saved) == sorted(key for key, _ in entries[:len(saved)])
    todo = entries[len(saved):len(saved)+limit]
    assert todo, "The frozen score stage already measured all prompts."
    started = monotonic()
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    for digest, (prefix, target) in todo:
        full = prefix + target
        assert digest == text_hash(full)
        batch = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        positions = target_token_indices(full, batch["offset_mapping"], (len(prefix), len(full)))
        tokens = batch["input_ids"]
        assert len(tokens) <= 512
        observed = [tokens[j] for j in positions]
        pred = model(mx.array([tokens]))[0][mx.array([j-1 for j in positions])].astype(mx.float32)
        indices = mx.arange(len(observed))
        loss = pred[indices, mx.array(observed)] - mx.logsumexp(pred, axis=-1)
        values = loss.tolist()
        assert len(values) == len(observed) and all(math.isfinite(value) and value <= 0 for value in values)
        saved[digest] = {"token_ids": observed, "indices": positions, "logp": values}
    packet = {"freeze_sha256": commitment, "measurements": saved,
              "forwards": len(saved), "seconds": round(elapsed + monotonic() - started, 3)}
    temporary = run / "readings-checkpoint.tmp"
    temporary.unlink(missing_ok=True)
    write(temporary, packet)
    temporary.replace(checkpoint)
    if len(saved) == len(prompts):
        write(run / "readings.json", packet)
    print(f"Measured {len(saved)}/{len(prompts)} prompts; peak MLX memory {mx.get_peak_memory()} bytes", flush=True)


def focus_features(window, readings):
    near, far, empty = (readings[window["conditions"][kind]] for kind in CONDITIONS)
    if not near["token_ids"] == far["token_ids"] == empty["token_ids"]:
        return None
    avg = [math.fsum(row["logp"]) / len(row["logp"]) for row in (near, far, empty)]
    score = min(max(window["score"], 1e-7), 1-1e-7)
    return [avg[2], avg[0]-avg[2], avg[0]-avg[1], math.log(score / (1-score))]


def useful(body, focus, truth):
    if truth is None:
        return False
    a = {i for i in range(*focus) if not body[i].isspace()}
    b = {i for i in range(*truth) if not body[i].isspace()}
    return len(a & b) >= .50 * len(b) and len(a & b) >= .30 * len(a | b)


def evaluate(parent, cases, readings):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows, fit_features, fit_labels = [], [], []
    for case in cases:
        body = (parent / "fixtures" / case["file"]).read_bytes().decode()
        fitting = case["scope"] != "fresh"
        scored = []
        for window in case["eligible"]:
            features = focus_features(window, readings)
            if features is None:
                continue
            exact = useful(body, window["focus"], case["truth"])
            scored.append((window["focus"], features, exact))
            if fitting and (exact or case["kind"] == "human"):
                fit_features.append(features)
                fit_labels.append(int(exact))
        rows.append({"case": case, "scored": scored})
    fits = {}
    for name, columns in zip(MODELS, ([0], [0, 1, 2], [3], [0, 1, 2, 3]), strict=True):
        fitted = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", random_state=0))
        fitted.fit(np.asarray(fit_features)[:, columns], fit_labels)
        fits[name] = fitted, columns
    for row in rows:
        predictions = {}
        for name, (fitted, columns) in fits.items():
            if not row["scored"]:
                predictions[name] = None
                continue
            values = fitted.decision_function(np.asarray([data for _, data, _ in row["scored"]])[:, columns])
            index = int(np.argmax(values))
            focus, _, exact = row["scored"][index]
            predictions[name] = {"score": float(values[index]), "focus": focus, "useful": exact}
        row["predictions"] = predictions
    cutoffs = {name: cutoff(rows, name, "exposed_rewrite_transfer", POINT_GROUP_BUDGET) for name in MODELS}
    return rows, cutoffs, Counter(fit_labels)


def cutoff(rows, arm, scope, false_groups):
    genuine = [row["predictions"][arm]["score"] for row in rows if row["case"]["scope"] == scope
               and row["case"]["kind"] == "human" and row["predictions"][arm] is not None]
    return sorted(genuine, reverse=True)[false_groups]


def summarize(rows, arm, point, scope, *, union=False):
    counts = Counter()
    for row in rows:
        case = row["case"]
        if case["scope"] != scope:
            continue
        prediction = row["predictions"][arm] if arm else None
        old = case["incumbent"]
        active = prediction is not None and prediction["score"] > point
        use_old = union or arm is None
        flagged = active or (use_old and old["flag"])
        located = bool(active and prediction["useful"]) or (use_old and old["useful"])
        if case["kind"] == "human":
            counts["false_groups"] += int(flagged)
            counts["genuine_groups"] += 1
        else:
            counts["useful"] += int(located)
            counts["edits"] += 1
            counts[f"{case['kind']}_useful"] += int(located)
            counts["wrong_only"] += int(flagged and not located)
        counts["assessed"] += int(arm is None or prediction is not None)
    return counts


def new_locations(rows, arm, point):
    counts = Counter()
    works = set()
    for row in rows:
        case, prediction = row["case"], row["predictions"][arm]
        if (case["scope"] == "fresh" and case["kind"] != "human" and not case["incumbent"]["useful"]
                and prediction and prediction["score"] > point and prediction["useful"]):
            works.add(case["group"])
            counts[case["kind"]] += 1
    return {"works": len(works), "qwen": counts["qwen"], "smol2": counts["smol2"]}


def report(parent, run):
    cases, prompts = checked(parent, run)
    packet = read(run / "readings.json")
    readings = packet["measurements"]
    assert packet["freeze_sha256"] == file_hash(run / "freeze.json") and set(readings) == set(prompts)
    rows, points, training = evaluate(parent, cases, readings)
    scope = "fresh"
    result = {"schema": SCHEMA, "design": PLAN, "freeze_sha256": file_hash(run / "freeze.json"),
              "forwards": packet["forwards"], "seconds": packet["seconds"],
              "training_eligible": {str(k): v for k, v in training.items()}, "held_cutoffs": points,
              "prior": summarize(rows, None, 0, scope),
              "fixed": {name: summarize(rows, name, point, scope) for name, point in points.items()},
              "fixed_union": {name: summarize(rows, name, point, scope, union=True) for name, point in points.items()},
              "new_locations": {name: new_locations(rows, name, point) for name, point in points.items()},
              "equal_false_groups": {str(budget): {name: summarize(rows, name, cutoff(rows, name, scope, budget), scope)
                                                   for name in MODELS} for budget in (0, 1, 2, 3)},
              "binding_counts": {"articles": len(cases), "exact_prompts": len(prompts),
                                 "target_tokens": sum(len(value["token_ids"]) for value in readings.values()),
                                 "eligible_foci": sum(len(row["case"]["eligible"]) for row in rows),
                                 "comparable_foci": sum(len(row["scored"]) for row in rows)}}
    result["stage"] = "advance" if any(
        result["new_locations"][name]["works"] >= 3
        and result["fixed_union"][name]["false_groups"] <= result["prior"]["false_groups"] + 1
        and all(result["new_locations"][name][kind] > 0 for kind in ("qwen", "smol2"))
        for name in ("conditional", "combined")
    ) else "research"
    write(run / "result.json", result)
    print(result, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score", "report"))
    cli.add_argument("parent", type=Path)
    cli.add_argument("run", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--limit", type=int, default=128, help="Maximum new prompts in this model process")
    args = cli.parse_args()
    if args.phase == "report":
        report(args.parent, args.run)
    elif args.phase == "freeze":
        freeze(args.parent, args.run, args.snapshot)
    else:
        if args.limit < 1:
            cli.error("--limit must be positive")
        score(args.parent, args.run, args.snapshot, args.limit)


if __name__ == "__main__":
    main()
