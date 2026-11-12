"""Recheck native text, likelihood reductions and held ranking with a separate optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

FEATURES = {"absolute": (0,), "conditional": (0, 1, 2), "primary_rank": (3,), "combined": (0, 1, 2, 3)}


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_records(parent, run):
    seal = read(run / "freeze.json")
    assert all(sha((parent / name).read_bytes()) == commitment for name, commitment in seal["parent"].items())
    assert sha((run / "cases.json").read_bytes()) == seal["cases_sha256"]
    assert sha((run / "prompts.json").read_bytes()) == seal["prompts_sha256"]
    prompts = read(run / "prompts.json")
    readings = read(run / "readings.json")
    assert readings["freeze_sha256"] == sha((run / "freeze.json").read_bytes())
    measures = readings["measurements"]
    assert measures.keys() == prompts.keys()
    recorded = {row["file"]: row for row in read(parent / "inputs.json")["rows"] if row["format"] == "txt"}
    responses = read(parent / "primary.json")["responses"]
    cases = read(run / "cases.json")
    seen = set()
    bound = 0
    for case in cases:
        source = (parent / "fixtures" / case["file"]).read_bytes()
        baseline = recorded[case["file"]]
        assert sha(source) == case["file_sha256"] == baseline["file_sha256"]
        assert all(case[key] == baseline[key] for key in ("kind", "group", "scope", "truth"))
        text = source.decode()
        windows = {tuple(row["focus"]): row for row in baseline["windows"]}
        rows = []
        for row in case["eligible"]:
            a, b = row["focus"]
            assert row["primary"] == windows[a, b]["sha256"] and row["score"] == responses[row["primary"]]["score"]
            target = text[a:b]
            values, tokens = [], []
            for mode, source_span in (("near", row["near"]), ("far", row["far"]), ("empty", None)):
                context = text[slice(*source_span)] if source_span else ""
                prompt = f"Context:\n{context}\n\nSentence:\n"
                signature = sha((prompt + target).encode())
                assert signature == row["conditions"][mode] and prompts[signature] == [prompt, target]
                saved = measures[signature]
                logits = saved["logp"]
                indices = saved["indices"]
                assert len(logits) == len(saved["token_ids"]) == len(indices) > 0
                assert min(indices) > 0 and indices == sorted(set(indices)) and all(math.isfinite(x) and x <= 0 for x in logits)
                values.append(sum(logits) / len(logits))
                tokens.append(saved["token_ids"])
                seen.add(signature)
                bound += 1
            if len({tuple(value) for value in tokens}) != 1:
                continue
            likely = min(1-1e-7, max(1e-7, row["score"]))
            features = (values[2], values[0]-values[2], values[0]-values[1], math.log(likely)-math.log1p(-likely))
            edit = case["truth"]
            intersects = sum(not char.isspace() for char in text[max(a, edit[0]):min(b, edit[1])]) if edit else 0
            focus_count = sum(not char.isspace() for char in target)
            target_count = sum(not char.isspace() for char in text[slice(*edit)]) if edit else 0
            hit = bool(edit and 2 * intersects >= target_count and 10 * intersects >= 3 * (focus_count+target_count-intersects))
            rows.append((features, hit))
        case["measurements"] = rows
    assert seen == measures.keys() and len(cases) == len(recorded)
    return cases, bound


def fit(train, labels):
    average = train.mean(axis=0)
    spread = train.std(axis=0)
    spread[spread == 0] = 1
    x = np.column_stack(((train-average)/spread, np.ones(len(train))))
    weights = np.array([len(labels)/(2 * np.sum(labels == value)) for value in labels])

    def objective(beta):
        value = x @ beta
        penalty = beta.copy()
        penalty[-1] = 0
        return (np.dot(weights, np.logaddexp(0, value)-labels*value) + .5*np.dot(penalty, penalty),
                x.T @ (weights * (expit(value)-labels)) + penalty)

    answer = minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-8})
    assert np.max(np.abs(answer.jac)) < 1e-5
    return average, spread, answer.x


def replay(cases, public):
    training = [(row, hit) for case in cases if case["scope"] == "exposed_rewrite_transfer"
                for row, hit in case["measurements"] if case["kind"] == "human" or hit]
    all_rows, labels = zip(*training, strict=True)
    x, y = np.array(all_rows), np.array(labels, dtype=int)
    assert public["training_eligible"] == {"0": int((y == 0).sum()), "1": int(y.sum())}
    counts = 0
    for name, columns in FEATURES.items():
        center, scale, beta = fit(x[:, columns], y)
        partitions = {"fresh": [], "exposed_rewrite_transfer": []}
        for case in cases:
            observed = case["measurements"]
            if not observed:
                partitions[case["scope"]].append((case, float("-inf"), False))
                continue
            features, hits = zip(*observed, strict=True)
            rank = ((np.array(features)[:, columns]-center)/scale) @ beta[:-1] + beta[-1]
            ix = int(rank.argmax())
            partitions[case["scope"]].append((case, float(rank[ix]), hits[ix]))
        points = sorted((score for row, score, _ in partitions["exposed_rewrite_transfer"] if row["kind"] == "human"), reverse=True)
        assert abs(points[2]-public["held_cutoffs"][name]) < .015
        held = partitions["fresh"]
        human = sorted((score for row, score, _ in held if row["kind"] == "human"), reverse=True)
        comparisons = [(points[2], public["fixed"][name], False), (points[2], public["fixed_union"][name], True)]
        comparisons.extend((human[int(budget)], summary[name], False) for budget, summary in public["equal_false_groups"].items())
        for point, reported, union in comparisons:
            hits = false = wrong = 0
            for row, score, exact in held:
                original = row["incumbent"]
                flagged = score > point or (union and original["flag"])
                useful = score > point and exact or (union and original["useful"])
                if row["kind"] == "human":
                    false += int(flagged)
                else:
                    hits += int(useful)
                    wrong += int(flagged and not useful)
            assert (hits, false, wrong) == (reported["useful"], reported["false_groups"], reported["wrong_only"])
            counts += 3
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cases, bindings = extract_records(args.parent, args.run)
    public = read(args.run / "result.json")
    verified = replay(cases, public)
    result = {"articles": len(cases), "context_hash_bindings": bindings,
              "separate_fit_outcome_counts_verified": verified}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
