"""Reproduce the multi-paragraph writing comparison from its portable ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("primary_focus", "independent_focus", "omit_center", "combined")
ORDINARY = {"human", "splice"}
GENERATED = {"qwen", "smol2"}


def read(path):
    return json.loads(path.read_bytes())


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def logit(p):
    assert 0 <= p <= 1
    limited = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(limited / (1 - limited))


def hits(extent):
    same, total, target = extent
    assert all(type(x) is int for x in extent) and 0 <= same <= target <= total
    return target > 0 and same / total >= .3 and same / target >= .5


def overlap(body, truth, spans):
    affected = set()
    for start, end in spans:
        affected |= {i for i in range(start, end) if not body[i].isspace()}
    target = set() if truth is None else {i for i in range(*truth) if not body[i].isspace()}
    return [len(target & affected), len(target | affected), len(target)]


def compute_ranks(public):
    model = public["model"]
    assert len(model["means"]) == len(model["scales"]) == len(model["coefficients"]) == 4
    groups = {partition: {c["group"] for c in public["cases"] if c["partition"] == partition}
              for partition in ("fit", "held")}
    assert len(groups["fit"]) == 12 and len(groups["held"]) == 10 and not groups["fit"] & groups["held"]
    training = [(row["features"], int(case["kind"] in GENERATED))
                for case in public["cases"] if case["partition"] == "fit"
                for row in case["windows"] if case["kind"] == "human" or row["focus"] == case["truth"]]
    assert model["rows"] == len(training) and model["positives"] == sum(label for _, label in training)
    weights = [len(training) / (2 * sum(label == side for _, label in training)) for side in (0, 1)]
    means = [math.fsum(values) / len(training) for values in zip(*(row for row, _ in training), strict=True)]
    scales = [(math.fsum((row[i] - means[i])**2 for row, _ in training) / len(training))**.5 or 1 for i in range(4)]
    for old, new in zip(model["means"] + model["scales"], means + scales, strict=True):
        assert math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-12)

    def fit_response(row):
        standardized = [(value - center) / spread for value, center, spread in zip(row, means, scales, strict=True)]
        raw = model["intercept"] + sum(x * w for x, w in zip(standardized, model["coefficients"], strict=True))
        return raw, standardized

    gradient = [*model["coefficients"], 0.]
    for row, label in training:
        estimate, x = fit_response(row)
        error = ((1 / (1 + math.exp(-estimate))) - label) * weights[label]
        gradient = [value + error * item for value, item in zip(gradient, [*x, 1], strict=True)]
    # At the minimum of the convex regularized loss, the derivative vanishes.
    assert max(map(abs, gradient)) < .03

    ledger = []
    identities = set()
    for case in public["cases"]:
        identity = case["partition"], case["group"], case["kind"]
        assert identity not in identities and case["kind"] in ORDINARY | GENERATED
        identities.add(identity)
        candidates = case["windows"]
        assert len({tuple(row["focus"]) for row in candidates}) == len(candidates)
        score_rows = [(*row["features"][:3], fit_response(row["features"])[0]) for row in candidates]
        record = {"current": (True, bool(case["current"]), any(map(hits, case["current"])))}
        for i, name in enumerate(ARMS):
            values = [row[i] for row in score_rows]
            if not values or values.count(max(values)) != 1:
                record[name] = (bool(candidates), False, False)
                continue
            maximum = max(values)
            pick = candidates[values.index(maximum)]
            # Use the saved standardized moments for comparisons at exact ties.
            features = pick["features"]
            if name == "combined":
                maximum = model["intercept"] + math.fsum(
                    w * ((v - m) / s) for w, v, m, s in zip(model["coefficients"], features, model["means"], model["scales"], strict=True))
            record[name] = (True, maximum, hits(pick["overlap"]))
        ledger.append(record)

    for name in ARMS:
        maxima = {}
        for case, record in zip(public["cases"], ledger, strict=True):
            if case["partition"] == "fit" and case["kind"] in ORDINARY:
                _, value, _ = record[name]
                if value is not False:
                    maxima[case["group"]] = max(value, maxima.get(case["group"], -math.inf))
        assert sorted(maxima.values(), reverse=True)[2] == public["points"][name]
    return ledger


def public_review(public):
    assert public["schema"] == "chipmunk.writing-multiblock-independent-transfer/1"
    ledger = compute_ranks(public)
    for partition in ("fit", "held"):
        rows = [(c, r) for c, r in zip(public["cases"], ledger, strict=True) if c["partition"] == partition]
        baseline = {c["group"] for c, r in rows if c["kind"] in ORDINARY and r["current"][1]}
        for arm in ("current", *ARMS):
            visible = []
            for case, record in rows:
                measured, rank, hit = record[arm]
                marked = bool(rank) if arm == "current" else rank is not False and rank > public["points"][arm]
                visible.append((case, measured, marked, marked and hit, record["current"][2]))
            ordinary = {c["group"] for c, _, mark, _, _ in visible if c["kind"] in ORDINARY and mark}
            added = [(c["group"], c["kind"]) for c, _, _, hit, prior in visible if c["kind"] in GENERATED and hit and not prior]
            summary = public["outcomes"][f"{partition}/{arm}"]
            assert (len(ordinary), len(ordinary - baseline), len({g for g, _ in added}), sorted({k for _, k in added})) == (
                summary["ordinary_groups_flagged"], summary["added_ordinary_groups"], summary["added_useful_groups"], summary["added_useful_families"])
            for kind in ORDINARY | GENERATED:
                expected = summary[kind]
                subset = [row for row in visible if row[0]["kind"] == kind]
                assert expected == {
                    "files": len(subset), "flags": sum(mark for _, _, mark, _, _ in subset),
                    "useful": sum(hit for _, _, _, hit, _ in subset),
                    "unassessed": sum(not measured for _, measured, _, _, _ in subset),
                    "eligible_truth": sum(any(hits(w["overlap"]) for w in c["windows"]) for c, *_ in subset),
                    "wrong_only": sum(mark and not hit for _, _, mark, hit, _ in subset) if kind in GENERATED else 0,
                }
    selected = public["outcomes"]["held/combined"]
    assert public["advanced"] == (selected["added_useful_groups"] >= 2
                                   and set(selected["added_useful_families"]) == GENERATED
                                   and selected["added_ordinary_groups"] <= 1
                                   and sum(selected[k]["eligible_truth"] for k in GENERATED) >= 12)
    print(f"Replayed model fit, four operating points, and outcomes for {len(ledger)} files.")


def private_review(public, parent, run):
    seal = read(run / "freeze.json")
    assert sha((run / "freeze.json").read_bytes()) == public["freeze_sha256"]
    for root, hashes in ((parent, seal["parent"]), (run, public["private_commitments"])):
        assert all(sha((root / name).read_bytes()) == commitment for name, commitment in hashes.items())
    parent_manifest = {(c["group"], c["kind"]): c for c in read(parent / "inputs.json")["rows"] if c["format"] == "txt"}
    source_proofs = {c["group"]: c for c in read(parent / "sources.json")["works"]}
    primary = read(parent / "primary.json")["responses"] | read(run / "primary.json")["responses"]
    independent = read(run / "independent.json")["responses"]
    sources = read(run / "cases.json")
    for native, shown in zip(sources, public["cases"], strict=True):
        group, kind, body = native["group"], native["kind"], native["text"]
        source = parent_manifest[group, "human" if kind == "splice" else kind]
        original = (parent / "fixtures" / source["file"]).read_bytes()
        assert sha(original) == source["file_sha256"] == native["source_sha256"] == shown["source_sha256"]
        assert sha(body) == native["text_sha256"] == shown["text_sha256"]
        assert shown["group"] == sha(group)[:16] and native["truth"] == shown["truth"]
        if kind == "splice":
            proof = source_proofs[group]
            paragraph, a, b = proof["focus"]
            a, b = (sum(len(p["text"]) + 2 for p in proof["paragraphs"][:paragraph]) + v for v in (a, b))
            quote = proof["quote"]
            donor = native["donor"]
            word = quote[slice(*donor["donor_span"])]
            assert sha(quote) == donor["quote_sha256"] and sha(word) == donor["donor_sha256"]
            source_body = original.decode()
            assert word not in source_body and source_body[:a] + word + source_body[b:] == body
            assert native["truth"] == [a, a + len(word)]
        else:
            assert original.decode() == body and source["truth"] == native["truth"]
        observed = []
        for window in native["windows"]:
            text = " ".join(body[a:b] for a, b in window["spans"])
            assert sha(text) == window["primary"]
            if primary[window["primary"]]["score"] >= .5 and len(observed) < 6:
                observed.append(window["spans"])

        assert shown["current"] == [overlap(body, shown["truth"], spans) for spans in observed]
        for index, row in zip(native["eligible"], shown["windows"], strict=True):
            window = native["windows"][index]
            assert row["focus"] == window["focus"] and row["overlap"] == overlap(body, shown["truth"], [row["focus"]])
            left, focus, right = (body[a:b] for a, b in window["spans"])
            expected = {"left": left, "center": focus, "right": right, "full": f"{left} {focus} {right}", "omitted": f"{left} {right}"}
            responses = {}
            for label, value in expected.items():
                pin = sha(value)
                assert window["independent"][label] == pin
                evidence = independent[pin]
                assert evidence["coverage"]["token_windows"] == 1
                responses[label] = logit(evidence["score"])
            features = (logit(primary[window["primary"]]["score"]), responses["center"],
                        responses["full"] - responses["omitted"],
                        responses["center"] - max(responses["left"], responses["right"]))
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(features, row["features"], strict=True))
    print(f"Replayed {len(sources)} exact source texts, splice donors, model inputs and overlap records.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", nargs=2, metavar=("PARENT", "RUN"), type=Path)
    args = parser.parse_args()
    public = read(args.receipt)
    public_review(public)
    if args.native:
        private_review(public, *args.native)


if __name__ == "__main__":
    main()
