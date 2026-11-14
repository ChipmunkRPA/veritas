"""Test whether a contextual sentence score and its separation from neighbors transfer."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from scripts import writing_typed_decision_transfer as typed

source = typed.source
SCHEMA = "chipmunk.writing-context-margin-transfer/1"
PRIOR = typed.short.PANELS / "writing-typed-decision-transfer-2026-11-13.json"
ARMS = ("context", "margin")
GEOMETRY = (*typed.GEOMETRY, "target_index")
METHOD = {
    "selection": "Twenty report works with the existing source and phrase exclusions. Exclude all work commitments "
                 "in published writing ledgers. A second twenty-work fill was too small at source admission; "
                 "the first fill was kept before any inference.",
    "points": "On the 20 genuine GAO works from the completed typed study, freeze the second-highest authentic maximum "
              "for each arm. This is a comparison operating point, not a probability or production calibration.",
    "context": "Choose the highest untied interior sentence by mean of the two typed contextual responses.",
    "margin": "Require the same winning interior sentence to pass the domain-matched context point and independently "
              "strictly exceed the frozen separation point versus the stronger of the two other interior readings.",
    "comparison": "Retain the earlier cross-domain context point as a separate comparator. Re-run the existing "
                  "classifier on all five native three-sentence contexts with its original .50 point.",
    "advance": "Compare context plus margin to the domain-matched context reading at fixed points. Require at least "
               "three fewer untouched flags and at most two fewer correct locations, with at least eight correct "
               "locations across both generators. Also report the raw incumbent and all overlaps. Zero confidence weight.",
}


def calibrate(prior):
    scores = {row["group"]: peak(prior["rows"][row["id"]]["context"])
              for row in prior["geometry"] if row["cohort"] == "fresh" and row["family"] == "original"}
    source.require(len(scores) == 20 and all(value is not None for value in scores.values()),
                   "Twenty complete authentic work readings required")
    return {"context": sorted(v[0] for v in scores.values())[-2],
            "margin": sorted(v[1] for v in scores.values())[-2],
            "earlier_context": prior["result"]["points"]["context"]["cutoff"]}


def peak(responses):
    winner = typed.peak(responses)
    if winner is None:
        return None
    values = sorted((round(sum(votes) / 2, 5) for votes in responses), reverse=True)
    return winner[0], round(values[0] - values[1], 5), winner[1]


def freeze(run, cache):
    prior = typed.source.read(PRIOR)
    source.require(source.serialize(typed.review(prior)) == source.serialize(prior["result"]), "Prior ledger failed review")
    known = sorted(typed.short.PANELS.glob("writing-*.json"))
    known = [p for p in known if p.name != "writing-context-margin-transfer-2026-11-13.json"]
    groups = {match for path in known for match in re.findall(r'(?<=")[0-9a-f]{64}(?=")', path.read_text())}
    rows, selection = typed.intake.intake(cache, [{"group": group} for group in groups])
    source.seal(run / "roster.json", rows)
    source.seal(run / "freeze.json", {
        "schema": SCHEMA, "method": METHOD, "prior_sha256": source.digest(PRIOR),
        "points": calibrate(prior), "selection": selection,
        "earlier_ledgers": {path.name: source.digest(path) for path in known},
        "runner_sha256": source.digest(Path(__file__)), "roster_sha256": source.digest(run / "roster.json"),
        "candidate_revision": typed.MODEL_REVISION, "candidate_manifest": typed.MODEL_MANIFEST,
        "candidate_runtime": typed.RUNTIME, "question": typed.QUESTION, "options": typed.OPTIONS,
        "incumbent_revision": typed.BASELINE_REVISION, "versions": prior["freeze"]["versions"],
        "source_assets": prior["freeze"]["source_files"], "source_revision": prior["freeze"]["source_revision"],
    })
    print(selection, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["schema"] == SCHEMA and lock["method"] == METHOD
                   and lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["roster_sha256"] == source.digest(run / "roster.json"), "Source or procedure changed")
    return lock, source.read(run / "roster.json")


def read_candidate(run, path):
    import laya_mlx
    from laya_mlx.common import build_prefix, serialize_state

    _, rows = locked(run)
    source.require(os.getenv("HF_HUB_OFFLINE") == "1" and all(version(name) == v for name, v in typed.RUNTIME.items()),
                   "Use the pinned offline runtime")
    manifest = path / "manifest.json"
    source.require(source.digest(manifest) == typed.MODEL_MANIFEST, "Conversion manifest differs")
    for name, record in source.read(manifest)["files"].items():
        source.require(source.digest(path / name) == record["sha256"], f"Converted {name} differs")
    agent = laya_mlx.load(path, dtype="float16")
    questions = {k: {"type": "choice", "instructions": typed.QUESTION, "criteria": dict(v)}
                 for k, v in (("forward", typed.OPTIONS), ("reverse", typed.OPTIONS[::-1]))}
    prefix = max(len(build_prefix(agent.tok, agent._to_internal(q), agent.cfg["head_max_len"])[0])
                 for q in questions.values())
    inputs = {typed.state(row, i, "context") for row in rows for i in range(1, 4)}
    observed = {}
    for index, text in enumerate(sorted(inputs, key=source.sha), 1):
        ids = agent.tok(serialize_state(text).replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        if len(ids) <= agent.cfg["max_len"] - prefix - 1:
            answers = agent.predict(text, questions)["answers"]
            observed[text] = [answers[q]["probabilities"]["ai"] for q in questions]
        if index % 100 == 0:
            print("Candidate contexts", index, len(inputs), flush=True)
    result = {row["id"]: [observed.get(typed.state(row, i, "context")) for i in range(1, 4)] for row in rows}
    source.seal(run / "candidate.json", {"freeze": source.digest(run / "freeze.json"), "rows": result})


def read_incumbent(run, snapshot):
    from app.detectors.text.classifier import FourClassTextClassifier

    lock, rows = locked(run)
    typed.prior.torch_runtime(lock, snapshot, typed.BASELINE_ASSETS)
    texts = sorted({text for row in rows for text in row["context"]}, key=source.sha)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    observed = {}
    for start in range(0, len(texts), 48):
        batch = texts[start:start + 48]
        observed.update(zip(batch, classifier.predict(batch, batch=6, check=lambda _: None), strict=True))
        print("Incumbent contexts", len(observed), len(texts), flush=True)
    result = {row["id"]: [observed[text] for text in row["context"]] for row in rows}
    source.seal(run / "incumbent.json", {"freeze": source.digest(run / "freeze.json"), "rows": result})


def evaluate(rows, candidate, incumbent, points):
    groups, counts = defaultdict(set), defaultdict(lambda: defaultdict(Counter))
    outcomes = {}
    ids = {row["id"] for row in rows}
    source.require(len(rows) == len(ids) == 60 and ids == candidate.keys() == incumbent.keys(), "Incomplete response set")
    for row in rows:
        row_id, spans, truth, family = (row[key] for key in ("id", "spans", "truth", "family"))
        index = row["target_index"]
        source.require(1 <= index <= 3 and len(spans) == 5 and (truth is None if family == "original" else truth == spans[index]),
                       "Unexpected source geometry")
        groups[row["group"]].add(family)
        responses = candidate[row_id]
        previous = incumbent[row_id]
        source.require(len(responses) == 3 and len(previous) == 5 and all(0 <= value <= 1 for value in previous),
                       "Unexpected response geometry")
        old = [[spans[max(0, i - 1)][0], spans[min(4, i + 1)][1]]
               for i, score in enumerate(previous) if score >= .5]
        reading = peak(responses)
        locations = dict(incumbent=old, earlier_context=[], context=[], margin=[], incumbent_plus_margin=list(old))
        if reading:
            absolute, margin, winner = reading
            if absolute > points["earlier_context"]:
                locations["earlier_context"] = [spans[winner]]
            if absolute > points["context"]:
                locations["context"] = [spans[winner]]
                if margin > points["margin"]:
                    locations["margin"] = [spans[winner]]
                    locations["incumbent_plus_margin"].append(spans[winner])
        for arm, regions in locations.items():
            useful = any(typed.short.useful(region, truth, set(row["spaces"])) for region in regions)
            counts[family][arm].update(files=1, assessed=bool(reading) or arm.startswith("incumbent"),
                                       flagged=bool(regions), useful=useful)
        outcomes[row_id] = locations
    expected = {"original", *typed.interval.FAMILIES}
    source.require(len(groups) == 20 and all(values == expected for values in groups.values()), "Incomplete paired works")
    edits = {arm: sum(counts[f][arm]["useful"] for f in typed.interval.FAMILIES) for arm in ARMS}
    original = counts["original"]
    passes = (original["context"]["flagged"] - original["margin"]["flagged"] >= 3
              and edits["context"] - edits["margin"] <= 2 and edits["margin"] >= 8
              and all(counts[f]["margin"]["useful"] > 0 for f in typed.interval.FAMILIES))
    return {"counts": counts, "outcomes": outcomes, "passes": passes}


def review(packet):
    lock, rows = (packet[key] for key in ("freeze", "geometry"))
    earlier = source.read(PRIOR)
    source.require(lock["schema"] == SCHEMA and lock["method"] == METHOD
                   and lock["prior_sha256"] == source.digest(PRIOR)
                   and lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["points"] == calibrate(earlier), "Frozen procedure or points differ")
    prior_groups = {row["group"] for row in earlier["geometry"]}
    observed = Counter(str(row["target_index"]) for row in rows if row["family"] == "original")
    source.require(not (prior_groups & {row["group"] for row in rows})
                   and observed == lock["selection"]["target_positions"], "Held source overlap or changed target balance")
    return evaluate(rows, packet["candidate"], packet["incumbent"], lock["points"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "candidate", "incumbent", "publish", "review"))
    parser.add_argument("run", type=Path)
    for name in ("cache", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.run, args.cache)
    elif args.stage == "candidate":
        read_candidate(args.run, args.snapshot)
    elif args.stage == "incumbent":
        read_incumbent(args.run, args.snapshot)
    elif args.stage == "review":
        packet = source.read(args.run)
        result = review(packet)
        source.require(source.serialize(result) == source.serialize(packet["result"]), "Recorded outcomes differ")
        print(source.serialize({"counts": result["counts"], "passes": result["passes"]}), end="")
    else:
        lock, rows = locked(args.run)
        packet = {"freeze": lock, "geometry": [{key: row[key] for key in GEOMETRY} for row in rows]}
        for arm in ("candidate", "incumbent"):
            raw = source.read(args.run / f"{arm}.json")
            source.require(raw["freeze"] == source.digest(args.run / "freeze.json"), "Replayed responses differ from freeze")
            packet[arm] = raw["rows"]
        source.seal(args.output, packet | {"result": review(packet)})


if __name__ == "__main__":
    main()
