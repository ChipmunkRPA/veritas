"""Compare direct local sentence judgments with and without balanced answer labels."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from time import monotonic

from scripts.writing_blind_rewrite_probe import (
    BASELINE_ASSETS,
    FAMILIES,
    digest,
    fixtures,
    generator_configuration,
    load,
    overlap,
    require,
    save,
    text_hash,
)

SCHEMA = "chipmunk.writing-option-order-transfer/1"
OPTIONS = ("Sentence 2", "Sentence 3", "Neither")
ORDERS = tuple(permutations(range(len(OPTIONS))))
LABELS = ("A", "B", "C")
PROMPT = ("Read the passage below. One of the two interior sentences may have been rewritten by an AI tool. "
          "Which sentence, if either, shows clearer evidence of AI rewriting? Ordinary human revisions are possible. "
          "If neither sentence shows clear evidence, choose Neither. Reply with just one of the option letters.\n\n"
          "{passage}\n\nOptions:\n{options}")
MIN_LABEL_MASS = .20
MIN_ODDS = 1.50
MIN_VOTES = 4
METHOD = {
    "current": "Existing first-six positive native three-sentence context regions, with raw point .5.",
    "input": "One submitted passage, original sentence order fixed. Each candidate label A/B/C represents sentence 2, sentence 3 or neither. All six assignments are scored.",
    "response": "Sum full-vocabulary next-token probabilities of bare and leading-space uppercase labels where each is a single token. Each permutation requires at least .20 total label mass to be assessed.",
    "fixed": "First assignment's most likely label requires at least 1.50 times the second label's probability.",
    "balanced": "Require all six valid assignments, a shared semantic choice winning at least four times, and its mean log score at least log(1.50) ahead of either other semantic choice.",
    "agreement": "Both models independently select the same source sentence after balancing. Neither and inconclusive answers do not propose a source span.",
    "useful": "At least .30 non-whitespace character intersection over union with the single authored edit. Count flags on untouched and human-spliced controls and group by publisher.",
    "advance": "Agreement added to current must add useful edit groups in both publishers and both generator families, with no more than one extra flagged human source group. Development scope only.",
}


def prompt(sentences, order):
    passage = "\n".join(f"{i}. {text}" for i, text in enumerate(sentences, 1))
    options = "\n".join(f"{label}. {OPTIONS[target]}" for label, target in zip(LABELS, order, strict=True))
    return PROMPT.format(passage=passage, options=options)


def choose(readings):
    by_order, votes, likelihoods = {}, Counter(), [[], [], []]
    fixed = None
    for order, row in zip(ORDERS, readings, strict=True):
        probabilities = row["label_probability"]
        require(len(probabilities) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
                and sum(probabilities) <= 1.000001, "Invalid native label probabilities")
        if sum(probabilities) < MIN_LABEL_MASS or min(probabilities) == 0:
            continue
        winner, next_best, _ = sorted(range(3), key=lambda i: probabilities[i], reverse=True)
        mapped = order[winner]
        votes[mapped] += 1
        by_order[order] = mapped
        for i, value in enumerate(probabilities):
            likelihoods[order[i]].append(math.log(value))
        if order == ORDERS[0] and probabilities[winner] >= MIN_ODDS * probabilities[next_best]:
            fixed = mapped
    balanced = None
    if len(by_order) == len(ORDERS):
        means = [sum(values) / len(values) for values in likelihoods]
        winner, next_best, _ = sorted(range(3), key=means.__getitem__, reverse=True)
        if votes[winner] >= MIN_VOTES and means[winner] - means[next_best] >= math.log(MIN_ODDS):
            balanced = winner
    return {"fixed": fixed, "balanced": balanced, "orders_available": len(by_order),
            "order_disagreement": len(votes) > 1}


def freeze(run):
    old = Path(__file__).with_name("writing_blind_rewrite_probe.py")
    return {"schema": SCHEMA, "method": METHOD, "prompt": PROMPT, "labels": list(LABELS), "orders": [list(order) for order in ORDERS],
            "settings": {"minimum_label_mass": MIN_LABEL_MASS, "minimum_odds": MIN_ODDS, "minimum_votes": MIN_VOTES},
            "code_sha256": digest(Path(__file__)), "fixture_runner_sha256": digest(old),
            "cases_sha256": digest(run / "cases.jsonl"), "source_sha256": digest(run / "sources.json"),
            "intake_sha256": digest(run / "scoring-freeze.json"), "requests_sha256": digest(run / "requests.json"),
            "generators": generator_configuration(), "current_assets": BASELINE_ASSETS}


def requests(run, records):
    out = {}
    for record in records:
        case = record["case"]
        text = (run / "fixtures" / (case["id"] + ".txt")).read_text()
        source = [reading["spans"][0 if i == 0 else 1] for i, reading in enumerate(record["contexts"])]
        sentences = [text[a:b] for a, b in source]
        require(" ".join(sentences) == text, "Prompt source spans changed")
        out[case["id"]] = [{"source_spans": source, "prompt": prompt(sentences, order)} for order in ORDERS]
    return out


def verified_freeze(run):
    frozen = run / "order-freeze.json"
    require(load(frozen) == freeze(run), "Experiment freeze changed")
    return digest(frozen)


def native(run, family):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as model_load

    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    start = monotonic()
    freeze_hash = verified_freeze(run)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Generation requires local snapshots")
    require(all(importlib.metadata.version(name) == value for name, value in MLX_VERSIONS.items()), "Runtime differs")
    _, _, weights, snapshot = GENERATORS[family]
    require(digest(snapshot / "model.safetensors") == weights, "Model weights differ")
    mx.set_default_device(mx.gpu)
    model, tokenizer = model_load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    token_ids = []
    for label in LABELS:
        aliases = set()
        for spelling in (label, " " + label):
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1 and tokenizer.decode(ids).strip() == label:
                aliases.add(ids[0])
        require(aliases, "A permitted answer label has no exact single-token representation")
        token_ids.append(sorted(aliases))
    require(len(set().union(*token_ids)) == sum(map(len, token_ids)), "Answer token aliases overlap")
    result = {}
    for case, entries in load(run / "requests.json").items():
        rows = []
        for entry in entries:
            value = entry["prompt"]
            chat = tokenizer.apply_chat_template([{"role": "user", "content": value}], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer.encode(chat, add_special_tokens=False)
            require(0 < len(encoded) <= 512, "Candidate input exceeds bound")
            logits = model(mx.array([encoded]))[0, -1].astype(mx.float32)
            totals = [mx.exp(mx.logsumexp(logits[mx.array(ids)]) - mx.logsumexp(logits)) for ids in token_ids]
            rows.append({"prompt_sha256": text_hash(value), "chat_sha256": text_hash(chat), "input_tokens": len(encoded),
                         "label_probability": [float(total.item()) for total in totals]})
        result[case] = rows
        print(f"{family} {len(result)} cases", flush=True)
    save(run / f"order-{family}.json", {"freeze_sha256": freeze_hash, "runtime": MLX_VERSIONS,
                                        "label_token_ids": token_ids, "seconds": round(monotonic() - start, 3), "cases": result})


def readout(run, snapshot):
    import torch

    from scripts.writing_paired_email_revision import native_scores

    torch.set_num_threads(1)
    freeze_hash = verified_freeze(run)
    require({name: digest(snapshot / name) for name in BASELINE_ASSETS} == BASELINE_ASSETS, "Current snapshot differs")
    records, _, windows = fixtures(run)
    inputs = requests(run, records)
    require(inputs == load(run / "requests.json"), "Native prompts differ from exact source")
    arms = {family: load(run / f"order-{family}.json") for family in FAMILIES}
    require(all(arm["freeze_sha256"] == freeze_hash and set(arm["cases"]) == set(inputs) for arm in arms.values()), "Probe receipt differs")
    baseline, forwards = native_scores(snapshot, windows, second=False)
    sources = load(run / "sources.json")["sources"]
    counts, useful, human, useful_by_family = defaultdict(Counter), defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        case = record["case"]
        key, group, kind = case["id"], case["group"], case["kind"]
        publisher = sources[group]["publisher"]
        text = (run / "fixtures" / (key + ".txt")).read_text()
        current = []
        for context in record["contexts"]:
            context["native"] = baseline[context["sha256"]]
            if context["native"]["score"] >= .5:
                current.append(context["spans"])
        record["choices"] = {}
        for family, arm in arms.items():
            readings = arm["cases"][key]
            require([x["prompt_sha256"] for x in readings] == [text_hash(x["prompt"]) for x in inputs[key]], "Native prompt mismatch")
            record["choices"][family] = choose(readings)
        selected = {f"{family}_{mode}": summary[mode] for family, summary in record["choices"].items() for mode in ("fixed", "balanced")}
        a, b = (record["choices"][family]["balanced"] for family in FAMILIES)
        selected["agreement"] = a if a == b else None
        proposals = {"current": current[:6]}
        for name, item in selected.items():
            proposals[name] = [[record["focus"][item]["span"]]] if item in (0, 1) else []
        proposals["union"] = proposals["current"] + proposals["agreement"]
        record["proposals"] = proposals
        for scope in (("all", kind), (publisher, kind)):
            counts[scope]["cases"] += 1
            for family, choice in record["choices"].items():
                counts[scope][f"{family}_all_orders"] += choice["orders_available"] == len(ORDERS)
                counts[scope][f"{family}_disagreed_by_order"] += choice["order_disagreement"]
                for mode in ("fixed", "balanced"):
                    counts[scope][f"{family}_{mode}_abstentions"] += choice[mode] is None
                    counts[scope][f"{family}_{mode}_neither"] += choice[mode] == 2
            for arm, regions in proposals.items():
                hit = any(overlap(text, window, case["replacement"]) for window in regions)
                counts[scope][f"{arm}_flags"] += bool(regions)
                counts[scope][f"{arm}_locations"] += hit and kind != "original"
                if kind in FAMILIES and hit:
                    useful[scope[0], arm].add(group)
                    if scope[0] == "all":
                        useful_by_family[kind, arm].add(group)
                if kind not in FAMILIES and regions:
                    human[scope[0], arm].add(group)
    response = run / "private-order-readout.json"
    save(response, records)
    scopes = {scope for scope, _ in counts}
    arm_names = list(proposals)
    extra_useful = {s: len(useful[s, "union"] - useful[s, "current"]) for s in sorted(scopes)}
    extra_human = {s: len(human[s, "union"] - human[s, "current"]) for s in sorted(scopes)}
    extra_by_family = {family: len(useful_by_family[family, "union"] - useful_by_family[family, "current"])
                       for family in FAMILIES}
    advances = (all(extra_useful[p] > 0 for p in scopes - {"all"})
                and all(value > 0 for value in extra_by_family.values()) and extra_human["all"] <= 1)
    save(run / "order-result.json", {"schema": SCHEMA, "method": METHOD, "freeze_sha256": freeze_hash,
                                     "readout_sha256": digest(response), "cases": len(records), "native_current_forwards": forwards,
                                     "native_current_inputs": len(windows), "order_forwards_per_model": len(records) * len(ORDERS),
                                     "native_probe_seconds": {k: arm["seconds"] for k, arm in arms.items()},
                                     "counts": {scope: {kind: dict(row) for (s, kind), row in counts.items() if scope == s} for scope in sorted(scopes)},
                                     "useful_groups": {s: {a: len(useful[s, a]) for a in arm_names} for s in sorted(scopes)},
                                     "human_groups": {s: {a: len(human[s, a]) for a in arm_names} for s in sorted(scopes)},
                                     "extra_useful_groups": extra_useful, "extra_human_groups": extra_human,
                                     "extra_useful_groups_by_generator": extra_by_family, "advance": advances})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "probe", "readout"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--current-snapshot", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    run = args.directory
    if args.stage == "freeze":
        save(run / "requests.json", requests(run, fixtures(run)[0]))
        save(run / "order-freeze.json", freeze(run))
    elif args.stage == "probe":
        require(args.family in FAMILIES, "Choose one checkpoint")
        native(run, args.family)
    else:
        require(args.current_snapshot is not None, "Choose the current snapshot")
        readout(run, args.current_snapshot)


if __name__ == "__main__":
    main()
