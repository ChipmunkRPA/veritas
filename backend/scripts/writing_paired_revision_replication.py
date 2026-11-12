"""Retest the frozen two-version writing rule with different local generators."""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from time import monotonic

from scripts.video_appearance_native_v108 import read, require
from scripts.video_appearance_native_v108 import write as save
from scripts.writing_blind_rewrite_probe import (
    EDIT_PROMPT,
    generator_configuration,
    text_hash,
    valid_sentence,
    words,
)
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-paired-revision-replication/1"
SEEDS = {"qwen": 31916, "smol2": 28271}
ARMS = ("current", "agreement", "paired", "combined")
SETTINGS = {"temperature": .60, "top_p": .90, "max_tokens": 96}
PUBLISHED = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/writing-reference-revision-transfer-2026-11-11.json"
METHOD = {
    "diagnostic": "Reuse the exact earlier and later F1000 versions. Reuse prior detector responses only for byte-identical "
                  "native input. Different generation weights and seed; same source groups as the previous transfer.",
    "holdout": "Use qualified publisher-deposited versions from works absent from the earlier source panel. "
               "Retain native-window geometry and group by publisher and generator. No reuse of prior scores.",
    "generation": "Exactly one completed rewrite attempt at each original center per generator. All flanking sentences "
                  "must remain exact. Reject invalid output or an exact duplicate of the known human revision.",
    "current": "First six native windows at or above the existing raw comparison point of 0.5.",
    "agreement": "Current positives must have a second response of at least 0.5 on the exact same input.",
    "paired": "With exact matched neighboring sentences, mark the center only when both later model-specific "
              "AI-like odds are independently at least twice the earlier odds.",
    "combined": "Use agreement on changed native inputs and the paired center.",
    "outcomes": "Useful localization is at least 0.30 non-whitespace character IoU and 0.50 recall. "
                "Also count exact center, wrong-only, no flag and false flags on natural later revisions.",
    "replication": "An adequate additional generator lane admits at least six source groups. It supports the result "
                   "if paired adds at least two useful source groups, loses none and adds at most one human "
                   "false-flag group over same-window agreement. Reused F1000 sources remain diagnostic.",
}


def lock(run, source, scope):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    published = read(PUBLISHED)
    originals, receipt = source / "originals.json", source / "source-receipt.json"
    require(read(receipt)["originals_sha256"] == digest(originals) and read(originals), "Unverified source roster")
    prior = None
    if scope == "diagnostic":
        names = ("source-receipt.json", "aggregate.json", "current.json", "second.json", "audit.json")
        prior = {name: digest(source / name) for name in names}
        require(all(published["commitments"][name] == value for name, value in prior.items()),
                "Source exposure or prior score commitments differ")
    else:
        from scripts.writing_publisher_revision_holdout import SCHEMA as HOLDOUT

        require(read(receipt)["schema"] == HOLDOUT, "Publisher holdout lacks its exact source receipt")
    metadata = generator_configuration()
    identities = {}
    for family in SEEDS:
        _, _, expected, folder = GENERATORS[family]
        require(metadata[family]["license"] == "Apache-2.0" and digest(folder / "model.safetensors") == expected,
                "Generator weights or terms differ")
        identities[family] = metadata[family]
    return {"schema": SCHEMA, "scope": scope, "method": METHOD, "code_sha256": digest(Path(__file__)),
            "source_sha256": digest(originals), "source_receipt_sha256": digest(receipt), "prior": prior,
            "generator_assets": identities, "runtime": MLX_VERSIONS, "seeds": SEEDS,
            "prompt": EDIT_PROMPT, "settings": SETTINGS, "published_f1000_sha256": digest(PUBLISHED)}


def generate(run, source, protocol, family):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS

    require(all(version(key) == value for key, value in protocol["runtime"].items()), "Generator runtime differs")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEEDS[family])
    model, tokenizer = load(str(GENERATORS[family][3]), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    sampler = make_sampler(temp=SETTINGS["temperature"], top_p=SETTINGS["top_p"])
    results = {}
    for row in read(source / "originals.json"):
        original = row["before"]["focus_text"]
        count = len(words(original))
        prompt = EDIT_PROMPT.format(sentence=original, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(chat, add_special_tokens=False)
        reply = list(stream_generate(model, tokenizer, tokens, max_tokens=SETTINGS["max_tokens"], sampler=sampler))
        finish = reply[-1].finish_reason
        require(finish in {"stop", "length"}, "Unknown generator termination")
        results[row["group"]] = {"text": "".join(part.text for part in reply), "eos": finish == "stop",
                                 "input_sha256": text_hash(original)}
        print(f"{family} {len(results)}", flush=True)
    save(run / f"generated-{family}.json", {"freeze_sha256": digest(run / "freeze.json"), "results": results})


def register(run, source):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_revision import compare
    from scripts.writing_paired_email_revision import encoded, native_contexts

    originals = read(source / "originals.json")
    generations = {family: read(run / f"generated-{family}.json") for family in SEEDS}
    require(all(packet["freeze_sha256"] == digest(run / "freeze.json")
                and packet["results"].keys() == {row["group"] for row in originals}
                for packet in generations.values()), "Generation receipt or group inventory differs")
    inputs, rows, admission = {}, [], defaultdict(Counter)
    for article in originals:
        before, after = article["before"], article["after"]
        original = before["focus_text"]
        start, stop = before["focus"]
        versions = {"before": before, "edited": after}
        for family, packet in generations.items():
            attempt = packet["results"][article["group"]]
            require(attempt["input_sha256"] == text_hash(original), "Generated focus differs")
            admission[family]["attempted"] += 1
            replacement = valid_sentence(attempt, original, generated=True)
            if replacement is None or replacement == after["focus_text"]:
                continue
            revised = before["text"][:start] + replacement + before["text"][stop:]
            doc = from_string(revised)
            units = segment(doc)
            if doc.text != revised or len(units) != 5 or units[2].text != replacement:
                continue
            versions[family] = {"text": revised, "focus": [start, start + len(replacement)], "focus_text": replacement}
            admission[family]["admitted"] += 1
        reference, earlier_flanks = None, None
        for role, revision in versions.items():
            text = revision["text"]
            doc = from_string(text)
            sentences = segment(doc)
            require(len(sentences) == 5 and doc.text == text and [sentences[2].start, sentences[2].end] == revision["focus"],
                    "Native focus geometry differs")
            flanks = [sentence.text for i, sentence in enumerate(sentences) if i != 2]
            require(earlier_flanks is None or flanks == earlier_flanks, "A matched neighboring sentence changed")
            earlier_flanks = flanks
            windows = []
            for span in native_contexts(sentences):
                row, value = encoded(span)
                windows.append(row)
                inputs[row["sha256"]] = value
            native = [row["sha256"] for row in windows]
            reference = native if reference is None else reference
            diff = compare(before["text"].encode(), text.encode(), ".txt")["changes"]
            before_a, before_b = before["focus"]
            after_a, after_b = revision["focus"]
            bounded = bool(diff) and all(
                change["role"] == "body"
                and before_a <= change["earlier"]["start"] <= change["earlier"]["end"] <= before_b
                and after_a <= change["later"]["start"] <= change["later"]["end"] <= after_b for change in diff
            )
            rows.append({"group": article["group"], "publisher": article.get("publisher", "f1000"),
                         "role": role, "text": text, "text_sha256": text_hash(text), "target": 2,
                         "focus": revision["focus"], "windows": windows, "reference": reference,
                         "draft_hunks": len(diff), "hunks_within_focus": bounded})
    return rows, inputs, admission


def scores_lock(run, current, second):
    from scripts.writing_paired_email_revision import SECONDARY

    assets = {"current": BASELINE_ASSETS, "second": SECONDARY}
    require(all(digest(path / name) == sha for path, arm in ((current, "current"), (second, "second"))
                for name, sha in assets[arm].items()), "Checkpoint assets differ")
    return {"freeze_sha256": digest(run / "freeze.json"), "cases_sha256": digest(run / "cases.json"),
            "generation": {name: digest(run / f"generated-{name}.json") for name in SEEDS}, "checkpoint_assets": assets}


def score(run, source, protocol, arm, path, values):
    import torch

    from scripts.writing_paired_email_revision import native_scores

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cache = read(source / f"{arm}.json")["responses"] if protocol["scope"] == "diagnostic" else {}
    missing = {key: value for key, value in values.items() if key not in cache}
    start = monotonic()
    fresh, forwards = native_scores(path, missing, second=arm == "second") if missing else ({}, 0)
    responses = {key: (cache[key] if key in cache else fresh[key]) for key in values}
    save(run / f"{arm}.json", {"freeze_sha256": digest(run / "scoring-freeze.json"), "responses": responses,
                              "forwards": forwards, "cached": len(values) - len(missing),
                              "seconds": round(monotonic() - start, 3)})


def outcomes(row, first, second):
    from scripts.writing_paired_email_revision import rise

    values = [window["sha256"] for window in row["windows"]]
    baseline = row["reference"]
    center = row["target"]
    a, b = ([scores[key]["score"] for key in values] for scores in (first, second))
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in a + b), "A checkpoint response is invalid")
    current = [i for i, raw in enumerate(a) if raw >= .5][:6]
    agreement = [i for i in current if b[i] >= .5]
    paired = values[center] != baseline[center] and all(
        rise(model[baseline[center]]["score"], model[values[center]]["score"]) >= math.log(2) for model in (first, second)
    )
    result = {"unchanged_agreement": any(values[i] == baseline[i] for i in agreement),
              "paired_eligible": values[center] != baseline[center],
              "draft_hunks": row["draft_hunks"], "hunks_within_focus": row["hunks_within_focus"]}
    focus = row["focus"]
    truth = {i for i in range(*focus) if not row["text"][i].isspace()}
    paired_windows = [None] if paired else []
    changed = [i for i in agreement if values[i] != baseline[i]]
    for arm, chosen in zip(ARMS, (current, agreement, paired_windows, changed + paired_windows), strict=True):
        useful = False
        for index in chosen:
            spans = [focus] if index is None else row["windows"][index]["spans"]
            pixels = {i for a, b in spans for i in range(a, b) if not row["text"][i].isspace()}
            overlap = len(truth & pixels)
            useful |= 10 * overlap >= 3 * len(truth | pixels) and 2 * overlap >= len(truth)
        hit = row["role"] in SEEDS and useful
        result.update({f"{arm}_flag": bool(chosen), f"{arm}_useful": hit,
                       f"{arm}_exact_focus": center in chosen or None in chosen,
                       f"{arm}_wrong_only": row["role"] in SEEDS and bool(chosen) and not hit,
                       f"{arm}_no_flag": not chosen})
    return result


def aggregate(run, protocol, rows, values, admission):
    from scripts.review_writing_reference_revision_transfer import native

    receipts = [read(run / f"{arm}.json") for arm in ("current", "second")]
    require(all(receipt["freeze_sha256"] == digest(run / "scoring-freeze.json")
                and receipt["responses"].keys() == values.keys() for receipt in receipts), "Score receipt differs")
    counts, exact, cases = defaultdict(Counter), set(), []
    for row in rows:
        _, inputs = native(row)
        exact.update(inputs)
        result = outcomes(row, *(receipt["responses"] for receipt in receipts))
        counts[f"{row['publisher']}/{row['role']}"].update({"cases": 1, **result})
        cases.append({key: row[key] for key in ("group", "publisher", "role", "text_sha256")} | result)
    require(exact == values.keys(), "Exact independent native input replay differs")
    human = [case for case in cases if case["role"] == "edited"]
    control = {case["group"] for case in human if case["agreement_flag"]}
    added_false = {case["group"] for case in human if case["paired_flag"]} - control
    decisions = {}
    for family in SEEDS:
        lane = [case for case in cases if case["role"] == family]
        gains = {case["group"] for case in lane if case["paired_useful"] and not case["current_useful"]}
        losses = {case["group"] for case in lane if case["current_useful"] and not case["paired_useful"]}
        decisions[family] = {"admitted_groups": len(lane), "added_useful_groups": len(gains),
                             "lost_useful_groups": len(losses), "added_false_groups": len(added_false),
                             "replicated": len(lane) >= 6 and len(gains) >= 2 and not losses and len(added_false) <= 1}
    return cases, {"schema": SCHEMA, "scope": protocol["scope"], "method": METHOD, "distinct_inputs": len(values),
                   "score_freeze_sha256": digest(run / "scoring-freeze.json"), "admission": admission, "counts": counts,
                   "decisions": decisions, "cached_responses": {arm: r["cached"] for arm, r in zip(("current", "second"), receipts, strict=True)},
                   "native_forwards": {arm: r["forwards"] for arm, r in zip(("current", "second"), receipts, strict=True)},
                   "score_receipts": {arm: digest(run / f"{arm}.json") for arm in ("current", "second")}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "freeze-scoring", "current", "second", "report", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("source", type=Path)
    cli.add_argument("--scope", choices=("diagnostic", "holdout"), required=True)
    cli.add_argument("--family", choices=SEEDS)
    cli.add_argument("--current", type=Path)
    cli.add_argument("--second", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with offline snapshots")
    run, source, phase = args.run, args.source, args.phase
    protocol = lock(run, source, args.scope)
    freeze = run / "freeze.json"
    if phase == "freeze":
        save(freeze, protocol)
        return
    require(read(freeze) == protocol, "Generation or source freeze differs")
    if phase == "generate":
        require(args.family, "Choose a frozen generator")
        generate(run, source, protocol, args.family)
        return
    rows, values, admission = register(run, source)
    cases, locked = run / "cases.json", run / "scoring-freeze.json"
    if phase == "freeze-scoring":
        save(cases, rows)
        save(locked, scores_lock(run, args.current, args.second))
        print(f"Frozen {len(rows)} cases and {len(values)} exact native inputs.", flush=True)
        return
    require(read(cases) == rows and read(locked) == scores_lock(run, args.current, args.second), "Scoring freeze differs")
    if phase in ("current", "second"):
        score(run, source, protocol, phase, getattr(args, phase), values)
        return
    results, summary = aggregate(run, protocol, rows, values, admission)
    path, result = run / "private-outcomes.json", run / "aggregate.json"
    if phase == "report":
        save(path, results)
        save(result, summary | {"outcomes_sha256": digest(path)})
    else:
        require(read(path) == results and read(result) == summary | {"outcomes_sha256": digest(path)}, "Outcome replay differs")
    print(summary["decisions"], flush=True)


if __name__ == "__main__":
    main()
