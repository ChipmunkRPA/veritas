"""Evaluate whether local copyediting stability can locate a rewritten sentence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from scripts.video_appearance_native_v108 import read as load
from scripts.video_appearance_native_v108 import require
from scripts.video_appearance_native_v108 import sha as digest
from scripts.video_appearance_native_v108 import write as save
from scripts.writing_focus_v3 import BASELINE_ASSETS

SCHEMA = "chipmunk.writing-blind-rewrite-probe/1"
WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
FAMILIES = ("qwen", "smol2")
PROMPT = "Copyedit this sentence lightly for clarity and natural wording. Preserve its meaning and facts. Return only the revised sentence.\n\n{sentence}"
EDIT_PROMPT = ("Rewrite this sentence in different words, preserving all facts, names and numbers. Return only one complete replacement "
               "sentence between {minimum} and {maximum} words. Use no labels, quotations or extra lines.\n\n{sentence}")
SETTINGS = {
    "qwen": {"seed": 82456, "temperature": .5, "top_p": .9},
    "smol2": {"seed": 25967, "temperature": .6, "top_p": .9},
}
MARGIN = .15
METHOD = {
    "baseline": "First six positive (raw response at least .5) existing three-sentence classifier contexts in source order.",
    "geometry": "Two exact interior sentence spans, each with 12–40 words, in a single four-sentence source excerpt.",
    "probe": "Each local instruction model independently lightly copyedits each exact isolated focus with greedy decoding.",
    "distance": "Casefold lexical-word Levenshtein edits divided by the greater lexical word count. Abstain if either probe input lacks one valid sentence.",
    "selection": "Select the focus whose probe copyedit changes fewer words by at least .15 in both models. Test each model alone and the union with current.",
    "useful": "Non-whitespace character intersection over union at least .30 with the generated replacement. Human flags counted for both untouched and genuine-prose replacements.",
    "advance": "At least two more useful source groups in the union with current, useful detections from both generation families and at most one additional human source group flagged. Fresh single-publisher development only.",
}


def cases(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def words(text):
    return [token.casefold() for token in WORDS.findall(text)]


def distance(earlier, later):
    left, right = words(earlier), words(later)
    require(left and right, "Copyedit distance needs nonempty text")
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + int(a != b)))
        previous = row
    return previous[-1] / max(len(left), len(right))


def select(values):
    if any(value is None for value in values):
        return None
    left, right = values
    if abs(left - right) < MARGIN:
        return None
    return int(right < left)


def valid_sentence(output, source, *, generated):
    from app.detectors.text.segment import spans

    if not output["eos"]:
        return None
    value = output["text"].strip()
    if not value or len(value) > 450 or "\n" in value or value[0] in '"“\'‘#*[' or value[-1] not in ".!?":
        return None
    if spans(value, [(0, len(value))]) != [(0, len(value), 0)]:
        return None
    ratio = len(words(value)) / len(words(source))
    if generated:
        if not 12 <= len(words(value)) <= 40 or not .55 <= ratio <= 1.65 or distance(value, source) < .25:
            return None
    elif not .55 <= ratio <= 1.65:
        return None
    return value


def fixtures(run):
    from app.detectors.text.extract import extract
    from app.detectors.text.segment import segment
    from scripts.writing_paired_email_revision import encoded, native_contexts

    records, probes, current = [], {}, {}
    for case in cases(run / "cases.jsonl"):
        file = run / "fixtures" / (case["id"] + ".txt")
        require(digest(file) == case["sha256"], "Fixture bytes changed")
        doc = extract(file, ".txt")
        native = segment(doc)
        require(doc.text == file.read_text() and len(doc.paragraphs) == 1 and len(native) == 4,
                "Expected four exact sentences and one paragraph")
        focus = []
        for sentence in native[1:3]:
            require(12 <= len(words(sentence.text)) <= 40, "Ineligible interior sentence")
            key = text_hash(sentence.text)
            probes[key] = sentence.text
            focus.append({"span": [sentence.start, sentence.end], "sha256": key})
        contexts = []
        for group in native_contexts(native):
            reading, text = encoded(group)
            current[reading["sha256"]] = text
            contexts.append(reading)
        records.append({"case": case, "focus": focus, "contexts": contexts})
    return records, probes, current


def generator_configuration():
    from scripts.text_elife_rewrites_v273 import FREEZE, GENERATORS

    frozen = load(FREEZE)["generators"]
    for family in FAMILIES:
        repo, revision, weights, folder = GENERATORS[family]
        entry = frozen[family]
        require((repo, revision, weights, "Apache-2.0") ==
                (entry["repo"], entry["revision"], entry["weights_sha256"], entry["license"]), "Generator identity changed")
        require(all(digest(folder / name) == value for name, value in entry["local_config_sha256"].items()),
                "Generator config or tokenizer differs")
    return frozen


def lock(run, *, probe):
    value = {"schema": SCHEMA, "code_sha256": digest(Path(__file__)), "sources_sha256": digest(run / "sources.json"),
             "originals_sha256": digest(run / "originals.jsonl"), "generators": generator_configuration(),
             "prompts": {"edit": EDIT_PROMPT, "probe": PROMPT}, "settings": SETTINGS, "method": METHOD}
    if probe:
        value.update(cases_sha256=digest(run / "cases.jsonl"), generation_freeze_sha256=digest(run / "generation-freeze.json"),
                     generation_outputs={family: digest(run / (f"generate-{family}.json")) for family in FAMILIES},
                     probe_inputs_sha256=digest(run / "probe-inputs.json"), current_assets=BASELINE_ASSETS)
    return value


def mlx(run, family, *, probe):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as model_load
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    started = monotonic()
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Generation requires offline snapshots")
    require(all(importlib.metadata.version(name) == version for name, version in MLX_VERSIONS.items()), "MLX runtime differs")
    lock_path = run / ("scoring-freeze.json" if probe else "generation-freeze.json")
    require(load(lock_path) == lock(run, probe=probe), "Experimental inputs or code differ from their freeze")
    if probe:
        inputs = load(run / "probe-inputs.json")
    else:
        inputs = {row["group"]: row["sentences"][row["target_index"]] for row in cases(run / "originals.jsonl")}
    _, _, weights, snapshot = GENERATORS[family]
    require(digest(snapshot / "model.safetensors") == weights, "Generator weights changed")
    mx.set_default_device(mx.gpu)
    cfg = SETTINGS[family]
    mx.random.seed(cfg["seed"])
    model, tokenizer = model_load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    sampler = make_sampler(temp=0 if probe else cfg["temperature"], top_p=1 if probe else cfg["top_p"])
    results = {}
    for key, sentence in inputs.items():
        count = len(words(sentence))
        prompt = PROMPT.format(sentence=sentence) if probe else EDIT_PROMPT.format(
            minimum=max(12, int(count * .65)), maximum=min(40, int(count * 1.4)), sentence=sentence)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(chat, add_special_tokens=False)
        output, finish = [], None
        for part in stream_generate(model, tokenizer, tokens, max_tokens=96, sampler=sampler):
            output.append(part.text)
            finish = part.finish_reason
        require(finish in {"stop", "length"}, "Unrecognized generator termination")
        results[key] = {"text": "".join(output), "eos": finish == "stop", "input_sha256": text_hash(sentence)}
        print(f"{family} {len(results)}/{len(inputs)}", flush=True)
    prefix = "probe" if probe else "generate"
    save(run / f"{prefix}-{family}.json", {"inputs": results, "freeze_sha256": digest(lock_path), "runtime": MLX_VERSIONS,
                                           "seconds": round(monotonic() - started, 3)})


def admission(run):
    from app.detectors.text.segment import spans

    frozen = run / "generation-freeze.json"
    require(load(frozen) == lock(run, probe=False), "Originals or code changed after generation freeze")
    generated = {family: load(run / f"generate-{family}.json") for family in FAMILIES}
    require(all(value["freeze_sha256"] == digest(frozen) for value in generated.values()), "Unfrozen generated evidence")
    folder = run / "fixtures"
    folder.mkdir()
    rows, admitted = [], Counter()
    for row in cases(run / "originals.jsonl"):
        group, original, index = row["group"], row["sentences"], row["target_index"]
        source = original[index]
        versions = {"original": source, "human_splice": row["donor"]}
        for family, attempt in generated.items():
            response = attempt["inputs"][group]
            require(response["input_sha256"] == text_hash(source), "Wrong generation focus")
            value = valid_sentence(response, source, generated=True)
            if value is not None:
                versions[family] = value
        for kind, replacement in versions.items():
            pieces = [*original[:index], replacement, *original[index + 1:]]
            text = " ".join(pieces)
            geometry = spans(text, [(0, len(text))])
            if len(geometry) != 4 or [text[a:b] for a, b, _ in geometry] != pieces:
                require(kind in FAMILIES, "Original or donor changed segmentation")
                continue
            if kind in FAMILIES:
                require(replacement not in " ".join(original), "Generated replacement equals another original sentence")
            a, b, _ = geometry[index]
            case_id = f"{group}-{kind}"
            path = folder / (case_id + ".txt")
            path.write_text(text)
            rows.append({"id": case_id, "group": group, "kind": kind, "sha256": digest(path), "replacement": [a, b],
                         "target_index": index, "replacement_sha256": text_hash(replacement)})
            admitted[kind] += 1
    with (run / "cases.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(dict(admitted), flush=True)


def overlap(text, ranges, target):
    truth = {i for i in range(*target) if not text[i].isspace()}
    marked = {i for a, b in ranges for i in range(a, b) if not text[i].isspace()}
    return bool(marked and len(truth & marked) / len(truth | marked) >= .30)


def summarize(run, snapshot):
    import torch

    from scripts.writing_paired_email_revision import native_scores

    torch.set_num_threads(1)
    started = monotonic()
    freeze = run / "scoring-freeze.json"
    require(load(freeze) == lock(run, probe=True), "Scoring freeze differs")
    require({name: digest(snapshot / name) for name in BASELINE_ASSETS} == BASELINE_ASSETS, "Current baseline snapshot differs")
    records, inputs, windows = fixtures(run)
    require(inputs == load(run / "probe-inputs.json"), "Probe input text differs from the sealed source")
    probes = {family: load(run / f"probe-{family}.json") for family in FAMILIES}
    require(all(value["freeze_sha256"] == digest(freeze) and set(value["inputs"]) == set(inputs) for value in probes.values()),
            "Probe responses do not match the frozen inputs")
    baseline, forwards = native_scores(snapshot, windows, second=False)
    result = defaultdict(Counter)
    useful_groups, human_groups = defaultdict(set), defaultdict(set)
    for record in records:
        case = record["case"]
        text = (run / "fixtures" / (case["id"] + ".txt")).read_text()
        for entry in record["contexts"]:
            entry["native"] = baseline[entry["sha256"]]
        current = [entry["spans"] for entry in record["contexts"] if entry["native"]["score"] >= .5][:6]
        measured = {family: [] for family in FAMILIES}
        for entry in record["focus"]:
            key = entry["sha256"]
            for family in FAMILIES:
                output = probes[family]["inputs"][key]
                require(output["input_sha256"] == key, "Probe response references a different input")
                response = valid_sentence(output, inputs[key], generated=False)
                measured[family].append(distance(inputs[key], response) if response is not None else None)
        selected = {family: select(values) for family, values in measured.items()}
        both = selected[FAMILIES[0]] if selected[FAMILIES[0]] == selected[FAMILIES[1]] else None
        assessments = {"current": current, **{family: [] if pos is None else [[record["focus"][pos]["span"]]]
                                              for family, pos in {**selected, "both": both}.items()}}
        assessments["union"] = current + assessments["both"]
        record.update(distances=measured, selected=assessments)
        for arm, proposed in assessments.items():
            useful = any(overlap(text, ranges, case["replacement"]) for ranges in proposed)
            group = case["group"]
            kind = case["kind"]
            result[kind][f"{arm}_flags"] += int(bool(proposed))
            result[kind][f"{arm}_locates_replacement"] += int(kind != "original" and useful)
            if kind in FAMILIES and useful:
                useful_groups[arm].add(group)
            if kind not in FAMILIES and proposed:
                human_groups[arm].add(group)
        result[case["kind"]]["cases"] += 1
        for family, values in measured.items():
            result[case["kind"]][f"{family}_available"] += int(all(value is not None for value in values))
    response = run / "private-responses.json"
    save(response, records)
    save(run / "result.json", {"schema": SCHEMA, "method": METHOD, "counts": {kind: dict(counts) for kind, counts in result.items()},
                               "human_groups": {arm: len(human_groups[arm]) for arm in ("current", *FAMILIES, "both", "union")},
                               "useful_groups": {arm: len(useful_groups[arm]) for arm in ("current", *FAMILIES, "both", "union")},
                               "new_useful_groups": len(useful_groups["union"] - useful_groups["current"]),
                               "extra_human_groups": len(human_groups["union"] - human_groups["current"]),
                               "scoring_freeze_sha256": digest(freeze), "response_sha256": digest(response),
                               "probe_inputs_per_model": len(inputs), "current_inputs": len(windows),
                               "native_current_forwards": forwards, "scoring_seconds": round(monotonic() - started, 3),
                               "probe_seconds": {family: value["seconds"] for family, value in probes.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generation-freeze", "generate", "admit", "scoring-freeze", "probe", "summarize"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--current-snapshot", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    run = args.directory
    if args.stage.endswith("-freeze"):
        if args.stage == "scoring-freeze":
            save(run / "probe-inputs.json", fixtures(run)[1])
        save(run / f"{args.stage}.json", lock(run, probe=args.stage == "scoring-freeze"))
    elif args.stage in {"generate", "probe"}:
        require(args.family in FAMILIES, "Choose a local model family")
        mlx(run, args.family, probe=args.stage == "probe")
    elif args.stage == "admit":
        admission(run)
    else:
        require(args.current_snapshot is not None, "Choose the current local snapshot")
        summarize(run, args.current_snapshot)


if __name__ == "__main__":
    main()
