"""Compare source-scoped mixed-writing evidence on previously unused article revisions."""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from scripts.video_appearance_native_v108 import read, require
from scripts.video_appearance_native_v108 import write as save
from scripts.writing_agreement_revision_transfer import PRIOR_RESULT, PRIOR_RUN, SETTINGS
from scripts.writing_blind_rewrite_probe import EDIT_PROMPT, text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-reference-revision-transfer/1"
SEEDS = {"granite": 90461, "olmo": 79481}
METHOD = {
    "controls": "Before and after versions of the exact source excerpt are dated ordinary-writing controls. Use "
                "one generated replacement of the before center from each independently pinned model. Require a "
                "valid single output and preserve all flanking sentences.",
    "current": "First six positive later native windows in source order. Raw comparison point remains .5.",
    "agreement": "Retain a current positive if the second checkpoint is positive on the same native input.",
    "changed_agreement": "Retain agreement only when the exact model input differs between earlier and later.",
    "paired_direction": "Earlier and later complete native context must have the same exact flanking sentences. "
                        "Only mark the changed focus if both independent later AI-like odds are at least twice "
                        "the earlier odds. This is the same directional rule used for paired email research.",
    "combined": "Compare changed-agreement selection with or without an additional paired-direction focus.",
    "geometry": "Compare the actual app draft comparator's exact changed-token ranges as source evidence. An "
                "ordinary author revision is a genuine difference and never an AI attribution.",
    "outcomes": "Useful generated localization requires .30 non-whitespace character IoU and at least .50 recall. "
                "Count exact focus, wrong-only and missed separately. For genuine controls report group-level flags "
                "and distinguish exact unchanged native inputs from changed inputs and changed focus.",
    "advance": "For a statistical follow-up, preserve all useful current locations, recover at least two additional "
               "groups across both generation families and add at most one false-flag group to same-window "
               "agreement on the later dated human revisions. Before-versus-self is only a scope control. "
               "A single publisher remains development evidence. No serving confidence change.",
}
ARMS = ("current", "agreement", "changed_agreement", "paired", "combined")


def generation_lock(run):
    source = read(run / "source-receipt.json")
    originals = digest(run / "originals.json")
    require(source["originals_sha256"] == originals and read(run / "originals.json"), "Frozen publisher cases differ")
    model_protocol = PRIOR_RUN / "protocol.json"
    require(digest(model_protocol) == read(PRIOR_RESULT)["commitments"]["protocol.json"], "Generator provenance differs")
    archived = read(model_protocol)
    models = {family: archived["weights"][family] for family in SEEDS}
    for family, record in models.items():
        directory = PRIOR_RUN / "models" / family
        require(record["license"] == "Apache-2.0" and
                all(digest(directory / name) == sha for name, sha in record["assets"].items()),
                "A pinned generator differs")
    return {"schema": SCHEMA, "code_sha256": digest(Path(__file__)), "method": METHOD, "seeds": SEEDS,
            "source_receipt_sha256": digest(run / "source-receipt.json"), "originals_sha256": originals,
            "prompt": EDIT_PROMPT, "settings": SETTINGS, "runtime": archived["generation_runtime"], "models": models}


def generate(run, freeze, family):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    require({name: version(name) for name in freeze["runtime"]} == freeze["runtime"], "Generator runtime differs")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEEDS[family])
    started, results = monotonic(), {}
    record = freeze["models"][family]
    with TemporaryDirectory(prefix="revision-generator-", dir=run) as folder:
        target = Path(folder)
        for name in record["assets"]:
            source = PRIOR_RUN / "models" / family / name
            if name == "config.json" and record["typed_config_overrides"]:
                save(target / name, read(source) | record["typed_config_overrides"])
            else:
                (target / record["loader_aliases"].get(name, name)).symlink_to(source.resolve())
        model, tokenizer = load(str(target), tokenizer_config={"trust_remote_code": False})
    sampler = make_sampler(temp=SETTINGS["temperature"], top_p=SETTINGS["top_p"])
    for row in read(run / "originals.json"):
        original = row["before"]["focus_text"]
        count = len(words(original))
        prompt = EDIT_PROMPT.format(sentence=original, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))
        request = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokens = list(stream_generate(model, tokenizer, request, sampler=sampler, max_tokens=SETTINGS["max_tokens"]))
        finish = tokens[-1].finish_reason
        require(finish in {"stop", "length"}, "Unknown generator termination")
        results[row["group"]] = {"input_sha256": text_hash(original), "text": "".join(token.text for token in tokens),
                                 "eos": finish == "stop"}
        print(f"{family} attempted {len(results)}", flush=True)
    save(run / f"generated-{family}.json", {"generation_sha256": digest(run / "generation-freeze.json"),
                                           "results": results, "seconds": round(monotonic() - started, 3)})


def plans(run):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_revision import compare
    from scripts.writing_paired_email_revision import encoded, native_contexts

    originals = read(run / "originals.json")
    receipts = {family: read(run / f"generated-{family}.json") for family in SEEDS}
    groups = {row["group"] for row in originals}
    lock = digest(run / "generation-freeze.json")
    require(all(entry["generation_sha256"] == lock and entry["results"].keys() == groups for entry in receipts.values()),
            "Generator receipts have missing or unbound inputs")
    cases, texts, admission = [], {}, defaultdict(Counter)
    for row in originals:
        group, before, after = (row[key] for key in ("group", "before", "after"))
        source = before["focus_text"]
        start, stop = before["focus"]
        width = len(segment(from_string(before["text"])))
        require(width in (5, 7), "Unexpected frozen excerpt geometry")
        target = width // 2
        variants = {"before": before, "edited": after}
        for family, receipt in receipts.items():
            attempt = receipt["results"][group]
            require(attempt["input_sha256"] == text_hash(source), "Generator targeted another input")
            revised = valid_sentence(attempt, source, generated=True)
            if revised:
                text = before["text"][:start] + revised + before["text"][stop:]
                if len(segment(from_string(text))) == width:
                    variants[family] = {"text": text, "focus": [start, start + len(revised)], "focus_text": revised}
            admission[family]["attempted"] += 1
            admission[family]["admitted"] += int(family in variants)
        prior_units, reference = None, None
        for role, variant in variants.items():
            text = variant["text"]
            doc = from_string(text)
            units = segment(doc)
            focal = units[target]
            require(doc.text == text and len(units) == width and [focal.start, focal.end] == variant["focus"]
                    and focal.text == variant["focus_text"], "Selected text or focus geometry differs")
            flanks = [unit.text for unit in (*units[:target], *units[target + 1:])]
            require(prior_units is None or prior_units == flanks, "A non-target sentence changed")
            prior_units = flanks
            windows = []
            for context in native_contexts(units):
                record, model_input = encoded(context)
                windows.append(record)
                texts[record["sha256"]] = model_input
            current = [window["sha256"] for window in windows]
            reference = current if reference is None else reference
            changes = compare(before["text"].encode(), text.encode(), ".txt")["changes"]
            require(all(change["role"] == "body" for change in changes), "Unexpected draft role")
            changes = [change["later"] for change in changes if change["later"]["start"] < change["later"]["end"]]
            spans = [[change["start"], change["end"]] for change in changes]
            require((role == "before") == (current == reference) and (role != "before" or not spans),
                    "Reference text or native scope differs")
            cases.append({"group": group, "role": role, "text": text, "text_sha256": text_hash(text),
                          "windows": windows, "reference": reference, "target": target, "focus": variant["focus"],
                          "diff": spans})
    return cases, texts, admission


def scores_lock(run, current, second):
    from scripts.writing_paired_email_revision import SECONDARY

    assets = {"current": BASELINE_ASSETS, "second": SECONDARY}
    for path, model in ((current, "current"), (second, "second")):
        require(path is not None and all(digest(path / name) == expected for name, expected in assets[model].items()),
                "A pinned detector differs")
    return {"schema": SCHEMA, "code_sha256": digest(Path(__file__)), "method": METHOD, "checkpoints": assets,
            "generation_sha256": digest(run / "generation-freeze.json"), "case_sha256": digest(run / "cases.json"),
            "generation_receipts": {family: digest(run / f"generated-{family}.json") for family in SEEDS}}


def assess(row, first, second):
    from scripts.writing_paired_email_revision import rise

    native = [window["sha256"] for window in row["windows"]]
    reference = row["reference"]
    target = row["target"]
    a, b = ([result[key]["score"] for key in native] for result in (first, second))
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in a + b), "Invalid score")
    baseline = [i for i, value in enumerate(a) if value >= .5][:6]
    agreement = [i for i in baseline if b[i] >= .5]
    unchanged = [i for i in agreement if native[i] == reference[i]]
    changed = [i for i in agreement if native[i] != reference[i]]
    matched = native[target] != reference[target]
    paired = matched and all(rise(result[reference[target]]["score"], result[native[target]]["score"]) >= math.log(2)
                             for result in (first, second))
    windows = row["windows"]
    exact = [row["focus"]]
    regions = {key: [windows[i]["spans"] for i in indices] for key, indices in (
        ("current", baseline), ("agreement", agreement), ("changed_agreement", changed),
    )}
    regions["paired"] = [exact] if paired else []
    regions["combined"] = regions["changed_agreement"] + regions["paired"]

    def pixels(spans):
        return {i for a, b in spans for i in range(a, b) if not row["text"][i].isspace()}

    truth = pixels(exact)

    def useful(regions):
        return any(10 * len(p & truth) >= 3 * len(p | truth) and 2 * len(p & truth) >= len(truth)
                   for p in map(pixels, regions))

    generated = row["role"] in SEEDS
    out = {"windows_with_unchanged_agreement": len(unchanged), "windows_with_changed_agreement": len(changed),
           "unchanged_agreement_file": bool(unchanged), "changed_agreement_file": bool(changed),
           "both_agreement_types": bool(unchanged and changed), "agreement_at_focus": target in agreement,
           "diff_located_focus": bool(row["diff"]) and all(pixels([span]) <= truth for span in row["diff"]),
           "reference_comparable_focus": matched}
    for arm, selected in regions.items():
        focus = (paired if arm == "paired" else paired or target in changed if arm == "combined" else
                 target in (baseline if arm == "current" else agreement if arm == "agreement" else changed))
        hit = generated and useful(selected)
        out.update({f"{arm}_flag": bool(selected), f"{arm}_useful": hit, f"{arm}_focus": bool(focus),
                    f"{arm}_wrong_only": generated and bool(selected) and not hit,
                    f"{arm}_no_flag": not selected})
    return out


def report(run, lock, cases, inputs, admission):
    bindings = {"freeze_sha256": digest(run / "scoring-freeze.json"), "case_sha256": lock["case_sha256"]}
    receipts = [read(run / f"{arm}.json") for arm in ("current", "second")]
    require(all(r["responses"].keys() == inputs.keys() and all(r[key] == value for key, value in bindings.items())
                for r in receipts), "Score receipts differ")
    first, second = (r["responses"] for r in receipts)
    counts, groups, readout = defaultdict(Counter), defaultdict(set), []
    for row in cases:
        result = assess(row, first, second)
        counts[row["role"]].update({"cases": 1, **result})
        for metric, value in result.items():
            if value:
                groups[row["role"], metric].add(row["group"])
        readout.append({"group": row["group"], "role": row["role"], "text_sha256": row["text_sha256"], **result})
    statistics = {}
    for arm in ARMS[2:]:
        losses = [row for row in readout if row["current_useful"] and not row[f"{arm}_useful"]]
        gains = [row for row in readout if row[f"{arm}_useful"] and not row["current_useful"]]
        genuine = {row["group"] for row in readout if row["role"] == "edited" and row[f"{arm}_flag"]}
        false_agreement = {row["group"] for row in readout if row["role"] == "edited" and row["agreement_flag"]}
        statistics[arm] = {"lost_current_useful": len(losses), "added_useful_groups": len({r["group"] for r in gains}),
                           "added_useful_families": len({r["role"] for r in gains}),
                           "added_false_groups": len(genuine - false_agreement)}
        statistics[arm]["follow_up"] = (not losses and statistics[arm]["added_useful_groups"] >= 2
                                        and statistics[arm]["added_useful_families"] == 2
                                        and statistics[arm]["added_false_groups"] <= 1)
    path = run / "private-outcomes.json"
    save(path, readout)
    output = {"schema": SCHEMA, "method": METHOD, **bindings, "outcomes_sha256": digest(path),
              "source_receipt_sha256": digest(run / "source-receipt.json"), "admission": admission,
              "counts": counts, "group_counts": {f"{role}/{key}": len(items) for (role, key), items in sorted(groups.items())},
              "follow_up": statistics, "distinct_inputs": len(inputs),
              "classifier_forwards": {arm: r["forwards"] for arm, r in zip(("current", "second"), receipts, strict=True)},
              "score_sha256": {arm: digest(run / f"{arm}.json") for arm in ("current", "second")}}
    save(run / "aggregate.json", output)
    print(statistics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze-generation", "generate", "freeze-scoring", "current", "second", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--family", choices=SEEDS)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with local model access only")
    run, phase = args.run, args.phase
    generation = generation_lock(run)
    freeze = run / "generation-freeze.json"
    if phase == "freeze-generation":
        save(freeze, generation)
        return
    require(read(freeze) == generation, "Generation freeze changed")
    if phase == "generate":
        require(args.family in SEEDS, "Select one frozen family")
        generate(run, generation, args.family)
        return
    cases, values, admission = plans(run)
    cases_path, score_freeze = run / "cases.json", run / "scoring-freeze.json"
    if phase == "freeze-scoring":
        save(cases_path, cases)
        save(score_freeze, scores_lock(run, args.current, args.second))
        print(f"{len(cases)} cases; {len(values)} exact inputs", flush=True)
        return
    lock = scores_lock(run, args.current, args.second)
    require(read(cases_path) == cases and read(score_freeze) == lock, "Scoring freeze changed")
    if phase in ("current", "second"):
        import torch

        from scripts.writing_paired_email_revision import native_scores

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        started = monotonic()
        responses, forwards = native_scores(getattr(args, phase), values, second=phase == "second")
        save(run / f"{phase}.json", {"freeze_sha256": digest(score_freeze), "case_sha256": lock["case_sha256"],
                                   "responses": responses, "forwards": forwards, "seconds": round(monotonic() - started, 3)})
        return
    report(run, lock, cases, values, admission)


if __name__ == "__main__":
    main()
