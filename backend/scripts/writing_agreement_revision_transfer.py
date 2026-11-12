"""Compare writing checkpoints on exact public editorial pairs and generated replacements."""

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
from scripts.writing_blind_rewrite_probe import EDIT_PROMPT, valid_sentence, words
from scripts.writing_blind_rewrite_probe import text_hash as sha
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-agreement-revision-transfer/1"
FAMILIES = {"tiny": 91243, "granite": 80963}
ROLES = ("before", "edited", *FAMILIES)
ARMS = ("current", "second", "agreement")
METHOD = {
    "human": "Use a pre-2020 revision pair with the same three surrounding sentences in the selected excerpt. "
             "Both versions are ordinary writing controls. The edited focus is the complete changed sentence.",
    "generation": "Replace only the original edited-position sentence. Attempt one rewrite per model. "
                  "Require completed output, unchanged 4-sentence scope, and a valid non-copy rewrite.",
    "current": "Use the first six source-ordered current responses at or above .5.",
    "second": "Use the first six source-ordered second responses at or above .5.",
    "agreement": "Keep a current positive only if the second response on exactly the same native input is at or above .5.",
    "useful": "Use .30 non-whitespace character IoU between a selected full context and the exact replaced sentence. "
              "Also count exact focus selection separately, and distinguish wrong locations from unflagged positives.",
    "validation": "No fitted point. Report both versions and paired flips separately. A default-policy transfer would "
                  "need both generator families, no loss of useful positives and cleared human source groups in at "
                  "least two publisher communities.",
}
SETTINGS = {"temperature": .60, "top_p": .90, "max_tokens": 96}
ROOT = Path(__file__).resolve().parents[2]
PRIOR_RESULT = ROOT / "docs/evaluation-panels/writing-grammar-seams-2026-11-11.json"
PRIOR_RUN = ROOT / "backend/data/independent-evaluation/writing-grammar-seams-2026-11-11"


def generator_identity():
    protocol = PRIOR_RUN / "protocol.json"
    previous = read(protocol)
    published = read(PRIOR_RESULT)
    require(digest(protocol) == published["commitments"]["protocol.json"], "Generator archive does not match prior public receipt")
    for family in FAMILIES:
        row = previous["weights"][family]
        path = PRIOR_RUN / "models" / family
        require(row["license"] == "Apache-2.0" and all(digest(path / name) == expected
                for name, expected in row["assets"].items()), "A pinned generator file differs")
    return {"weights": {family: previous["weights"][family] for family in FAMILIES},
            "runtime": previous["generation_runtime"], "prior_result_sha256": digest(PRIOR_RESULT)}


def lock_generation(run):
    receipt = read(run / "source-receipt.json")
    require(receipt["originals_sha256"] == digest(run / "originals.json"), "Original revision panel differs")
    return {"schema": SCHEMA, "method": METHOD, "prompt": EDIT_PROMPT, "generation": SETTINGS, "seeds": FAMILIES,
            "generators": generator_identity(), "source_receipt_sha256": digest(run / "source-receipt.json"),
            "originals_sha256": receipt["originals_sha256"], "code_sha256": digest(Path(__file__))}


def verify_generation(run):
    frozen = read(run / "generation-freeze.json")
    require(frozen == lock_generation(run), "Generation freeze differs")
    return frozen


def generate(run, family):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    frozen = verify_generation(run)
    runtime = frozen["generators"]["runtime"]
    require({name: version(name) for name in runtime} == runtime, "Generator runtime differs")
    model_files = frozen["generators"]["weights"][family]
    mx.set_default_device(mx.gpu)
    mx.random.seed(FAMILIES[family])
    started = monotonic()
    with TemporaryDirectory(prefix="writing-revision-model-", dir=run) as folder:
        target = Path(folder)
        original = PRIOR_RUN / "models" / family
        for name in model_files["assets"]:
            overrides = model_files["typed_config_overrides"]
            if name == "config.json" and overrides:
                save(target / name, {**read(original / name), **overrides})
            else:
                (target / model_files["loader_aliases"].get(name, name)).symlink_to((original / name).resolve())
        model, tokenizer = load(str(target), tokenizer_config={"trust_remote_code": False})
    sampler = make_sampler(temp=SETTINGS["temperature"], top_p=SETTINGS["top_p"])
    results = {}
    for row in read(run / "originals.json"):
        text = row["before"][row["target"]]
        count = len(words(text))
        prompt = EDIT_PROMPT.format(sentence=text, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        reply = list(stream_generate(model, tokenizer, chat, max_tokens=SETTINGS["max_tokens"], sampler=sampler))
        finish = reply[-1].finish_reason
        require(finish in {"stop", "length"}, "Unknown generation termination")
        results[row["group"]] = {"text": "".join(part.text for part in reply), "eos": finish == "stop",
                                  "input_sha256": sha(text)}
        print(f"{family}: {len(results)}", flush=True)
    save(run / f"generated-{family}.json", {"generation_sha256": digest(run / "generation-freeze.json"),
                                           "seconds": round(monotonic() - started, 3), "results": results})


def derive_cases(run):
    from scripts.writing_agreement_revision_sources import usable

    source = read(run / "originals.json")
    generated = {family: read(run / f"generated-{family}.json") for family in FAMILIES}
    require(all(packet["generation_sha256"] == digest(run / "generation-freeze.json") for packet in generated.values()),
            "Generator source differs")
    candidates, outcomes = [], Counter()
    for row in source:
        target, before, after = row["target"], row["before"], row["after"]
        original = before[target]
        variants = {"before": before, "edited": after}
        require(before[:target] == after[:target] and before[target + 1:] == after[target + 1:]
                and before[target] != after[target], "Source edit no longer has exact common flanks")
        for family, packet in generated.items():
            item = packet["results"][row["group"]]
            require(item["input_sha256"] == sha(original), "A rewrite received a different sentence")
            replacement = valid_sentence(item, original, generated=True)
            if replacement:
                updated = [*before[:target], replacement, *before[target + 1:]]
                if usable(updated):
                    variants[family] = updated
            outcomes[family, "admitted" if family in variants else "rejected"] += 1
        for role, parts in variants.items():
            text = " ".join(parts)
            start = sum(len(part) + 1 for part in parts[:target])
            candidates.append({"group": row["group"], "publisher": row["publisher"], "role": role, "text": text,
                               "text_sha256": sha(text), "focus": [start, start + len(parts[target])], "target": target})
    return candidates, {family: {state: outcomes[family, state] for state in ("admitted", "rejected")} for family in FAMILIES}


def lock_scoring(run, current, second):
    from scripts.writing_paired_email_revision import SECONDARY

    require(all(digest(current / name) == value for name, value in BASELINE_ASSETS.items())
            and all(digest(second / name) == value for name, value in SECONDARY.items()), "Classifier files differ")
    return {"schema": SCHEMA, "method": METHOD, "code_sha256": digest(Path(__file__)),
            "generation_sha256": digest(run / "generation-freeze.json"), "cases_sha256": digest(run / "cases.json"),
            "generated_sha256": {family: digest(run / f"generated-{family}.json") for family in FAMILIES},
            "checkpoints": {"current": BASELINE_ASSETS, "second": SECONDARY}}


def inputs(run):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_paired_email_revision import encoded, native_contexts

    plan = read(run / "cases.json")
    rows, texts = [], {}
    for case in plan["cases"]:
        text = case["text"]
        doc = from_string(text)
        sentences = segment(doc)
        require(sha(text) == case["text_sha256"] and text == doc.text and len(sentences) == 4
                and [sentences[case["target"]].start, sentences[case["target"]].end] == case["focus"],
                "Sentence fixture or source position differs")
        windows = []
        for nearby in native_contexts(sentences):
            record, value = encoded(nearby)
            windows.append(record)
            texts[record["sha256"]] = value
        rows.append({**case, "windows": windows})
    return rows, texts


def evaluate(row, current, second):
    from scripts.writing_checkpoint_agreement import useful

    model_scores = {key: [scores[entry["sha256"]]["score"] for entry in row["windows"]]
                    for key, scores in (("current", current), ("second", second))}
    require(all(math.isfinite(score) and 0 <= score <= 1 for values in model_scores.values() for score in values),
            "A model response is invalid")
    marked = {key: [i for i, value in enumerate(scores) if value >= .5][:6] for key, scores in model_scores.items()}
    marked["agreement"] = [i for i in marked["current"] if i in marked["second"]]
    return {
        "selected": marked,
        "useful": {key: any(useful(row["text"], row["focus"], row["windows"][i]["spans"]) for i in marks)
                   for key, marks in marked.items()},
        "exact_focus": {key: row["target"] in marks for key, marks in marked.items()},
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[row["publisher"]].append(row)
    report = {}
    for scope, cases in sorted(grouped.items()):
        by_role = {role: [row for row in cases if row["role"] == role] for role in ROLES}
        by_group = defaultdict(dict)
        for row in cases:
            by_group[row["group"]][row["role"]] = row
        arm_records = {}
        for arm in ARMS:
            roles = {}
            for role, group in by_role.items():
                flagged = sum(bool(row["selected"][arm]) for row in group)
                hits = sum(row["useful"][arm] for row in group)
                record = {"files": len(group), "flagged": flagged}
                if role in FAMILIES:
                    record.update(useful=hits, wrong_location=flagged - hits, unflagged=len(group) - flagged,
                                  exact_focus=sum(row["exact_focus"][arm] for row in group),
                                  lost_useful=sum(row["useful"]["current"] and not row["useful"][arm] for row in group))
                elif role == "edited":
                    record["human_edit_overlapped"] = hits
                roles[role] = record
            flips = Counter((bool(pair["before"]["selected"][arm]), bool(pair["edited"]["selected"][arm]))
                            for pair in by_group.values())
            arm_records[arm] = {"roles": roles, "human_source_groups_flagged": len(by_group) - flips[False, False],
                                "human_before_to_after": {"no_flag_either": flips[False, False], "added": flips[False, True],
                                                          "removed": flips[True, False], "flagged_both": flips[True, True]}}
        report[scope] = arm_records
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-generation", "generate", "freeze-scoring", "score", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--arm", choices=("current", "second"))
    parser.add_argument("--current", type=Path)
    parser.add_argument("--second", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with local model access only")
    run = args.run
    if args.stage == "freeze-generation":
        save(run / "generation-freeze.json", lock_generation(run))
        return
    if args.stage == "generate":
        require(args.family is not None, "Select a frozen generator family")
        generate(run, args.family)
        return
    require(args.current and args.second, "Supply both pinned classifier snapshots")
    from scripts.writing_paired_email_revision import SECONDARY, native_scores

    if args.stage == "freeze-scoring":
        verify_generation(run)
        variants, admission = derive_cases(run)
        save(run / "cases.json", {"cases": variants, "admission": admission})
        save(run / "scoring-freeze.json", lock_scoring(run, args.current, args.second))
        return
    require(read(run / "scoring-freeze.json") == lock_scoring(run, args.current, args.second), "Scoring freeze differs")
    freeze_hash = digest(run / "scoring-freeze.json")
    rows, texts = inputs(run)
    if args.stage == "score":
        import torch

        require(args.arm is not None, "Select the current or second checkpoint")
        torch.set_num_threads(1)
        started = monotonic()
        snapshot = args.second if args.arm == "second" else args.current
        scores, forwards = native_scores(snapshot, texts, second=args.arm == "second")
        save(run / f"{args.arm}.json", {"freeze_sha256": freeze_hash, "responses": scores, "forwards": forwards,
                                       "seconds": round(monotonic() - started, 3)})
        return
    paths = {arm: run / f"{arm}.json" for arm in ("current", "second")}
    packets = {key: read(path) for key, path in paths.items()}
    require(all(item["freeze_sha256"] == freeze_hash and set(item["responses"]) == set(texts)
                for item in packets.values()), "Scored input commitments differ")
    evidence = [{key: row[key] for key in ("group", "publisher", "role", "text_sha256", "windows", "focus", "target")}
                | evaluate(row, *[item["responses"] for item in packets.values()]) for row in rows]
    save(run / "private-readout.json", evidence)
    save(run / "aggregate.json", {"schema": SCHEMA, "method": METHOD, "sources": read(run / "source-receipt.json"),
                                 "generators": read(run / "generation-freeze.json")["generators"],
                                 "generation_sha256": digest(run / "generation-freeze.json"),
                                 "checkpoints": {"current": BASELINE_ASSETS, "second": SECONDARY},
                                 "scoring_sha256": freeze_hash, "outcomes": summarize(evidence),
                                 "admission": read(run / "cases.json")["admission"], "files": len(rows),
                                 "windows": sum(len(row["windows"]) for row in rows), "unique_inputs": len(texts),
                                 "score_receipts": {key: {k: item[k] for k in ("forwards", "seconds")}
                                                    | {"sha256": digest(paths[key])} for key, item in packets.items()},
                                 "readout_sha256": digest(run / "private-readout.json")})


if __name__ == "__main__":
    main()
