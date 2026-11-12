"""Extend the completed revision comparison to a separately frozen third generator."""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from scripts.video_appearance_native_v108 import read, require
from scripts.video_appearance_native_v108 import write as save
from scripts.writing_agreement_revision_transfer import PRIOR_RESULT, PRIOR_RUN, SETTINGS, evaluate, inputs
from scripts.writing_blind_rewrite_probe import EDIT_PROMPT, text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-agreement-revision-olmo/1"
SEED = 280061
METHOD = "Add one frozen OLMo rewrite attempt per original revision target after the initial TinyLlama failures. " \
         "Retain the completed initial study and every target. Use the same generation admission and exact-window comparison. " \
         "This measures another family on the same known sources and supplies no fresh publisher replication."


def freeze(parent):
    initial = read(parent / "aggregate.json")
    protocol = PRIOR_RUN / "protocol.json"
    published = read(PRIOR_RESULT)
    require(digest(protocol) == published["commitments"]["protocol.json"], "Prior generator identity differs")
    retained = read(protocol)
    record = retained["weights"]["olmo"]
    require(initial["admission"]["tiny"]["admitted"] == 0 and record["license"] == "Apache-2.0"
            and all(digest(PRIOR_RUN / "models/olmo" / name) == value for name, value in record["assets"].items()),
            "Initial admission or new pinned generator differs")
    return {"schema": SCHEMA, "method": METHOD, "model": record, "runtime": retained["generation_runtime"],
            "seed": SEED, "settings": SETTINGS, "prompt": EDIT_PROMPT, "code_sha256": digest(Path(__file__)),
            "initial_result_sha256": digest(parent / "aggregate.json"), "originals_sha256": digest(parent / "originals.json")}


def generate(parent, run, lock):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    require({name: version(name) for name in lock["runtime"]} == lock["runtime"], "Generator runtime differs")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)
    record = lock["model"]
    with TemporaryDirectory(prefix="revision-olmo-", dir=run) as folder:
        staging = Path(folder)
        for name in record["assets"]:
            source = PRIOR_RUN / "models/olmo" / name
            if name == "config.json" and record["typed_config_overrides"]:
                save(staging / name, {**read(source), **record["typed_config_overrides"]})
            else:
                (staging / record["loader_aliases"].get(name, name)).symlink_to(source.resolve())
        model, tokenizer = load(str(staging), tokenizer_config={"trust_remote_code": False})
    sample = make_sampler(temp=SETTINGS["temperature"], top_p=SETTINGS["top_p"])
    generated = {}
    for row in read(parent / "originals.json"):
        source = row["before"][row["target"]]
        count = len(words(source))
        prompt = EDIT_PROMPT.format(sentence=source, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))
        request = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        pieces = list(stream_generate(model, tokenizer, request, sampler=sample, max_tokens=SETTINGS["max_tokens"]))
        require(pieces[-1].finish_reason in {"stop", "length"}, "Generator did not terminate normally")
        generated[row["group"]] = {"text": "".join(item.text for item in pieces),
                                    "eos": pieces[-1].finish_reason == "stop", "input_sha256": text_hash(source)}
    save(run / "generated-olmo.json", {"freeze_sha256": digest(run / "freeze.json"), "results": generated})


def admit(parent, run):
    from scripts.writing_agreement_revision_sources import usable

    response = read(run / "generated-olmo.json")
    require(response["freeze_sha256"] == digest(run / "freeze.json"), "Generator did not use the frozen input")
    originals = read(parent / "originals.json")
    require({row["group"] for row in originals} == set(response["results"]), "Generator did not attempt every target")
    cases = []
    for row in originals:
        target, before = row["target"], row["before"]
        value = response["results"][row["group"]]
        require(value["input_sha256"] == text_hash(before[target]), "Generator source sentence differs")
        replacement = valid_sentence(value, before[target], generated=True)
        if not replacement:
            continue
        parts = [*before[:target], replacement, *before[target + 1:]]
        if not usable(parts):
            continue
        text = " ".join(parts)
        start = sum(len(part) + 1 for part in parts[:target])
        cases.append({"group": row["group"], "publisher": row["publisher"], "role": "olmo", "text": text,
                      "text_sha256": text_hash(text), "focus": [start, start + len(replacement)], "target": target})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "generate", "admit", "current", "second", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--second", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with local model access only")
    run, parent = args.run, args.run.parent
    lock = freeze(parent)
    frozen = run / "freeze.json"
    if args.stage == "freeze":
        save(frozen, lock)
        return
    require(read(frozen) == lock, "Supplemental freeze differs")
    if args.stage == "generate":
        generate(parent, run, lock)
        return
    cases = admit(parent, run)
    if args.stage == "admit":
        save(run / "cases.json", {"cases": cases})
        print(f"{len(cases)} admitted", flush=True)
        return
    from scripts.writing_paired_email_revision import SECONDARY, native_scores

    require(read(run / "cases.json") == {"cases": cases} and cases, "No frozen admitted supplemental inputs")
    require(all(digest(args.current / name) == value for name, value in BASELINE_ASSETS.items())
            and all(digest(args.second / name) == value for name, value in SECONDARY.items()), "Classifier files differ")
    rows, values = inputs(run)
    bindings = {"freeze_sha256": digest(frozen), "cases_sha256": digest(run / "cases.json"),
                "generated_sha256": digest(run / "generated-olmo.json")}
    if args.stage in ("current", "second"):
        import torch

        torch.set_num_threads(1)
        arm, start = args.stage, monotonic()
        scores, forwards = native_scores(getattr(args, arm), values, second=arm == "second")
        save(run / f"{arm}.json", {**bindings, "responses": scores, "forwards": forwards,
                                  "seconds": round(monotonic() - start, 3)})
        return
    packets = [read(run / f"{arm}.json") for arm in ("current", "second")]
    require(all(set(packet["responses"]) == set(values) and all(packet[key] == value for key, value in bindings.items())
                for packet in packets), "A response set differs from the frozen inputs")
    evidence = [{key: row[key] for key in ("group", "publisher", "role", "text_sha256", "windows", "focus", "target")}
                | evaluate(row, *(packet["responses"] for packet in packets)) for row in rows]
    save(run / "private-readout.json", evidence)
    outcomes = {}
    for arm in ("current", "second", "agreement"):
        marked = sum(bool(item["selected"][arm]) for item in evidence)
        hits = sum(item["useful"][arm] for item in evidence)
        outcomes[arm] = {"flagged": marked, "useful": hits, "wrong_location": marked - hits,
                         "unflagged": len(evidence) - marked,
                         "exact_focus": sum(item["exact_focus"][arm] for item in evidence),
                         "lost_useful": sum(item["useful"]["current"] and not item["useful"][arm] for item in evidence)}
    save(run / "aggregate.json", {**lock, **bindings, "planned": len(read(parent / "originals.json")), "admitted": len(rows),
                                  "unique_inputs": len(values), "outcomes": outcomes,
                                  "readout_sha256": digest(run / "private-readout.json"),
                                  "score_receipts": {arm: {"sha256": digest(run / f"{arm}.json"),
                                                          "forwards": packet["forwards"], "seconds": packet["seconds"]}
                                                     for arm, packet in zip(("current", "second"), packets, strict=True)}})


if __name__ == "__main__":
    main()
