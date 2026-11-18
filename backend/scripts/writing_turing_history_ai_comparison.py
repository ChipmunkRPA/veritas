# SPDX-License-Identifier: Apache-2.0
"""Create and score one exact sentence rewrite alongside each editorial control."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range
from scripts.text_elife_rewrites_v273 import CONTACT, GENERATORS, WRAPPER
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_blind_rewrite_probe import distance, words
from scripts.writing_entropy_prominence import existing
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
from scripts.writing_multi_passage_panel import native, score
from scripts.writing_transition_deviation import file_hash, read, save, text_hash

SCHEMA = "chipmunk.writing-turing-history-ai-comparison/1"
PROMPT = ("Copyedit only the target sentence from a research handbook. Write one complete English sentence. "
          "Preserve the meaning and any numbers or proper names. Use different wording without adding a new claim. "
          "Output only the revised target sentence.")


def generate(source, family, output):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    _, revision, expected, folder = GENERATORS[family]
    if file_hash(folder / "model.safetensors") != expected:
        raise ValueError("The local editor weights do not match the frozen release")
    model, tokenizer = load(str(folder), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    mx.set_cache_limit(96 * 1024 * 1024)
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    works = read(source)["works"]

    def _prompt(work):
        before = work["variants"][0]["text"]
        start, end = work["witnesses"][0]["focus"]
        content = (f"{PROMPT}\n\nPrior context: {before[max(0, start - 380):start].strip()}\n"
                   f"Target: {before[start:end]}\nFollowing context: {before[end:end + 380].strip()}")
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True)
        return tokenizer.encode(prompt, add_special_tokens=False)

    replies = []
    for start in range(0, len(works), 4):
        batch = works[start:start + 4]
        completions = batch_generate(model, tokenizer, [_prompt(work) for work in batch], max_tokens=100,
                                     sampler=make_sampler(temp=0.0))
        replies.extend([work["document"], value] for work, value in zip(batch, completions.texts, strict=True))
    save(output, {"schema": SCHEMA, "family": family, "revision": revision, "weights_sha256": expected,
                  "source_sha256": file_hash(source), "prompt_sha256": text_hash(PROMPT), "replies": replies})


def prepare(source, output, transcripts):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    works = read(source)["works"]
    attempts, bindings = {}, {}
    for path in transcripts:
        record = read(path)
        family = record["family"]
        _, revision, expected, _ = GENERATORS[family]
        if (record["schema"] != SCHEMA or record["source_sha256"] != file_hash(source)
                or record["weights_sha256"] != expected or record["revision"] != revision
                or record["prompt_sha256"] != text_hash(PROMPT) or family in bindings
                or [key for key, _ in record["replies"]] != [work["document"] for work in works]):
            raise ValueError("A generation receipt differs from its historical source or editor")
        attempts[family], bindings[family] = dict(record["replies"]), file_hash(path)
    if attempts.keys() != GENERATORS.keys():
        raise ValueError("Keep the complete two-generator attempt ledger")
    variants, rejected = [], Counter()
    for work in works:
        case = {"document": work["document"], "split": "test", "variants": []}
        body = work["variants"][0]["text"]
        start, end = work["witnesses"][0]["focus"]
        for family, answers in sorted(attempts.items()):
            response = answers[work["document"]]
            revised = " ".join(response.split())
            ratio = len(words(revised)) / len(words(body[start:end]))
            span = (start, start + len(revised))
            if (revised and .60 <= ratio <= 1.55 and not WRAPPER.match(revised) and not CONTACT.search(revised)
                    and not re.search(r"```|\n\s*\n", response.strip()) and revised[-1] in ".?!"
                    and len(segment(from_string(revised))) == 1 and .05 < distance(body[start:end], revised) <= .70
                    and (row := native(body[:start] + revised + body[end:], [span]))
                    and len(row["sentences"]) == 12 and row["sentences"][work["witnesses"][0]["sentence"]][:2] == list(span)):
                case["variants"].append(row | {"version": family})
            else:
                rejected[family] += 1
        if case["variants"]:
            variants.append(case)
    save(output, {"schema": SCHEMA, "source_sha256": file_hash(source), "transcripts": bindings,
                  "implementation_sha256": file_hash(Path(__file__)), "baseline_assets": BASELINE_ASSETS,
                  "baseline_revision": BASELINE_REVISION, "works": variants, "rejected": dict(rejected)})
    print("Admitted", dict(Counter(row["version"] for work in variants for row in work["variants"])), "Rejected", dict(rejected))


def totals(works):
    groups = {name: Counter() for name in GENERATORS}
    for work in works:
        for row in work["variants"]:
            ranges, assessed = existing(row, row)
            regions = [compact_range(row["sentences"], span) for span in ranges]
            hit = matching(regions, [item["compact"] for item in row["truth"]])
            groups[row["version"]].update(delivered=1, assessed=assessed, located=hit, alerted=bool(regions),
                                         misplaced=len(regions) - hit)
    return {name: dict(group) for name, group in groups.items()}


def evaluate(plan, snapshot, output):
    sealed = read(plan)
    if sealed["implementation_sha256"] != file_hash(Path(__file__)) or sealed["baseline_assets"] != BASELINE_ASSETS:
        raise ValueError("The comparison differs from its frozen scoring plan")
    readings = output.with_name("machine-readings.json.gz")
    score({"works": sealed["works"]}, snapshot, readings)
    cases = read(readings)["works"]
    save(output, {"schema": SCHEMA, "plan_sha256": file_hash(plan), "source_sha256": sealed["source_sha256"],
                  "results": totals(cases), "rejected": sealed["rejected"], "works": cases})
    review(output)


def review(path):
    result = read(path)
    if result["schema"] != SCHEMA or any("text" in case for work in result["works"] for case in work["variants"]):
        raise ValueError("The portable response must omit publisher and generated prose")
    if totals(result["works"]) != result["results"]:
        raise ValueError("The scored positions and counts disagree")
    print(result["results"], result["rejected"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name, fields in (("generate", ("source", "output")), ("prepare", ("source", "output")),
                         ("evaluate", ("plan", "snapshot", "output")), ("review", ("path",))):
        cmd = actions.add_parser(name)
        for field in fields:
            cmd.add_argument(field, type=Path)
        if name == "generate":
            cmd.add_argument("--family", choices=tuple(GENERATORS), required=True)
        elif name == "prepare":
            cmd.add_argument("transcripts", type=Path, nargs=2)
    args = vars(parser.parse_args())
    {"generate": generate, "prepare": prepare, "evaluate": evaluate, "review": review}[args.pop("action")](**args)
