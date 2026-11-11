"""Construct one offline, source-aligned paragraph rewrite per frozen generator and article."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from types import FunctionType, SimpleNamespace

from scripts.video_appearance_native_v108 import BASE, PANEL, ROOT, read, require, sha, write

NAME = "text-elife-rewrites-v273"
MODULE = "scripts.text_elife_rewrites_v273"
TRIAL = BASE / NAME
FREEZE = PANEL / "text-elife-rewrites-freeze-v273.json"
INTAKE = PANEL / "text-elife-source-intake-complete-v272.json"
INTAKE_SHA = "e494c33e29a893abc16a1845f028bedcf9a65b5a8ce9f963323c267c006627a2"
SOURCE = BASE / "text-elife-source-intake-v272"
GROUPS = ("elife-81108", "elife-80609", "elife-80718", "elife-80699", "elife-80392")
UPGRADE = ROOT / "backend/data/text-upgrade"
MLX_PYTHON = UPGRADE / "mlx-env/bin/python"
MLX_VERSIONS = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.22.1"}
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SMOL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
GENERATORS = {
    "qwen": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        QWEN_REVISION,
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        ROOT / "models/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" / QWEN_REVISION,
    ),
    "smol2": (
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        SMOL_REVISION,
        "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
        UPGRADE / "models/smol2",
    ),
}
ASSETS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)
WRAPPER = re.compile(
    r"^\s*(?:[#>*`]|[-+]\s|\d+[.)]\s|sure\b|here(?:'s| is)\b|"
    r"(?:revised|rewritten|replacement|target|original)\s+paragraph\s*:|i (?:can(?:not|'t)|am unable)\b)",
    re.I,
)
CONTACT = re.compile(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,})", re.I)
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def text_sha(value):
    return sha256(value.encode()).hexdigest()


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def targets(receipt, rows):
    admitted = [work for work in receipt["works"] if work["status"] == "admitted"]
    require(
        receipt["schema"] == "chipmunk.text-elife-source-intake/272"
        and receipt["status"] == "complete_intake"
        and receipt["planned_works"] == 9
        and receipt["admitted_works"] == len(GROUPS)
        and tuple(work["group"] for work in admitted) == tuple(row["group"] for row in rows) == GROUPS,
        "Exact five admitted v272 groups required",
    )
    for work, row in zip(admitted, rows, strict=True):
        require(
            {key: value for key, value in row.items() if key != "text"}
            == {key: work[key] for key in row if key != "text"},
            "Private row differs from v272 receipt",
        )
        raw, edges, target = (row[key] for key in ("text", "paragraph_codepoints", "edit_target_codepoints"))
        require(
            text_sha(raw) == row["text_sha256"] == work["text_sha256"]
            and target == work["edit_target_codepoints"]
            and target in edges[1:-1]
            and edges == work["paragraph_codepoints"]
            and edges[0][0] == 0
            and edges[-1][1] == len(raw)
            and all(0 <= a < b <= len(raw) for a, b in edges)
            and all(raw[b:c] == "\n\n" for (_, b), (c, _) in zip(edges, edges[1:]))
            and raw == "\n\n".join(raw[a:b] for a, b in edges),
            "Exact source paragraph and interior target coordinates required",
        )
    return rows


def inputs():
    require(sha(INTAKE) == INTAKE_SHA, "Exact published v272 source receipt required")
    receipt = read(INTAKE)
    originals = SOURCE / "private-originals.jsonl"
    require(
        receipt == read(SOURCE / "aggregate.json") and receipt["private_originals_sha256"] == sha(originals),
        "Exact owner-local v272 intake and originals required",
    )
    return targets(receipt, records(originals))


def identity():
    rows = inputs()
    require(
        all((path / asset).is_file() for *_, path in GENERATORS.values() for asset in ASSETS[:-1]),
        "Existing tokenizer and generator configuration files required",
    )
    paths = (
        Path(__file__),
        ROOT / "backend/scripts/generate_email_upgrade.py",
        ROOT / "backend/scripts/video_appearance_native_v108.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        ROOT / "backend/scripts/video_appearance_corrected_v112.py",
        ROOT / "backend/scripts/audio_tree_sampler_v64.py",
        ROOT / "backend/scripts/grid_source_speech_v147.py",
        ROOT / "backend/scripts/local_probe_limits_v24.py",
        ROOT / f"docs/{NAME}-method.md",
        ROOT / "backend/tests/test_text_elife_rewrites_v273.py",
    )
    return {
        "schema": "chipmunk.text-elife-rewrites-freeze/273",
        "v272_receipt_sha256": INTAKE_SHA,
        "originals_sha256": sha(SOURCE / "private-originals.jsonl"),
        "groups": [row["group"] for row in rows],
        "expected_generated_attempts": len(rows) * len(GENERATORS),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "generators": {
            family: {
                "repo": repo,
                "revision": revision,
                "weights_sha256": digest,
                "license": "Apache-2.0",
                "local_config_sha256": {name: sha(path / name) for name in ASSETS if (path / name).is_file()},
            }
            for family, (repo, revision, digest, path) in GENERATORS.items()
        },
        "runtime": {
            "versions": MLX_VERSIONS,
            "device": "mps",
            "precision": "mlx-affine-q4-g64",
            "seed": 56174,
            "temperature": 0.65,
            "top_p": 0.9,
            "max_new_tokens": 340,
            "batch_size": 1,
        },
    }


def freeze():
    require(not TRIAL.exists() and not FREEZE.exists(), "Preserve earlier attempts")
    locked = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(FREEZE, locked)
    return {"status": "frozen", "planned_rewrites": locked["expected_generated_attempts"]}


def verify():
    require(
        read(FREEZE) == read(TRIAL / "freeze.json") == identity(), "Exact v273 generation freeze required"
    )
    return inputs()


def prompt(tokenizer, row):
    raw = row["text"]
    a, b = row["edit_target_codepoints"]
    words = len(raw[a:b].split())
    paragraphs = raw.split("\n\n")
    target_index = row["paragraph_codepoints"].index([a, b])
    message = (
        "Rewrite only the target paragraph in English for a general science audience. Preserve the facts, "
        "the meaning and the connection to the adjacent paragraphs. Use your own wording. Return one plain "
        f"paragraph of {math.ceil(words * 0.75)}–{math.floor(words * 1.25)} words. "
        "Do not return the adjacent paragraphs, a heading, a note, a quotation, links or a list.\n\n"
        f"<previous>\n{paragraphs[target_index - 1]}\n</previous>\n"
        f"<target>\n{raw[a:b]}\n</target>\n"
        f"<next>\n{paragraphs[target_index + 1]}\n</next>"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )


def copy_fraction(text, source):
    tokens = TOKEN.findall(text.casefold())
    originals = TOKEN.findall(source.casefold())
    width = 8
    source_phrases = {tuple(originals[i : i + width]) for i in range(len(originals) - width + 1)}
    copied = set()
    for i in range(len(tokens) - width + 1):
        if tuple(tokens[i : i + width]) in source_phrases:
            copied.update(range(i, i + width))
    return len(copied) / len(tokens) if tokens else 0.0


def rewrite(row, family, raw):
    a, b = row["edit_target_codepoints"]
    source = row["text"]
    replacement = " ".join(raw.split())
    words, target_words = len(replacement.split()), len(source[a:b].split())
    reason = None
    if not replacement:
        reason = "empty_response"
    elif WRAPPER.match(raw) or re.search(r"\n\s*\n|</?(?:previous|target|next)>|```", raw.strip(), re.I):
        reason = "wrapper_or_multiple_paragraphs"
    elif CONTACT.search(replacement):
        reason = "link_or_contact"
    elif not max(20, math.ceil(0.60 * target_words)) <= words <= min(170, math.floor(1.50 * target_words)):
        reason = "outside_word_bound"
    elif replacement.casefold() == source[a:b].casefold():
        reason = "unchanged"
    elif copy_fraction(replacement, source) > 0.25:
        reason = "source_copy"
    mixed = source[:a] + replacement + source[b:] if reason is None else None
    interval = [a, a + len(replacement)] if mixed is not None else None
    if mixed is not None:
        end = interval[1]
        require(
            mixed[:a] == source[:a]
            and mixed[end:] == source[b:]
            and mixed[a:end] == replacement
            and mixed[:a] + source[a:b] + mixed[end:] == source,
            "Exact splice correspondence required",
        )
    return {
        "case": f"{row['group']}-{family}",
        "article_group": row["group"],
        "generator": family,
        "arm": "paragraph_ai",
        "status": "constructed" if reason is None else "unavailable",
        "reason": reason,
        "source_sha256": row["text_sha256"],
        "source_target": [a, b],
        "raw_output_sha256": text_sha(raw),
        "replacement_words": words,
        "text": mixed,
        "text_sha256": text_sha(mixed) if mixed is not None else None,
        "ai_spans": [interval] if interval is not None else None,
    }


def worker(batch, output):
    rows = verify()
    family = tuple(GENERATORS)[batch]
    folder = TRIAL / f"extract-{batch:02d}"
    require(
        output == folder / "output"
        and (folder / "resources.jsonl").is_file()
        and (folder / "worker.log").is_file()
        and not (folder / "profile.json").exists()
        and sys.prefix == str(MLX_PYTHON.parent.parent)
        and os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Use the guarded isolated offline MLX worker",
    )
    require(
        {name: importlib.metadata.version(name) for name in MLX_VERSIONS} == MLX_VERSIONS,
        "Exact MLX runtime packages required",
    )
    _, _, expected, checkpoint = GENERATORS[family]
    require(sha(checkpoint / "model.safetensors") == expected, "Pinned generator checkpoint changed")
    sys.path.append(str(Path(__file__).parent))
    email = importlib.import_module("generate_email_upgrade")
    loader = email.mlx_generate
    # Preserve the previous generator and substitute only the prompt in this worker.
    local_loader = FunctionType(loader.__code__, {**loader.__globals__, "prompt": prompt})
    predict, precision = local_loader(checkpoint, "mps")
    require(precision == "mlx-affine-q4-g64", "Expected pinned generator precision")
    output.mkdir(mode=0o700)
    journal = output / "private-rewrites.jsonl"
    with journal.open("x", encoding="utf-8") as stream:
        for row in rows:
            (raw,) = predict([row])
            case = rewrite(row, family, raw)
            stream.write(json.dumps({"case": case, "raw_output": raw}, ensure_ascii=False) + "\n")
            stream.flush()
    write(
        output / "result.json",
        {
            "status": "complete",
            "family": family,
            "freeze_sha256": sha(FREEZE),
            "journal_sha256": sha(journal),
            "weights_sha256": expected,
            "new_neural_calls": len(rows),
        },
    )


def run(batch):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    require(type(batch) is int and 0 <= batch < len(GENERATORS), "One exact generator stage required")
    require(MLX_PYTHON.is_file(), "Existing isolated MLX interpreter required")

    def launch(argv, **kwargs):
        expected = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(expected), str(batch)],
            "Exact owned generation command required",
        )
        return subprocess.Popen(
            [str(MLX_PYTHON), *argv[1:3], MODULE, "_worker", str(batch), str(expected)], **kwargs
        )

    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda _, stage: stage == 0 or (TRIAL / "extract-00/profile.json").is_file(),
            "__file__": __file__,
            "subprocess": SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
        },
    )("extract", batch)


def readout():
    from scripts.grid_source_speech_v147 import qualification_profile
    from scripts.text_wikinews_archive_v205 import retain

    rows = verify()
    stages = [TRIAL / f"extract-{batch:02d}" for batch in range(len(GENERATORS))]
    profiles = {
        family: read(stage / "profile.json") if (stage / "profile.json").is_file() else None
        for family, stage in zip(GENERATORS, stages, strict=True)
    }
    stages_status = {
        family: "pending"
        if profile is None
        else (
            "unavailable"
            if profile["guard"] or profile["child_exit_code"] != 0 or profile["sample_errors"]
            else "complete"
        )
        for family, profile in profiles.items()
    }
    if any(status != "complete" for status in stages_status.values()):
        return {
            "status": "pending_generation"
            if "pending" in stages_status.values()
            else "generation_unavailable",
            "planned_rewrites": len(rows) * len(GENERATORS),
            "generator_stages": stages_status,
            "rewrites_without_completed_stage": len(rows)
            * sum(v != "complete" for v in stages_status.values()),
        }
    by_group = {
        row["group"]: [
            {
                "case": f"{row['group']}-human",
                "article_group": row["group"],
                "generator": None,
                "arm": "human",
                "status": "constructed",
                "text": row["text"],
                "text_sha256": row["text_sha256"],
                "ai_spans": [],
            }
        ]
        for row in rows
    }
    profiles, counts = {}, {}
    for batch, (family, stage) in enumerate(zip(GENERATORS, stages, strict=True)):
        result = qualification_profile(stage, len(rows), {}, Path(__file__).name, "extract", batch)
        journal = stage / "output/private-rewrites.jsonl"
        require(
            result["family"] == family
            and result["freeze_sha256"] == sha(FREEZE)
            and result["journal_sha256"] == sha(journal)
            and result["weights_sha256"] == GENERATORS[family][2],
            "Exact complete guarded generation receipt required",
        )
        dispositions = Counter()
        for original, attempt in zip(rows, records(journal), strict=True):
            case = rewrite(original, family, attempt["raw_output"])
            require(case == attempt["case"], "Recorded coordinates or generation disposition changed")
            by_group[original["group"]].append(case)
            dispositions[case["reason"] or "constructed"] += 1
        profiles[family], counts[family] = sha(stage / "profile.json"), dict(dispositions)
    cases = TRIAL / "private-cases.jsonl"
    journal = "".join(
        json.dumps(case, ensure_ascii=False) + "\n" for group in by_group.values() for case in group
    )
    retain(cases, journal.encode())
    result = {
        "schema": "chipmunk.text-elife-rewrites-result/273",
        "status": "complete",
        "freeze_sha256": sha(FREEZE),
        "source_groups": len(rows),
        "human_controls": len(rows),
        "planned_rewrites": len(rows) * len(GENERATORS),
        "generation_dispositions": counts,
        "private_cases_sha256": sha(cases),
        "resource_profile_sha256": profiles,
        "new_detector_calls": 0,
    }
    result_path = TRIAL / "result.json"
    if result_path.exists():
        require(result == read(result_path), "Retained result differs from exact replay")
    else:
        write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "readout", "_worker"))
    parser.add_argument("batch", type=int, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "run":
        require(args.output is None, "Only the owned worker receives an output path")
        run(args.batch)
    elif args.command == "_worker":
        worker(args.batch, args.output)
    else:
        require(args.batch is None and args.output is None, "Unexpected phase arguments")
        print(json.dumps(freeze() if args.command == "freeze" else readout(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
