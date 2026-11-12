"""Replay frozen paired writing inference on the two new journal body cohorts."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_paired_revision_replication as paired
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-journal-body-transfer/1"
SOURCE_SCHEMA = "chipmunk.writing-journal-body-holdout/1"
LAYOUT = "Test native three-sentence inputs across a paragraph boundary. Compare existing app eligibility with a "+\
         "whitespace-only layout relaxation. Require the existing wording, focus, and exact input-hash checks."


def freeze_document(source):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    originals, receipt = source / "originals.json", source / "source-receipt.json"
    binding = read(receipt)
    require(binding["schema"] == SOURCE_SCHEMA and binding["originals_sha256"] == digest(originals)
            and read(originals), "A qualified and replayed journal roster is required")
    metadata = paired.generator_configuration()
    assets = {}
    for family in paired.SEEDS:
        _, _, expected, folder = GENERATORS[family]
        require(metadata[family]["license"] == "Apache-2.0" and digest(folder / "model.safetensors") == expected,
                "Generator weights or commercial license differs")
        assets[family] = metadata[family]
    own = Path(__file__).resolve()
    runtime = own.parents[1] / "app/evidence/document_revision_writing.py"
    return {"schema": SCHEMA, "scope": "holdout", "method": paired.METHOD, "app_layout": LAYOUT,
            "code_sha256": digest(own), "replication_sha256": digest(Path(paired.__file__)),
            "app_scope_sha256": digest(runtime), "source_sha256": digest(originals),
            "source_receipt_sha256": digest(receipt), "generator_assets": assets, "runtime": MLX_VERSIONS,
            "seeds": paired.SEEDS, "prompt": paired.EDIT_PROMPT, "settings": paired.SETTINGS}


def app_scope(rows, outcomes):
    from app.evidence.document_revision import compare_with_text
    from app.evidence.document_revision_writing import paired_input

    counts = defaultdict(Counter)
    earlier = {row["group"]: row["text"] for row in rows if row["role"] == "before"}
    for row, outcome in zip(rows, outcomes, strict=True):
        role = row["role"]
        if role == "before":
            continue
        a, b = earlier[row["group"]], row["text"]
        current, old, new = compare_with_text(a.encode(), b.encode(), ".txt")
        scoped = paired_input(current, old, new)
        bucket = counts[f"{row['publisher']}/{role}"]
        bucket["pairs"] += 1
        if scoped is None and "\n\n" in a and "\n\n" in b:
            flat = [text.replace("\n\n", " ").encode() for text in (a, b)]
            comparison, old, new = compare_with_text(*flat, ".txt")
            scoped = paired_input(comparison, old, new)
            key = "added_layout_only"
        else:
            key = "current_scope"
        if scoped is not None:
            index = row["target"]
            hashes = (scoped.geometry["earlier"]["input_sha256"], scoped.geometry["later"]["input_sha256"])
            if hashes == (row["reference"][index], row["windows"][index]["sha256"]):
                bucket[key] += 1
                bucket[key + "_flag"] += outcome["paired_flag"]
                continue
            bucket["layout_changed_native_input"] += 1
        bucket["unassessed"] += 1
    return {name: dict(value) for name, value in counts.items()}


def main():
    from scripts.review_writing_paired_revision_replication import audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "generate", "freeze-scoring", "current", "second", "report", "verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--family", choices=paired.SEEDS)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--second", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with offline snapshots")
    run, source, phase = args.run, args.source, args.phase
    freeze = run / "freeze.json"
    expected = freeze_document(source)
    if phase == "freeze":
        run.mkdir(exist_ok=True)
        write(freeze, expected)
        return
    protocol = read(freeze)
    if protocol != expected:
        original = run / "frozen-generation-controller.py"
        require(phase != "generate" and original.exists() and digest(original) == protocol["code_sha256"]
                and protocol | {"code_sha256": expected["code_sha256"]} == expected,
                "Frozen method or source differs")
    if phase == "generate":
        require(args.family, "Select one frozen generator")
        paired.generate(run, source, protocol, args.family)
        return
    rows, values, admission = paired.register(run, source)
    cases, scoring = run / "cases.json", run / "scoring-freeze.json"
    if phase == "freeze-scoring":
        write(cases, rows)
    locked = paired.scores_lock(run, args.current, args.second) | {"scoring_controller_sha256": expected["code_sha256"]}
    if phase == "freeze-scoring":
        write(scoring, locked)
        print(f"Frozen {len(rows)} cases and {len(values)} exact inputs.", flush=True)
        return
    require(read(cases) == rows and read(scoring) == locked, "Frozen scoring cases or weights differ")
    if phase in ("current", "second"):
        paired.score(run, source, protocol, phase, getattr(args, phase), values)
        return
    outcomes, result = paired.aggregate(run, protocol, rows, values, admission)
    scope = app_scope(rows, outcomes)
    results_path, outcomes_path = run / "aggregate.json", run / "private-outcomes.json"
    if phase == "report":
        write(outcomes_path, outcomes)
        write(results_path, result | {"outcomes_sha256": digest(outcomes_path), "app_scope": scope})
    else:
        require(read(outcomes_path) == outcomes and read(results_path) == result | {
            "outcomes_sha256": digest(outcomes_path), "app_scope": scope,
        }, "App or model replay differs")
    print(audit(run, source), flush=True)
    print(scope, flush=True)


if __name__ == "__main__":
    main()
