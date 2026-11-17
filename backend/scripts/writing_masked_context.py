# SPDX-License-Identifier: Apache-2.0
"""Run and replay a source-separated comparison of local masked-word readings."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from app.bench import writing_masked_context as bench
from scripts import writing_masked_word_reader as reader
from scripts import writing_multi_passage_panel as source
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_entropy_prominence import existing

SCHEMA = "chipmunk.writing-masked-context/1"
ROOT = Path(__file__).resolve().parents[2]


def implementation():
    return {Path(path).resolve().relative_to(ROOT).as_posix(): reader.hash_file(path)
            for path in (__file__, bench.__file__, reader.__file__)}


def binding(plan_file):
    return reader.hash_file(plan_file)


def previous_predictions(baseline):
    if "cases" in baseline:
        return {(case["document"], case["role"]): case for case in baseline["cases"]}
    return {(work["document"], variant["version"]): variant
            for work in baseline["works"] for variant in work["variants"] if variant["layout"] == "published"}


def incumbent(case, variant):
    if "existing" in case:
        if case["native_sha256"] != variant["native_text_sha256"] or case["sentences"] != variant["sentences"]:
            raise ValueError("Existing predictions refer to different displayed source text")
        return case["existing"], case["incumbent_assessed"]
    return existing(variant, case)


def capture(plan_file, baseline_file, checkpoint, output, device, freeze_file=None):
    plan = source.read(plan_file)
    prior = previous_predictions(source.read(baseline_file))
    freeze = source.read(freeze_file) if freeze_file else None
    current = implementation()
    if freeze and (freeze["implementation"] != current or binding(plan_file) not in freeze["held_inputs"]):
        raise ValueError("The held plan or measurement code differs from the frozen comparison")
    environment = {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "transformers", "tokenizers")}
    if freeze and freeze["runtime"] != {**environment, "device": device}:
        raise ValueError("Held measurements need the frozen runtime")
    measure = reader.read_checkpoint(checkpoint, device)
    cases = []
    for index, work in enumerate(plan["works"], 1):
        for variant in work["variants"]:
            role = variant["version"]
            if variant["layout"] != "published":
                continue
            text = variant["text"]
            digest = source.digest(text.encode())
            sentences = variant["sentences"]
            if digest != variant["native_text_sha256"]:
                raise ValueError("The extracted document differs from the frozen source bytes")
            baseline, covered = incumbent(prior[work["document"], role], variant)
            count, readings = measure(text, sentences)
            cases.append({"document": work["document"], "group": work["group_sha256"], "native_sha256": digest,
                          "genre": work["genre"], "evaluation_panel": work.get("evaluation_panel", plan["cohort"]),
                          "role": role, "split": work["split"], "generator": work.get("generator_backend"),
                          "sentences": sentences, "truth": [row["compact"] for row in variant["truth"]],
                          "existing": baseline, "incumbent_assessed": covered,
                          "token_count": count, "readings": readings})
        if index % 16 == 0 or index == len(plan["works"]):
            print(plan["cohort"], index, "of", len(plan["works"]), "source groups", flush=True)
    source.save(output, {"schema": SCHEMA, "implementation": current, "plan_sha256": binding(plan_file),
                         "baseline_sha256": binding(baseline_file), "runtime": {**environment, "device": device},
                         "checkpoint": reader.REVISION, "assets": reader.ASSETS, "cases": cases,
                         "freeze_sha256": binding(freeze_file) if freeze_file else None})


def calibrate(capture):
    train = [case for case in capture["cases"] if case["split"] == "validation"]
    selection = [case for case in capture["cases"] if case["split"] == "test"]
    if not train or not selection or {case["group"] for case in train} & {case["group"] for case in selection}:
        raise ValueError("Earlier fitting and selection sources are missing or overlap")
    ordinary = [case for case in train if case["role"] == "ordinary"]
    scales = bench.calibrate_scales(ordinary)
    weights = bench.paired_weights(train, scales)
    thresholds = bench.points(ordinary, scales, weights)
    comparison = bench.summarize(selection, scales, weights, thresholds)
    return {"scales": scales, "weights": weights, "thresholds": thresholds,
            "choice": bench.choose(comparison), "selection": comparison}


def fit(earlier, output, held_plans):
    measured = source.read(earlier)
    if measured["implementation"] != implementation() or measured["freeze_sha256"] is not None:
        raise ValueError("Create the development receipt with this measurement code")
    groups = {case["group"] for case in measured["cases"]}
    documents = {case["document"] for case in measured["cases"]}
    digests = {case["native_sha256"] for case in measured["cases"]}
    for path in held_plans:
        for work in source.read(path)["works"]:
            inputs = {variant["native_text_sha256"] for variant in work["variants"]}
            if work["group_sha256"] in groups or work["document"] in documents or inputs & digests:
                raise ValueError("A held document or native source overlaps earlier measurements")
            groups.add(work["group_sha256"])
            documents.add(work["document"])
            digests.update(inputs)
    frozen = calibrate(measured)
    print("Selection", json.dumps(frozen["choice"], sort_keys=True), "weights", frozen["weights"], flush=True)
    source.save(output, frozen | {"schema": SCHEMA, "implementation": measured["implementation"],
                                  "runtime": measured["runtime"], "earlier_sha256": binding(earlier),
                                  "held_inputs": sorted(binding(path) for path in held_plans)})


def verify_ranges(cases):
    seen = set()
    for case in cases:
        key = (case["document"], case["role"])
        if key in seen:
            raise ValueError("The same source delivery occurs twice")
        seen.add(key)
        sentences = case["sentences"]
        previous = -1
        for index, tokens, words, pieces, full, alone, offsets in case["readings"]:
            if not (previous < index < len(sentences) and 0 < tokens <= reader.MAX_FOCUS
                    and reader.MIN_WORDS <= words == len(offsets) <= reader.WORDS_PER_SENTENCE
                    and words <= pieces <= min(tokens, reader.MAX_WORD_PIECES * words)
                    and min(full, alone) >= 0 and 0 < case["token_count"] <= reader.MAX_DOCUMENT):
                raise ValueError("A masked reading falls outside its declared native sentence")
            first, end, *_ = sentences[index]
            for lo, hi in offsets:
                if not first <= lo < hi <= end:
                    raise ValueError("A masked word is outside its original sentence")
                first = hi
            previous = index
        for first, stop in case["existing"]:
            if not 0 <= first < stop <= len(sentences):
                raise ValueError("An existing writing finding falls outside its source")


def evaluate(freeze, cases, matcher=None):
    results = bench.summarize(cases, freeze["scales"], freeze["weights"], freeze["thresholds"], match=matcher)
    panels = {}
    for panel in sorted({case["evaluation_panel"] for case in cases}):
        subset = [case for case in cases if case["evaluation_panel"] == panel]
        outcomes = bench.summarize(subset, freeze["scales"], freeze["weights"], freeze["thresholds"], match=matcher)
        panels[panel] = {key: value for key, value in outcomes.items() if key.endswith("/all")}
    return {"overall": results, "by_panel": panels}


def verify(earlier, freeze_file, held_files, output=None):
    initial, freeze = source.read(earlier), source.read(freeze_file)
    if freeze["earlier_sha256"] != binding(earlier) or freeze["implementation"] != implementation():
        raise ValueError("The frozen comparison differs from the measured earlier inputs or code")
    if any(freeze[key] != val for key, val in calibrate(initial).items()):
        raise ValueError("Development scales, points or selection do not reproduce")
    bound, cases = [], []
    for path in held_files:
        measured = source.read(path)
        if (measured["freeze_sha256"] != binding(freeze_file) or measured["implementation"] != freeze["implementation"]
                or measured["runtime"] != freeze["runtime"] or measured["assets"] != reader.ASSETS):
            raise ValueError("These readings did not use the frozen operating points and checkpoint")
        bound.append(measured["plan_sha256"])
        cases.extend(measured["cases"])
    if sorted(bound) != freeze["held_inputs"]:
        raise ValueError("Held measurements differ from the frozen input roster")
    verify_ranges(initial["cases"])
    verify_ranges(cases)
    for key in ("group", "document", "native_sha256"):
        if {row[key] for row in initial["cases"]} & {row[key] for row in cases}:
            raise ValueError("The measured held source overlaps earlier measurements")
    scored = evaluate(freeze, cases)
    if scored != evaluate(freeze, cases, matching):
        raise ValueError("Independent interval assignment differs from the benchmark counts")
    primary = freeze["choice"]
    key = f"{primary['policy']}/{primary['arm']}/all"
    print(json.dumps({"choice": primary, "held": scored["overall"][key],
                      "sources": dict(Counter(case["evaluation_panel"] for case in cases))}, sort_keys=True), flush=True)
    if output:
        source.save(output, scored | {"selection": primary, "inputs": [binding(path) for path in held_files],
                                     "freeze_sha256": binding(freeze_file)})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="action", required=True)
    run = mode.add_parser("capture")
    for name in ("plan_file", "baseline_file", "checkpoint", "output"):
        run.add_argument(name, type=Path)
    run.add_argument("--device", choices=("cpu", "mps"), required=True)
    run.add_argument("--freeze", dest="freeze_file", type=Path)
    frozen = mode.add_parser("fit")
    frozen.add_argument("earlier", type=Path)
    frozen.add_argument("output", type=Path)
    frozen.add_argument("held_plans", type=Path, nargs="+")
    replay = mode.add_parser("verify")
    replay.add_argument("earlier", type=Path)
    replay.add_argument("freeze_file", type=Path)
    replay.add_argument("held_files", type=Path, nargs="+")
    replay.add_argument("--output", type=Path)
    options = vars(cli.parse_args())
    {"capture": capture, "fit": fit, "verify": verify}[options.pop("action")](**options)


if __name__ == "__main__":
    main()
