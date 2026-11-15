# SPDX-License-Identifier: MIT
"""Compare an independent CPU sentence classifier with the frozen word-range check."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_range_context_transfer as previous

SCHEMA = "chipmunk.writing-independent-range-transfer/1"
RELEASE = "8b677e1bd3bc6421ae710352cda13ca65502147b"
ASSETS = {
    "model.onnx": "d720129864ed3ecb7be5de85acf33d2fd62e5d4f8351de0e4cd1310de03adfe7",
    # The scanner treats this public tokenizer file digest as a generic token.
    "tokenizer.json": "c24b1448c7b6762d0e4bf1c0b7b1e74a360039228f9ecf5a96a5fc789c07f49b",  # gitleaks:allow
    "config.json": "0b244e69cdd0bd936e7cca524a37faca8725023ca7607e909149e36e7b4f7035",
    "LICENSE": "7e096e455b9dac457008fa960d25e64066416f417a71c7ad61fbdd65158ba9b9",
}
CENTERS = (1, 2, 3)
ARMS = ("independent", "independent_contrast", "anchored", "anchored_contrast")
METHOD = {
    "inference": "Score each of the three original interior sentences in isolation, with the published tokenizer. Keep the AI minus human logit, without the publisher's whole-document probability or cutoff.",
    "comparison": "Rank independent responses by absolute score or by the margin over the other two candidates. Attach the frozen word proposal for the resulting sentence when available. The anchored arms use 75% paired word-range rank and 25% independent sentence rank.",
    "points": "Rank each component against ordinary calibration excerpts at the same sentence position and genre. Leave each ordinary work out of its own empirical ranking. Set each arm's point by the maximum allowed flagged ordinary works at the prior tighter and wider budgets, before held arithmetic.",
    "scope": "Locks the candidate model and mixture weights. Word ranges come from the previous head; a selected sentence does not imply the second model pointed to specific words. Compare exact word overlap, right sentence, wrong sentence, and matched ordinary work alerts separately.",
}


def reference():
    report, transfer = previous.signed_inputs()
    old, later = report[1], transfer[1]
    fresh = previous.read(previous.FRESH)
    previous.replication.review(fresh, previous.INPUTS["report"][1])
    elife = [row for row, _ in previous.captures(previous.EXTRA, report)]
    abstracts = [row for row, _ in previous.abstract_capture(previous.ABSTRACTS, report)]
    calibration = {"reports": old["calibration"]}
    for genre in ("news", "abstracts"):
        calibration[genre] = [row for row in later["calibration"] if row["domain"] == genre]
    panels = {
        "reports": ("reports", old["held"]),
        "report_replication": ("reports", fresh["held"]),
        "elife_body": ("abstracts", elife),
        "dated_revisions": ("reports", [row for row in later["held"] if row["domain"] == "journals"]),
        "additional_abstracts": ("abstracts", abstracts),
    }
    panels.update({genre: (genre, [row for row in later["held"] if row["domain"] == genre])
                   for genre in ("news", "abstracts")})
    point_groups = {row["group"] for rows in calibration.values() for row in rows}
    held_groups = {row["group"] for _, rows in panels.values() for row in rows}
    previous.require(not point_groups & held_groups, "A held work appears in point selection")
    paths = [previous.INPUTS["report"][1], previous.INPUTS["transfer"][1], previous.FRESH, previous.EXTRA, previous.ABSTRACTS]
    points = {"reports": old["points"]["range"]} | {key: later["points"][key]["range"] for key in ("news", "abstracts")}
    return calibration, panels, points, {path.name: previous.digest(path) for path in paths}


def abstract_text(gemini, gpt):
    from scripts import writing_focus_inside_context as intake

    data = previous.read(previous.ABSTRACTS)
    inputs = defaultdict(list)
    for row in data["inputs"]:
        inputs[row["group"]].append(row)
    for (family, pin), path in zip(previous.ABSTRACT_PINS.items(), (gemini, gpt), strict=True):
        table = intake.shard(path, "abstracts", family, pin)
        for key, versions in table.items():
            group = previous.pool.sha(key)
            pair = inputs.get(group)
            if not pair or pair[1]["family"] != family:
                continue
            selected = intake.excerpt_options(key, versions)[pair[1]["position"]]
            for (text, spans, variant), row in zip(selected, pair, strict=True):
                previous.require(row["spans"] == spans and row["variant"] == variant, "Additional abstract geometry changed")
                yield row | {"text": text}


def native(args):
    from scripts import writing_edit_occurrence_transfer as ordinary

    reports, _, _ = previous.words.input_plan(args.reports)
    transfer, _, _ = previous.words.transfer_plan(args.reports, args.matched, args.additional)
    dated, _, _ = ordinary.ordinary_inputs(args.dated)
    report, _ = previous.signed_inputs()
    exclusions = {value for view in report for item in view["fit"] for value in item[1::2]}
    exclusions.update(value for view in report for row in view["calibration"] + view["held"]
                      for value in (row["group"], row["sha256"]))
    elife, _, _ = previous.elife_cases(args.elife, exclusions)
    fresh = previous.read(args.fresh)["rows"]
    abstract = list(abstract_text(args.gemini, args.gpt))
    return {row["sha256"]: row for row in reports + transfer + dated + elife + fresh + abstract}


def capture(args):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    previous.require(all(previous.digest(args.assets / name) == pin for name, pin in ASSETS.items()),
                     "The released checkpoint, tokenizer or license differs")
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    model = ort.InferenceSession(str(args.assets / "model.onnx"), sess_options=settings, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(args.assets / "tokenizer.json"))
    calibration, panels, baseline_points, sources = reference()
    text = native(args)
    controls = {row["sha256"]: row for rows in calibration.values() for row in rows}
    held = {row["sha256"]: row for _, rows in panels.values() for row in rows}
    previous.require(not set(controls) & set(held), "Calibration and held source excerpts overlap")
    scored = {}
    points = None
    for i, (sha, expected) in enumerate((controls | held).items(), 1):
        source = text[sha]
        previous.require(source["spans"] == expected["spans"] and previous.pool.sha(source["text"]) == sha,
                         "A source excerpt or range differs from the frozen roster")
        reading = []
        for index in CENTERS:
            start, stop = source["spans"][index]
            ids = tokenizer.encode(source["text"][start:stop]).ids
            previous.require(2 < len(ids) <= 512, "An isolated sentence is outside the tested tokenizer extent")
            array = np.array([ids], dtype=np.int64)
            [logits] = model.run(["logits"], {"input_ids": array, "attention_mask": np.ones_like(array)})
            reading.append(round(float(logits[0, 1]) - float(logits[0, 0]), 7))
        scored[sha] = reading
        if i == len(controls):
            _, points = calibrate(calibration, baseline_points, scored)
            print(f"Set operating points on {i} ordinary calibration excerpts.", flush=True)
        if i % 80 == 0:
            print(f"Read {i}/{len(controls) + len(held)} source excerpts.", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "model": {"revision": RELEASE, "assets": ASSETS},
            "sources": sources, "runtime": ort.__version__, "readings": scored, "points": points}


def features(row, readings):
    scores = readings[row["sha256"]]
    ranges = {proposal["sentence"]: proposal for proposal in row["by_sentence"]}
    values = {}
    for index, score in zip(CENTERS, scores, strict=True):
        neighbor = max(other for j, other in zip(CENTERS, scores, strict=True) if j != index)
        signals = {"independent": score, "independent_contrast": round(score - neighbor, 7)}
        if index in ranges:
            signals["word"] = ranges[index]["score"]
        values[index] = signals
    return values, ranges


def references(rows, readings):
    source_values = defaultdict(dict)
    for row in rows:
        previous.require(row["family"] == "original" and not row["changed"], "Calibration needs ordinary originals")
        values, _ = features(row, readings)
        for index, scores in values.items():
            for name, value in scores.items():
                source_values[index, name][row["group"]] = value
    previous.require(len(source_values) == 9 and all(len(values) >= 22 for values in source_values.values()),
                     "Too few ordinary calibration works for the chosen sequence positions")
    return source_values


def winners(row, readings, reference, *, exclude=None):
    values, ranges = features(row, readings)
    choices = defaultdict(list)
    for index, scores in values.items():
        ranks = {}
        for name, value in scores.items():
            pool = sorted(v for group, v in reference[index, name].items() if group != exclude)
            ranks[name] = 10_000 * bisect_left(pool, value) // (len(pool) + 1)
        for arm, signal in zip(ARMS, ("independent", "independent_contrast") * 2, strict=True):
            anchored = arm.startswith("anchored")
            if not anchored or "word" in ranks:
                rank = 3 * ranks["word"] + ranks[signal] if anchored else ranks[signal]
                choices[arm].append({"score": rank, "sentence": index, "words": ranges.get(index, {}).get("words", [])})
    result = {"paired": row["best"]["range"]}
    for arm in ARMS:
        ranked = sorted(choices[arm], key=lambda entry: -entry["score"])
        result[arm] = ranked[0] if ranked and (len(ranked) == 1 or ranked[0]["score"] > ranked[1]["score"]) else None
    return result


def calibrate(calibration, baseline_points, readings):
    refs, points = {}, {}
    for genre, rows in calibration.items():
        ref = refs[genre] = references(rows, readings)
        ordinary = [winners(row, readings, ref, exclude=row["group"]) for row in rows]
        points[genre] = {"paired": baseline_points[genre]}
        for arm in ARMS:
            values = sorted((sample[arm]["score"] if sample[arm] else 0 for sample in ordinary), reverse=True)
            points[genre][arm] = {name: values[len(values) // divisor] for name, divisor in previous.LEVELS.items()}
    return refs, points


def evaluate(document):
    calibration, panels, baseline_points, sources = reference()
    readings = document["readings"]
    expected = {row["sha256"] for batch in [*calibration.values(), *(rows for _, rows in panels.values())] for row in batch}
    previous.require(document["schema"] == SCHEMA and document["method"] == METHOD and document["sources"] == sources
                     and document["model"] == {"revision": RELEASE, "assets": ASSETS} and set(readings) == expected
                     and all(len(r) == 3 and all(type(v) in (float, int) and math.isfinite(v) for v in r) for r in readings.values()),
                     "Receipt inputs or model readings differ")
    refs, points = calibrate(calibration, baseline_points, readings)
    previous.require(document["points"] == points, "The frozen operating points differ")
    outcomes, known_sentence = {}, {}
    for panel, (genre, rows) in panels.items():
        measures = defaultdict(Counter)
        diagnostics = Counter()
        selections = [row | {"reports": winners(row, readings, refs[genre])} for row in rows]
        for row in selections:
            if row["family"] not in previous.ORDINARY:
                target = next((proposal for proposal in row["by_sentence"] if proposal["sentence"] == row["position"]), None)
                diagnostics.update(rewrites=1, target_range_available=bool(target),
                                   useful_with_known_sentence=bool(target and previous.words.useful_words(
                                       set(target["words"]), set(row["changed"]))))
            for arm, candidate in row["reports"].items():
                for level, point in points[genre][arm].items():
                    flagged = candidate is not None and candidate["score"] > point
                    depends_on_words = arm == "paired" or arm.startswith("anchored")
                    measures[f"{arm}/{level}/{row['family']}"].update(
                        unassessed=depends_on_words and not row["by_sentence"],
                        exact_sentence=flagged and candidate["sentence"] == row["position"],
                        no_word_range=flagged and not candidate["words"])
        outcomes[panel] = {name: values | dict(measures[name]) for name, values in
                           previous.count(selections, points[genre]).items()}
        known_sentence[panel] = dict(diagnostics)
    return {"points": points, "outcomes": outcomes, "known_sentence": known_sentence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    measure = phases.add_parser("capture")
    for name in ("assets", "reports", "matched", "additional", "dated", "elife", "fresh", "gemini", "gpt"):
        measure.add_argument(f"--{name}", type=Path, required=True)
    measure.add_argument("output", type=Path)
    review = phases.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "capture":
        document = capture(args)
        document.update(evaluate(document))
        with args.output.open("xb") as output:
            output.write(gzip.compress(json.dumps(document, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    else:
        document = previous.read(args.receipt)
        result = evaluate(document)
        previous.require(all(document[key] == value for key, value in result.items()), "Held range outcomes differ")
        print(json.dumps(result, sort_keys=True, indent=2))
