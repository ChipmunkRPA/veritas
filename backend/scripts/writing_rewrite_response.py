# SPDX-License-Identifier: Apache-2.0
"""Compare first- and second-rewrite similarity as exact sentence evidence."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.writing_transition_deviation import file_hash, read, save, text_hash

SCHEMA = "chipmunk.writing-rewrite-response/1"
METHODS = ("first_step", "two_step")
PROMPT = (
    "Paraphrase the sentence below in clear natural English. Preserve the meaning, proper names and numbers. "
    "Return one complete sentence only, about {words} words.\n\n{sentence}"
)
RULE = {
    "minimum_words": 14,
    "maximum_words": 58,
    "min_usable_foci": 3,
    "max_inputs": 500,
    "held_source_groups_per_publisher": 4,
    "valid_similarity_min": 0.65,
    "valid_length_ratio": [0.64, 1.55],
    "min_iou": 0.30,
    "min_recall": 0.50,
    "selection": "one unique strongest eligible focus; strict excess over the second-highest calibration source-group maximum",
    "first_step": "similarity(original, first)",
    "two_step": "similarity(original, first) - similarity(first, second)",
    "advance": "two additional held work groups across at least two editor families at no more than one added ordinary group",
    "prompt_sha256": text_hash(PROMPT),
}
FAMILIES = ("qwen", "smol2", "granite")
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def _units(text):
    from scripts.writing_alignment_preference import sentences

    return sentences(text)


def _record(group, publisher, role, split, text, digest, truth, supplied_bounds=None, url=None):
    from scripts.writing_multi_passage_panel import _prefix

    units = _units(text)
    physical = [[unit.start, unit.end] for unit in units]
    if (
        text_hash(text) != digest
        or supplied_bounds is not None
        and physical != supplied_bounds
        or any(span not in physical for span in truth)
    ):
        raise ValueError("Source text, labels or sentence geometry differs from the retained study")
    indices = [
        index
        for index, unit in enumerate(units)
        if RULE["minimum_words"] <= len(WORD.findall(unit.text)) <= RULE["maximum_words"]
    ]
    prefix = _prefix(text)
    return {
        "group": group,
        "publisher": publisher,
        "role": role,
        "split": split,
        "sha256": digest,
        "url": url,
        "physical": physical,
        "lengths": [unit.n_words for unit in units],
        "compact": [[prefix[a], prefix[b]] for a, b in physical],
        "truth": [[prefix[a], prefix[b]] for a, b in truth],
        "native_truth": truth,
        "indices": indices,
        "text": text,
    }


def freeze(alignment, economic, editorial, output):
    cases = []
    for row in read(alignment)["cases"]:
        cases.append(
            _record(
                row["id"],
                row["publisher"],
                row["role"],
                row["split"],
                row["text"],
                row["sha256"],
                [row["truth"]] if row["truth"] else [],
                url=row["url"],
            )
        )
    for work in read(economic)["works"]:
        for index, row in enumerate(work["cases"]):
            role = (
                work["generator"]
                if row["kind"] == "mixed"
                else "authentic"
                if index == 0
                else "same_article_swap"
            )
            cases.append(
                _record(
                    work["group"],
                    work["publisher"],
                    role,
                    "held",
                    row["text"],
                    row["native_text_sha256"],
                    [place["source"] for place in row["truth"]],
                    [unit[:2] for unit in row["sentences"]],
                    work["document"],
                )
            )
    source = read(editorial)
    if source["license"] != "https://creativecommons.org/licenses/by/4.0/":
        raise ValueError("Historical editorial controls require CC BY 4.0")
    for row in source["works"]:
        for version, role in zip(row["variants"], ("authentic", "historical_edit"), strict=True):
            cases.append(
                _record(
                    row["document"],
                    "turing_way",
                    role,
                    "held",
                    version["text"],
                    version["native_text_sha256"],
                    [],
                    [s[:2] for s in version["sentences"]],
                    f"https://github.com/the-turing-way/the-turing-way/blob/{row['after_commit']}/{row['path']}",
                )
            )
    groups_by_publisher = defaultdict(set)
    for row in cases:
        if row["split"] == "held":
            groups_by_publisher[row["publisher"]].add(row["group"])
    held_selection = {
        group
        for groups in groups_by_publisher.values()
        for group in sorted(groups, key=lambda value: text_hash(SCHEMA + value))[
            : RULE["held_source_groups_per_publisher"]
        ]
    }
    cases = [row for row in cases if row["split"] == "calibration" or row["group"] in held_selection]
    fit_groups = {row["group"] for row in cases if row["split"] == "calibration"}
    held_groups = {row["group"] for row in cases if row["split"] == "held"}
    hashes = {
        split: {digest for row in cases if row["split"] == split for digest in _probe_inputs(row)}
        for split in ("calibration", "held")
    }
    if fit_groups & held_groups or hashes["calibration"] & hashes["held"]:
        raise ValueError("Calibration overlaps with a held source group or exact probe sentence")
    if len(hashes["calibration"] | hashes["held"]) > RULE["max_inputs"]:
        raise ValueError("Probe input count exceeds the frozen compute budget")
    print(
        dict(Counter((row["split"], row["role"]) for row in cases)),
        {k: len(v) for k, v in hashes.items()},
        flush=True,
    )
    save(
        output,
        {
            "schema": SCHEMA,
            "rule": RULE,
            "runner_sha256": file_hash(Path(__file__)),
            "cases": cases,
            "input_receipts_sha256": [file_hash(path) for path in (alignment, economic, editorial)],
        },
    )


def _probe_inputs(row):
    phrases = [row["text"][slice(*row["physical"][index])] for index in row["indices"]]
    return {text_hash(phrase): phrase for phrase in phrases}


def _plan(path):
    frozen = read(path)
    if frozen["runner_sha256"] != file_hash(Path(__file__)) or frozen["rule"] != RULE:
        raise ValueError("The frozen probe implementation or operating plan changed")
    return frozen


def _valid(source, reply):
    value = " ".join(reply.strip().split())
    low, high = RULE["valid_length_ratio"]
    return (
        value
        if value
        and value[-1] in ".!?"
        and value[0] not in "\"“'‘#*["
        and low <= len(WORD.findall(value)) / len(WORD.findall(source)) <= high
        and sorted(NUMBER.findall(value)) == sorted(NUMBER.findall(source))
        and len(_units(value)) == 1
        else None
    )


def generate(plan, split, output, lock=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Run the pinned probe with offline model loading")
    cases = _plan(plan)["cases"]
    if split == "held" and (lock is None or read(lock)["plan_sha256"] != file_hash(plan)):
        raise ValueError("The genuine calibration must be frozen before held inputs are probed")
    inputs = {
        key: value for row in cases if row["split"] == split for key, value in _probe_inputs(row).items()
    }
    model_id, revision, expected, checkpoint = GENERATORS["qwen"]
    if file_hash(checkpoint / "model.safetensors") != expected:
        raise ValueError("The probe model differs from its pinned Apache-2.0 checkpoint")
    mx.set_cache_limit(96 * 1024 * 1024)
    model, tokenizer = load(
        str(checkpoint), tokenizer_config={"local_files_only": True, "trust_remote_code": False}
    )
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    sampler = make_sampler(temp=0.0)

    def run(phrases):
        ordered = sorted(set(phrases), key=lambda value: (len(value), text_hash(value)))
        outputs = {}
        for index in range(0, len(ordered), 12):
            texts = ordered[index : index + 12]
            prompts = [
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": PROMPT.format(words=len(WORD.findall(text)), sentence=text),
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    add_special_tokens=False,
                )
                for text in texts
            ]
            results = batch_generate(model, tokenizer, prompts, max_tokens=113, sampler=sampler).texts
            outputs.update((text, _valid(text, result)) for text, result in zip(texts, results, strict=True))
            if index % 120 == 0 or index + 12 >= len(ordered):
                print(split, min(index + 12, len(ordered)), "/", len(ordered), flush=True)
        return outputs

    first = run(inputs.values())
    second = run(value for value in first.values() if value is not None)
    pairs = {key: [a, second.get(a)] if (a := first[text]) else [None, None] for key, text in inputs.items()}
    save(
        output,
        {
            "schema": SCHEMA,
            "plan_sha256": file_hash(plan),
            "lock_sha256": file_hash(lock) if lock else None,
            "split": split,
            "pairs": pairs,
            "probe": [model_id, revision, expected],
            "runtime": "MLX four-bit greedy",
        },
    )


def capture(plan, transcript, encoder, output):
    import numpy as np

    from scripts.writing_transition_deviation import ASSETS, encode

    case_rows = _plan(plan)["cases"]
    responses = read(transcript)
    split = responses["split"]
    cases = [row for row in case_rows if row["split"] == split]
    phrases = {key: value for row in cases for key, value in _probe_inputs(row).items()}
    if responses["plan_sha256"] != file_hash(plan) or phrases.keys() != responses["pairs"].keys():
        raise ValueError("Probe responses belong to different detector inputs")
    additions = {value for pair in responses["pairs"].values() for value in pair if value is not None}
    vectors = encode(
        [
            {
                "cases": [
                    {"text": value, "sentences": [[0, len(value)]], "native_text_sha256": text_hash(value)}
                    for value in {*phrases.values(), *additions}
                ]
            }
        ],
        encoder,
        "cpu",
    )

    def cosine(a, b):
        return round(float(np.dot(vectors[a], vectors[b]).clip(-1.0, 1.0)), 8)

    readings = {}
    for key, phrase in phrases.items():
        first, second = responses["pairs"][key]
        readings[key] = [
            cosine(phrase, first) if first is not None else None,
            cosine(first, second) if second is not None else None,
        ]
    results = []
    for row in cases:
        probes = [[i, *readings[text_hash(row["text"][slice(*row["physical"][i])])]] for i in row["indices"]]
        results.append(
            {key: value for key, value in row.items() if key not in {"text", "indices"}}
            | {"readings": probes}
        )
    save(
        output,
        {
            "schema": SCHEMA,
            "plan_sha256": file_hash(plan),
            "transcript_sha256": file_hash(transcript),
            "lock_sha256": responses["lock_sha256"],
            "split": split,
            "cases": results,
            "probe": responses["probe"],
            "encoder": ASSETS,
        },
    )


def _score(row, method):
    _, first, second = row
    if first is None or second is None or min(first, second) < RULE["valid_similarity_min"]:
        return None
    return first if method == "first_step" else round(first - second, 8)


def _choice(row, method):
    eligible = [(_score(item, method), item[0]) for item in row["readings"]]
    ordered = sorted([(value, index) for value, index in eligible if value is not None], reverse=True)
    if len(ordered) < RULE["min_usable_foci"]:
        return None, None, len(ordered)
    highest, index = ordered[0]
    return highest, index if highest > ordered[1][0] else None, len(ordered)


def points(cases):
    result = {}
    for method in METHODS:
        grouped = defaultdict(list)
        for row in cases:
            if row["role"] in FAMILIES:
                raise ValueError("Calibrate only on untouched or ordinary edited input")
            if (value := _choice(row, method)[0]) is not None:
                grouped[row["group"]].append(value)
        values = sorted(max(scores) for scores in grouped.values())
        if len(values) < 8:
            raise ValueError("Fewer than eight independent assessable ordinary groups")
        result[method] = values[-2]
    return result


def calibrate(plan, captured, output):
    receipt = read(captured)
    if receipt["split"] != "calibration" or receipt["plan_sha256"] != file_hash(plan):
        raise ValueError("The calibration receipt belongs to a different source list")
    cutoffs = points(receipt["cases"])
    print("Frozen points", cutoffs, flush=True)
    save(
        output,
        {
            "rule": RULE,
            "plan_sha256": file_hash(plan),
            "calibration_sha256": file_hash(captured),
            "points": cutoffs,
        },
    )


def baseline(plan, output):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from app.detectors.text.pipeline import windows
    from app.evidence.document_passage_intervals import MIN_SENTENCES, _reviewed_choices
    from scripts.writing_alignment_preference import CLASSIFIER, model_location
    from scripts.writing_word_edit_transfer import model_assets

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    snapshot = model_location("wasitaigeneratedcom/ai-text-detector-small", CLASSIFIER)
    model_assets(snapshot)
    model = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    cases = [row for row in _plan(plan)["cases"] if row["split"] == "held"]
    views = [(windows(units, 3), [u.text for u in units]) for row in cases if (units := _units(row["text"]))]
    strings = sorted(
        {value for case in views for view in case for value in view},
        key=lambda value: (len(value), text_hash(value)),
    )
    scores = {}
    for start in range(0, len(strings), 192):
        entries = strings[start : start + 192]
        scores.update(zip(entries, model.predict(entries, batch=12, check=lambda _: None), strict=True))
    results = []
    for row, (contexts, foci) in zip(cases, views, strict=True):
        reads = {
            key: [round(float(scores[value]), 6) for value in values]
            for key, values in (("context", contexts), ("isolated", foci))
        }
        geometry = [[*places, count] for places, count in zip(row["physical"], row["lengths"], strict=True)]
        selected = (
            _reviewed_choices(geometry, reads, document_baseline=True, report_review=True)[0]
            if len(foci) >= MIN_SENTENCES
            else []
        )
        results.append(
            {
                "sha256": row["sha256"],
                "assessed": len(foci) >= MIN_SENTENCES,
                "readings": reads,
                "intervals": [[a, b] for a, b, *_ in selected],
            }
        )
    save(output, {"schema": SCHEMA, "plan_sha256": file_hash(plan), "cases": results})


def summarize(cases, current, cutoffs):
    from app.bench.writing_entropy_prominence import matched

    values = defaultdict(Counter)
    groups = defaultdict(set)
    for row, baseline_row in zip(cases, current, strict=True):
        if row["sha256"] != baseline_row["sha256"]:
            raise ValueError("The candidate and existing rule read different native inputs")
        compact, truth = row["compact"], row["truth"]
        existing = [[compact[a][0], compact[b - 1][1]] for a, b in baseline_row["intervals"]]
        old = len(matched(existing, truth))
        for method in METHODS:
            peak, index, usable = _choice(row, method)
            focus = compact[index] if index is not None else None
            new = focus if focus is not None and peak > cutoffs[method] else None
            hit = bool(new is not None and matched([new], truth))
            added = len(matched([*existing, new], truth)) - old if new is not None else 0
            available = [compact[item[0]] for item in row["readings"] if _score(item, method) is not None]
            if added:
                groups[method, "new"].add(row["group"])
                groups[method, "family"].add(row["role"])
            if new is not None and row["role"] not in FAMILIES and not existing:
                groups[method, "ordinary"].add(row["group"])
            for key in ("all", row["publisher"], row["role"]):
                values[method, key].update(
                    {
                        "files": 1,
                        "assessed": usable >= RULE["min_usable_foci"],
                        "truth": len(truth),
                        "baseline_assessed": baseline_row["assessed"],
                        "baseline_flags": bool(existing),
                        "baseline_hits": old,
                        "candidate_flags": new is not None,
                        "candidate_hits": hit,
                        "added_hits": added,
                        "added_ordinary_alerts": new is not None
                        and not existing
                        and row["role"] not in FAMILIES,
                        "wrong_edited_locations": new is not None and row["role"] in FAMILIES and not hit,
                        "potential_hits": len(matched(available, truth)),
                        "highest_eligible_hits": bool(focus is not None and matched([focus], truth)),
                    }
                )
    return {
        method: {key: dict(counts) for (name, key), counts in values.items() if name == method}
        | {
            "new_groups": len(groups[method, "new"]),
            "added_families": sorted(groups[method, "family"]),
            "added_ordinary_groups": len(groups[method, "ordinary"]),
            "advances": len(groups[method, "new"]) >= 2
            and len(groups[method, "family"]) >= 2
            and len(groups[method, "ordinary"]) <= 1,
        }
        for method in METHODS
    }


def publish(calibration, held, calibration_lock, current, output):
    a, b, lock, prior = (read(path) for path in (calibration, held, calibration_lock, current))
    if (
        len({value["plan_sha256"] for value in (a, b, lock, prior)}) != 1
        or lock["calibration_sha256"] != file_hash(calibration)
        or b["lock_sha256"] != file_hash(calibration_lock)
        or lock["points"] != points(a["cases"])
    ):
        raise ValueError("Incompatible calibration, held, or incumbent input")
    save(
        output,
        {
            "schema": SCHEMA,
            "rule": RULE,
            "calibration": a,
            "held": b,
            "baseline": prior,
            "points": lock["points"],
            "outcomes": summarize(b["cases"], prior["cases"], lock["points"]),
        },
    )
    review(output)


def review(path):
    result = read(path)
    cutoffs = points(result["calibration"]["cases"])
    counts = summarize(result["held"]["cases"], result["baseline"]["cases"], cutoffs)
    if cutoffs != result["points"] or counts != result["outcomes"] or result["rule"] != RULE:
        raise ValueError("The public measurements and decisions do not replay")
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    stages = {
        "freeze": (freeze, ("alignment", "economic", "editorial", "output")),
        "generate": (generate, ("plan", "output")),
        "capture": (capture, ("plan", "transcript", "encoder", "output")),
        "calibrate": (calibrate, ("plan", "captured", "output")),
        "baseline": (baseline, ("plan", "output")),
        "publish": (publish, ("calibration", "held", "calibration_lock", "current", "output")),
        "review": (review, ("path",)),
    }
    for name, (_, fields) in stages.items():
        mode = actions.add_parser(name)
        for field in fields:
            mode.add_argument(field, type=Path)
        if name == "generate":
            mode.add_argument("--split", choices=("calibration", "held"), required=True)
            mode.add_argument("--lock", type=Path)
    args = vars(parser.parse_args())
    stages[args.pop("action")][0](**args)
