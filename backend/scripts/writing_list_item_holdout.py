# SPDX-License-Identifier: Apache-2.0
"""Test native list-item scoring on single documents with one rewritten item."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from itertools import groupby
from pathlib import Path
from statistics import median

from scripts.writing_transition_deviation import file_hash, read, save, text_hash
from scripts.writing_turing_editorial_history import (
    CHAPTER,
    LICENSE,
    REVISION,
    SOURCE,
    git,
    license_digest,
    snapshot,
)

SCHEMA = "chipmunk.writing-list-item/1"
ITEM = re.compile(r"^[ ]{0,3}([-*+]|\d+[.)]|•)\s+([^\r\n]+)$")
FOREIGN = re.compile(r"`|https?:|www\.|@|\[|\]|[©“”«»\"]|\bcopyright\b|\badapted from\b", re.I)
WIDTH = 5
MIN_WORDS = 7
MAX_WORDS = 66
POINT = .90
MAX_FRESH = 30
ARMS = ("isolated", "relative")
BASE_SEED = 733
PROMPT = ("Rewrite the target checklist item as one complete English sentence. Preserve its meaning, numbers and proper "
          "names. Change the opening and sentence structure and use substantially different wording. Do not add new facts. "
          "Output only the revised item, with no introduction or list marker.")
RULES = {
    "item_count": WIDTH, "length_bounds": [MIN_WORDS, MAX_WORDS], "ordinary_point": POINT, "maximum_sources": MAX_FRESH,
    "position": "Choose one interior complete item by source hash before generation.",
    "candidate": "Compare the whole item with the existing local classifier. The relative arm subtracts the median log odds "
                 "of the three closest-length other items from the same upload. Each reading can add at most one item.",
    "holdout": "Use actual source lists from different files; assign entire chapter groups to calibration or held out. Keep "
               "one unaltered and one marker-only restyled control per held source. Score each delivery independently.",
    "advancement": "Require at least two additional useful locations that do not repeat at the same target in the ordinary "
                   "counterpart, across two generators, with at most two additional ordinary source groups flagged.",
    "range_match": {"iou": .30, "recall": .50},
    "prompt_sha256": text_hash(PROMPT), "seed": BASE_SEED,
}


def items(lines):
    """Keep consecutive list items at the same indentation and marker style."""
    def row(index, line):
        match = ITEM.fullmatch(line)
        if not match or FOREIGN.search(match[2]):
            return None
        marker, text = match.groups()
        text = text.rstrip()
        if (not MIN_WORDS <= len(re.findall(r"\b\w+\b", text)) <= MAX_WORDS or not text[0].isalpha()
                or re.search(r"\s{2}|\t", text)):
            return None
        return {"line": index, "prefix": line[:match.start(2)], "text": text,
                "style": (match.start(1), marker[-1])}

    for style, entries in groupby((row(i, line) for i, line in enumerate(lines, 1)),
                                   key=lambda item: item["style"] if item else None):
        if style is not None:
            block = list(entries)
            for index in range(len(block) - WIDTH + 1):
                yield block[index:index + WIDTH]


def reserve(repo, output, previous):
    from scripts.writing_economic_paraphrase_holdout import find_texts, fragments

    paths = [path for path in git(repo, "ls-tree", "-r", "--name-only", REVISION, "book/content", "book/website").decode().splitlines()
             if CHAPTER.fullmatch(path)]
    terms = license_digest(repo, REVISION)
    if not terms:
        raise ValueError("The historical publisher revision lacks the expected reuse terms")
    seen = {part for path in previous for text in find_texts(read(path)) for part in fragments(text)}
    works, groups = [], Counter()
    for path in sorted(paths, key=lambda value: text_hash(SCHEMA + value)):
        match = CHAPTER.fullmatch(path)
        chapter = match[2] if match[1] == "content" else match[3].split("/")[0]
        if groups[chapter] >= 2:
            continue
        native = snapshot(repo, REVISION, path).decode()
        if re.search(r"\bcopyright\b|©|third.party licen[sc]e", native, re.I):
            continue
        for block in items(native.splitlines()):
            text = " ".join(item["text"] for item in block)
            parts = fragments(text)
            if seen & parts:
                continue
            group = text_hash(path)
            targets = [i for i in range(1, WIDTH - 1) if block[i]["text"][-1] in ".?!"]
            if not targets:
                continue
            choice = targets[int(group[:8], 16) % len(targets)]
            works.append({"id": group, "path": path, "file_sha256": text_hash(native), "chapter": chapter,
                          "target": choice, "items": [dict(item) for item in block]})
            groups[chapter] += 1
            seen.update(parts)
            break
        if len(works) == MAX_FRESH:
            break
    chapters = sorted(groups, key=lambda value: text_hash(SCHEMA + "/split/" + value))
    available = max(4, math.ceil(len(chapters) * 2 / 5))
    calibration = set(chapters[:available])
    for work in works:
        work["split"] = "calibration" if work["chapter"] in calibration else "held"
    counts = Counter(work["split"] for work in works)
    save(output, {"schema": SCHEMA, "rules": RULES, "source": SOURCE, "revision": REVISION, "license": LICENSE,
                  "license_sha256": terms, "known_sources": [file_hash(path) for path in previous],
                  "implementation": file_hash(Path(__file__)), "works": works})
    print("Reserved", dict(counts), "across", len(chapters), "chapters", flush=True)
    if min(counts["held"], counts["calibration"]) < 7:
        raise ValueError("The licensed source has too few distinct held or ordinary list files")


def generate(source, family, output):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS

    _, revision, expected, folder = GENERATORS[family]
    if file_hash(folder / "model.safetensors") != expected:
        raise ValueError("The local editor does not match its verified release")
    model, tokenizer = load(str(folder), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    mx.set_cache_limit(96 * 1024 * 1024)
    mx.random.seed(BASE_SEED)
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    text = read(source)
    if text["implementation"] != file_hash(Path(__file__)) or text["rules"] != RULES:
        raise ValueError("The source reservation differs from the frozen implementation")
    works = [work for work in text["works"] if work["split"] == "held"]
    prompts = []
    for work in works:
        index, values = work["target"], work["items"]
        content = (f"{PROMPT}\n\nPrevious item: {values[index - 1]['text']}\n"
                   f"Target: {values[index]['text']}\nFollowing item: {values[index + 1]['text']}")
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                add_generation_prompt=True)
        prompts.append(tokenizer.encode(rendered, add_special_tokens=False))
    replies = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i + 4]
        first = batch_generate(model, tokenizer, batch, max_tokens=105, sampler=make_sampler(temp=0.0)).texts
        second = batch_generate(model, tokenizer, batch, max_tokens=105, sampler=make_sampler(temp=.65, top_p=.9)).texts
        replies.extend([pair[0], pair[1]] for pair in zip(first, second, strict=True))
    save(output, {"schema": SCHEMA, "family": family, "revision": revision, "weights_sha256": expected,
                  "source_sha256": file_hash(source), "replies": {work["id"]: pair for work, pair in zip(works, replies, strict=True)}})


def admitted(item, alternatives):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.text_elife_rewrites_v273 import CONTACT, WRAPPER
    from scripts.writing_blind_rewrite_probe import distance, words

    if len(alternatives) != 2:
        raise ValueError("Both fixed generation attempts are required")
    for raw in alternatives:
        value = " ".join(raw.strip().split())
        if (value and not WRAPPER.match(value) and not CONTACT.search(value) and not re.search(r"[\r\n]|```", raw.strip())
                and value[-1] in ".?!" and MIN_WORDS <= len(words(value)) <= MAX_WORDS
                and .60 <= len(words(value)) / len(words(item)) <= 1.52 and .12 <= distance(item, value) <= .78
                and len(segment(from_string(value))) == 1):
            return value
    return None


def input_case(work, name, replacement=None):
    from bisect import bisect_right

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_native_locations import clip_source_ranges
    from scripts.writing_multi_passage_panel import _prefix

    lines, focus, widths = [], [], []
    cursor = 0
    for index, item in enumerate(work["items"]):
        value = replacement if index == work["target"] and replacement is not None else item["text"]
        prefix = f"{index + 1}) " if name == "restyled" else item["prefix"]
        line = prefix + value
        lines.append(line)
        focus.append([cursor + len(prefix), cursor + len(line)])
        widths.append(len(re.findall(r"\b\w+\b", value)))
        cursor += len(line) + 1
    raw = "\n".join(lines)
    doc = from_string(raw)
    if len(doc.source_map) != 1:
        raise ValueError("Expected one native prose block for consecutive source lines")
    block = doc.source_map[0]
    ranges = block["source_ranges"]
    source_edges = {lo: (a, b, lo, hi) for a, b, lo, hi in ranges}
    keys = sorted(source_edges)

    def point(boundary):
        a, b, lo, hi = source_edges[keys[bisect_right(keys, boundary) - 1]]
        if boundary == lo:
            return a
        if boundary == hi:
            return b
        if not lo <= boundary <= hi or b - a != hi - lo:
            raise ValueError("A list-item boundary cannot be mapped to the original text")
        return a + boundary - lo

    source = [(point(start), point(end)) for start, end in focus]
    clipped = [clip_source_ranges(ranges, a, b) for a, b in source]
    if any(where != [witness] for where, witness in zip(clipped, focus, strict=True)):
        raise ValueError("A scored list item does not map directly to its original characters")
    prefix = _prefix(doc.text)
    native = [[item.start, item.end, prefix[item.start], prefix[item.end], item.paragraph, item.n_words] for item in segment(doc)]
    return {"name": name, "text": doc.text, "raw_sha256": text_hash(raw), "native_text_sha256": text_hash(doc.text),
            "sentences": native, "items": [[a, b, prefix[a], prefix[b], *focus[i], widths[i]] for i, (a, b) in enumerate(source)]}


def prepare(source, output, transcripts):
    from scripts.text_elife_rewrites_v273 import GENERATORS

    original = read(source)
    roster = [work for work in original["works"] if work["split"] == "held"]
    generated = {}
    for path in transcripts:
        receipt = read(path)
        family = receipt["family"]
        _, revision, expected, _ = GENERATORS[family]
        if (receipt["revision"] != revision or receipt["weights_sha256"] != expected or receipt["schema"] != SCHEMA
                or receipt["source_sha256"] != file_hash(source) or set(receipt["replies"]) != {work["id"] for work in roster}
                or family in generated):
            raise ValueError("A generation record differs from its exact held source set")
        generated[family] = receipt["replies"]
    if generated.keys() != GENERATORS.keys() or original["implementation"] != file_hash(Path(__file__)):
        raise ValueError("Use the complete frozen source and generation records")
    works, rejected = [], Counter()
    for work in original["works"]:
        cases = [input_case(work, "ordinary")]
        if work["split"] == "held":
            cases.append(input_case(work, "restyled"))
            for family, values in generated.items():
                if value := admitted(work["items"][work["target"]]["text"], values[work["id"]]):
                    cases.append(input_case(work, family, replacement=value))
                else:
                    rejected[family] += 1
        works.append({key: value for key, value in work.items() if key != "items"} | {"cases": cases,
                      "lines": [item["line"] for item in work["items"]]})
    save(output, {key: value for key, value in original.items() if key != "works"} | {"works": works,
                  "source_sha256": file_hash(source), "generator_binding": [file_hash(path) for path in transcripts],
                  "unadmitted": dict(rejected)})
    print("Validated standalone cases", dict(Counter(case["name"] for row in works for case in row["cases"])),
          "unadmitted", dict(rejected), flush=True)


def logit(value):
    p = max(.000001, min(.999999, value))
    return math.log(p / (1 - p))


def candidates(case):
    readings = [logit(value) for value in case["item_scores"]]
    lengths = [row[-1] for row in case["items"]]
    ranks = []
    for i, value in enumerate(readings):
        peers = sorted((abs(n - lengths[i]), j) for j, n in enumerate(lengths) if j != i)[:3]
        ranks.append((value, value - median(readings[j] for _, j in peers)))
    return {name: max((round(row[i], 6), -index) for index, row in enumerate(ranks)) for i, name in enumerate(ARMS)}


def read_cases(input_plan, snapshot, output, split):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from scripts.audio_strong_traffic_view import network_free
    from scripts.writing_entropy_prominence import existing
    from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
    from scripts.writing_multi_passage_panel import _inputs
    from scripts.writing_word_edit_transfer import model_assets

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    network_free()
    model_assets(snapshot)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    original = read(input_plan)
    if original["rules"] != RULES or original["implementation"] != file_hash(Path(__file__)):
        raise ValueError("Use the exact frozen scoring implementation and geometry")
    works = [work for work in original["works"] if work["split"] == split]
    waiting = set()
    for work in works:
        for row in work["cases"]:
            isolated, context = _inputs(row)
            waiting.update((*isolated, *context, *(row["text"][a:b] for a, b, *_ in row["items"])))
    inputs = sorted(waiting, key=lambda text: (len(text), text_hash(text)))
    values = classifier.predict(inputs, batch=12, check=lambda _: None)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite checkpoint response")
    readings = dict(zip(inputs, (round(float(value), 8) for value in values), strict=True))
    results = []
    for work in works:
        scored = []
        for row in work["cases"]:
            isolated, context = _inputs(row)
            scores = {"isolated": [readings[token] for token in isolated], "context": [readings[token] for token in context]}
            spans, assessed = existing(row, row | scores) if isolated else ([], False)
            proposed = [[row["sentences"][a][2], row["sentences"][b - 1][3]] for a, b in spans]
            scored.append({key: value for key, value in row.items() if key != "text"} | {"incumbent": proposed,
                           "incumbent_assessed": assessed, "item_scores": [readings[row["text"][a:b]] for a, b, *_ in row["items"]]})
        results.append(work | {"cases": scored})
    save(output, {"schema": SCHEMA, "split": split, "input_plan_sha256": file_hash(input_plan), "unique_inputs": len(inputs),
                  "revision": BASELINE_REVISION, "assets": BASELINE_ASSETS, "sources": results})


def points(ordinary):
    def quantile(values):
        return sorted(values)[math.ceil(len(values) * POINT) - 1]

    readings = [candidates(work["cases"][0]) for work in ordinary]
    return {key: quantile([row[key][0] for row in readings]) for key in ARMS}


def overlaps(proposals, target):
    a, b = target
    return any((intersection := max(0, min(b, y) - max(a, x))) >= .5 * (b - a)
               and intersection >= .30 * (max(b, y) - min(a, x)) for x, y in proposals)


def summarize(rows, cutoffs):
    outcome = {}
    for arm in ("incumbent", *ARMS):
        counts = Counter()
        for work in rows:
            target = work["target"]
            controls = []
            location = False
            for row in work["cases"]:
                role = row["name"]
                chosen = list(row["incumbent"])
                if arm != "incumbent":
                    score, index = candidates(row)[arm]
                    if score > cutoffs[arm]:
                        chosen.append(row["items"][-index][2:4])
                hit = overlaps(chosen, row["items"][target][2:4])
                already = overlaps(row["incumbent"], row["items"][target][2:4])
                counts.update({f"{role}_delivered": 1, f"{role}_alerts": bool(chosen),
                               f"{role}_assessed_by_incumbent": row["incumbent_assessed"]})
                if role in {"ordinary", "restyled"}:
                    controls.append(bool(chosen))
                    location |= hit
                else:
                    counts.update({f"{role}_located": hit, f"{role}_wrong_location": bool(chosen) and not hit,
                                   f"{role}_added_with_clean_target": hit and not already and not location,
                                   f"{role}_same_position_control": hit and location})
            counts.update(ordinary_source_groups_flagged=any(controls))
        outcome[arm] = dict(counts)
    return outcome


def calibrate(plan, snapshot, output):
    scratch = output.with_name("calibration-scores.json.gz")
    read_cases(plan, snapshot, scratch, "calibration")
    ordinary = read(scratch)
    save(output, ordinary | {"rules": RULES, "points": points(ordinary["sources"]), "implementation": file_hash(Path(__file__))})


def evaluate(plan, snapshot, calibration, output):
    seal = read(calibration)
    if (seal["input_plan_sha256"] != file_hash(plan) or seal["points"] != points(seal["sources"])
            or seal["implementation"] != file_hash(Path(__file__)) or seal["rules"] != RULES):
        raise ValueError("Held scoring requires the preregistered matching calibration")
    scratch = output.with_name("held-scores.json.gz")
    read_cases(plan, snapshot, scratch, "held")
    held = read(scratch)
    original = read(plan)
    save(output, {key: value for key, value in original.items() if key != "works"} | {"calibration": seal, "held": held,
                 "outcome": summarize(held["sources"], seal["points"])})
    review(output)


def review(path):
    result = read(path)
    seal, held = result["calibration"], result["held"]
    if (result["rules"] != RULES or seal["input_plan_sha256"] != held["input_plan_sha256"]
            or any(seal[key] != held[key] for key in ("assets", "revision"))
            or {work["chapter"] for work in seal["sources"]} & {work["chapter"] for work in held["sources"]}
            or seal["points"] != points(seal["sources"]) or result["outcome"] != summarize(held["sources"], seal["points"])):
        raise ValueError("The source groups, thresholds or outcome counts differ from the receipt")
    print(result["outcome"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest="action", required=True)
    for name, fields in (("reserve", ("repo", "output")), ("generate", ("source", "output")),
                         ("prepare", ("source", "output")), ("calibrate", ("plan", "snapshot", "output")),
                         ("evaluate", ("plan", "snapshot", "calibration", "output")), ("review", ("path",))):
        action = choices.add_parser(name)
        for field in fields:
            action.add_argument(field, type=Path)
        if name == "reserve":
            action.add_argument("previous", type=Path, nargs="+")
        if name == "prepare":
            action.add_argument("transcripts", type=Path, nargs="+")
        if name == "generate":
            action.add_argument("--family", choices=("qwen", "smol2"), required=True)
    args = vars(parser.parse_args())
    operation = args.pop("action")
    {"reserve": reserve, "generate": generate, "prepare": prepare, "calibrate": calibrate, "evaluate": evaluate,
     "review": review}[operation](**args)
