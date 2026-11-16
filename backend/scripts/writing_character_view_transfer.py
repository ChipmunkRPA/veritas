# SPDX-License-Identifier: MIT
"""Test exact source text and a limited Unicode reading on matched rewrites."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_function_sequence_transfer as prior

SCHEMA = "chipmunk.writing-character-view-transfer/1"
VARIANTS = ("baseline", "wrap_controls", "lookalikes", "both")
CHECKS = ("primary", "second")
QUOTA = 18
STRATA = tuple((genre, family) for genre in prior.GENRES for family in prior.FAMILIES)
REFERENCE = prior.PANEL / "writing-function-sequence-2026-11-15.json.gz"
TRANSFORM = {"a": "\u0430", "c": "\u0441", "e": "\u0435", "o": "\u03bf", "p": "\u0440", "x": "\u0445"}
BREAKS = ("\u00ad", "\u200b", "\u2060")
TOKEN = re.compile(r"\b[a-zA-Z]{6,}\b")
RULE = {
    "inputs": "Use held pairs of exact English original and one- or two-sentence rewritten excerpts. Keep source "
              "groups separate from every known exposed work and exclude normalized twelve-word overlap. Freeze "
              "the selected inputs before running either checkpoint.",
    "views": "Assign the same sentence slots on matched original and rewritten inputs. Apply three "
             "bounded formatting variants: invisible word breaks, look-alike letters and both on separate words. "
             "Keep every sentence and all visible punctuation and spelling unchanged after recovery. Retain "
             "checks on naturally occurring scripts and formatting that should remain unchanged.",
    "scoring": "Read every three-sentence interior context in order with two existing checkpoints. Compare the "
               "unaltered clean delivery, scores on formatting variants and scores on the adjusted input. Use "
               "one complete tokenizer window and freeze the ordinary .5 comparison point for both readings.",
    "location": "A review is useful if its focus is an entire labeled edited sentence. Count both touched "
                "sentences when the edit spans two. Count source groups with any off-target highlight, ordinary "
                "highlights, missed sources and score coverage separately. Exact word overlap is a second result.",
}
UNCHANGED = (
    "Anna kept the Greek term άνθρωπος and the Cyrillic word договор in the quoted text.",
    "Use this address https://exаmple.org/ and mail chаrles@example.com for the illustration.",
    "The Arabic joiner in مثال\u200cكتابي and the emoji 👨\u200d👩\u200d👧 stay in the source.",
    "Ordinary naïve wording has a non-breaking space\u00a0and a diacritic written as e\u0301.",
    "The paragraph contains a directional control \u2067בחינה\u2069 and a visible hyphen in well-known.",
    "A word boundary uses a zero-width space \u200b between two visible spaces.",
)


def sha(value):
    return prior.parent.sha(value)


def perturb(sentence, mode):
    words = list(TOKEN.finditer(sentence))
    changes = {}
    lookalikes = [(word.start() + i, TRANSFORM[ch]) for word in words
                  if (found := next(((i, ch) for i, ch in enumerate(word[0]) if ch in TRANSFORM), None)) is not None
                  for i, ch in [found]]
    if mode in {"lookalikes", "both"}:
        changes.update(lookalikes[:3 if mode == "lookalikes" else 2])
    if mode in {"wrap_controls", "both"}:
        eligible = [word for word in words if not any(word.start() <= offset < word.end() for offset in changes)]
        for word, mark in zip(eligible[:3 if mode == "wrap_controls" else 2], BREAKS):
            offset = word.start() + 2
            changes[offset] = mark + sentence[offset]
    return "".join(changes.get(i, ch) for i, ch in enumerate(sentence)), len(changes)


def prepare(role, source, targets):
    from app.detectors.text.character_view import derive
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    values = [source["text"][start:end] for start, end in source["spans"]]
    plain = " ".join(values)
    if plain != source["text"] or derive(plain)[1]:
        return None
    windows = {}
    for mode in VARIANTS:
        altered = list(values)
        if mode != "baseline":
            for index in targets:
                altered[index], count = perturb(values[index], mode)
                if count < 2 or derive(altered[index])[0] != values[index]:
                    return None
        text = " ".join(altered)
        units = segment(from_string(text))
        if [unit.text for unit in units] != altered:
            return None
        rows = []
        for i in range(1, len(units) - 1):
            context = units[i - 1:i + 2]
            adjusted = [derive(unit.text) for unit in context]
            native = " ".join(unit.text for unit in context)
            normalized = " ".join(view for view, _ in adjusted)
            if normalized != " ".join(values[i - 1:i + 2]):
                return None
            rows.append({"index": i, "focus": [units[i].start, units[i].end],
                         "spans": [[unit.start, unit.end] for unit in context], "native": native, "adjusted": normalized,
                         "focus_changes": [units[i].start + offset for offset, _ in adjusted[1][1]]})
        windows[mode] = {"sha256": sha(text), "chars": len(text), "windows": rows}
    return {"role": role, "deliveries": windows, "source_sha256": source["sha256"]}


def usable(extract, versions, allowed):
    try:
        yield from extract(versions, allowed)
    except (KeyError, ValueError):
        return  # Publisher annotations with inconsistent positions are outside the test scope.


def freeze(shards, parent_path, output, reference):
    from app.detectors.text import character_view
    from scripts import writing_ratio_fusion_transfer as native

    parent = prior.prior_plan(parent_path)
    previous = prior.read(REFERENCE)
    if reference:
        history_files = prior.editorial.exposure_paths(prior.read(reference)["intake"]["exposures"])
    else:
        history_files = [path for path in prior.editorial.exposure_files(None)
                         if not path.name.startswith("writing-character-view-")]
    exposure, known = prior.editorial.exposure_snapshot(history_files)
    if reference:
        prior.require(exposure == prior.read(reference)["intake"]["exposures"], "Exposure snapshot differs")
    source_rows = parent["fit"] + parent["calibration"] + parent["wiki_fit"] + [row for rows in parent["held"].values() for row in rows]
    known.update(row["group"] for row in (*source_rows, *previous["held"]))
    fingerprints = native.adjacent.intake.fingerprints
    terms = {term for row in source_rows for term in fingerprints(" ".join(s["text"] for s in row["sentences"]))}
    shards_list = prior.sources()
    terms.update(prior.inventory(shards, shards_list, known, fingerprints))
    counts, rows = Counter(), []
    accepted_slots = defaultdict(Counter)
    for genre, family, pin in shards_list:
        group_key = genre, family
        if counts[group_key] >= QUOTA:
            continue
        work = native.load_shard(shards, genre, family, pin)
        for key in sorted(work, key=lambda value: sha(f"{SCHEMA}/{genre}/{family}/{value}")):
            if counts[group_key] == QUOTA:
                break
            group, versions = sha(key), work[key]
            if group in known or "v0" not in versions:
                continue
            marks = set(fingerprints(versions["v0"]["text"]))
            if not marks or marks & terms:
                continue
            allowed = {slot for slot in prior.SINGLES | prior.DOUBLES if accepted_slots[group_key][slot] < 5}
            for targets, stage, before, after in usable(native.local_examples, versions, allowed):
                pair = [prepare(role, item, targets) for role, item in (("original", before), ("edited", after))]
                if any(item is None for item in pair):
                    continue
                words = [[a, b, index] for a, b, index, _ in native.adjacent.words.items(after)]
                rows.append({"group": group, "genre": genre, "family": family, "partition": pin[0], "stage": stage,
                             "targets": list(targets), "words": words, "changed": after["changed"], "pair": pair})
                known.add(group)
                terms.update(marks | set(fingerprints(after["text"])))
                counts[group_key] += 1
                accepted_slots[group_key][targets] += 1
                break
        print(f"{genre}/{family} {pin[0]}: {counts[group_key]} qualified works", flush=True)
        del work
    prior.require(set(counts) == set(STRATA) and min(counts.values()) >= 12, "Too few independently selected works per genre and generator")
    for text in UNCHANGED:
        prior.require(character_view.derive(text) == (text, []), "An ordinary Unicode control was changed")
    receipt = {"schema": SCHEMA, "method": RULE, "normalizer": sha(Path(character_view.__file__).read_bytes()),
               "intake": {"shards": shards_list, "exposures": exposure, "parent_sha256": sha(parent_path.read_bytes()),
                          "ordinary_control_hashes": [sha(text) for text in UNCHANGED]}, "rows": rows}
    prior.save(output, receipt)


def score(frozen, primary, second, output):
    import torch
    from app.detectors.text.adapters.agreement import score_inputs
    from app.detectors.text.character_view import __file__ as normalizer_path
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.evidence.document_matched_context import one_complete_window
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS
    from scripts.writing_focus_v3 import BASELINE_ASSETS, digest
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    frozen_data = prior.read(frozen)
    prior.require(frozen_data["schema"] == SCHEMA and frozen_data["method"] == RULE
                  and frozen_data["normalizer"] == sha(Path(normalizer_path).read_bytes()), "The frozen implementation differs")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = sorted({row[key] for work in frozen_data["rows"] for pair in work["pair"] for view in pair["deliveries"].values()
                     for row in view["windows"] for key in ("native", "adjusted")}, key=sha)
    readings = {}
    for name, path, pins in (("primary", primary, BASELINE_ASSETS), ("second", second, SECONDARY_ASSETS)):
        prior.require(all(digest(path / key) == value for key, value in pins.items()), "Pinned writing checkpoint differs")
        if name == "primary":
            classifier = FourClassTextClassifier.from_snapshot(path, "cpu")
        else:
            cfg = json.loads((path / "detector-config.json").read_bytes())
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                                        use_safetensors=True).eval()
            parts = tokenizer, model, cfg["temperature"]
        readings[name] = {}
        for start in range(0, len(inputs), 64):
            batch = inputs[start:start + 64]
            if name == "primary":
                values = classifier.predict(batch, batch=4)
                coverage = classifier.last_coverage
            else:
                values, coverage = score_inputs(batch, parts, "cpu", check=lambda *_: None)
            for value, response, proof in zip(batch, values, coverage, strict=True):
                readings[name][sha(value)] = [float(response), proof["tokens"]] if one_complete_window(proof) else None
            if start == 0 or (start + 64) % 512 == 0:
                print(f"{name} {min(start + 64, len(inputs))}/{len(inputs)} exact inputs", flush=True)
        if name == "primary":
            del classifier
    rows = []
    for work in frozen_data["rows"]:
        pair = []
        for item in work["pair"]:
            deliveries = {}
            for variant, delivery in item["deliveries"].items():
                windows = [{key: value for key, value in row.items() if key not in {"native", "adjusted"}}
                           | {"inputs": [sha(row[key]) for key in ("native", "adjusted")]} for row in delivery["windows"]]
                deliveries[variant] = delivery | {"windows": windows}
            pair.append(item | {"deliveries": deliveries})
        rows.append(work | {"pair": pair})
    report = {**frozen_data, "plan_sha256": sha(frozen.read_bytes()), "rows": rows, "readings": readings}
    report["results"] = recount(report)
    prior.save(output, report)
    print(json.dumps(report["results"]["all"], indent=2), flush=True)


def recount(packet):
    prior.require(packet["schema"] == SCHEMA and packet["method"] == RULE, "The Unicode review receipt differs")
    works = set()
    result = defaultdict(lambda: defaultdict(Counter))
    for work in packet["rows"]:
        group = work["group"]
        targets = set(work["targets"])
        prior.require(group not in works and targets in [set(slot) for slot in prior.SINGLES | prior.DOUBLES], "Held source roles differ")
        works.add(group)
        original, edited = work["pair"]
        prior.require([original["role"], edited["role"]] == ["original", "edited"], "Invalid matched source roles")
        for variant in VARIANTS:
            for pair in (original, edited):
                windows = pair["deliveries"][variant]["windows"]
                clean = pair["deliveries"]["baseline"]["windows"]
                prior.require(len(windows) == len(clean), "Source window count differs")
                for row, reference in zip(windows, clean, strict=True):
                    start, end = row["focus"]
                    prior.require(row["index"] == reference["index"] and row["inputs"][1] == reference["inputs"][0]
                                  and row["focus"] == row["spans"][1]
                                  and all(start <= i < end for i in row["focus_changes"]), "Geometry or canonical input differs")
            for strategy in ("native_primary", "native_both", "adjusted_primary", "adjusted_both"):
                outcomes = []
                for pair in (original, edited):
                    marked = set()
                    assessed = set()
                    for row in pair["deliveries"][variant]["windows"]:
                        key = row["inputs"][0 if strategy.startswith("native") else 1]
                        checks = CHECKS if strategy.endswith("both") else CHECKS[:1]
                        values = [packet["readings"][check][key] for check in checks]
                        if any(value is None for value in values):
                            continue
                        prior.require(all(math.isfinite(value[0]) and 0 <= value[0] <= 1 and 0 < value[1] <= 512
                                          for value in values), "Bad checkpoint response")
                        assessed.add(row["index"])
                        if all(value[0] >= .5 for value in values):
                            marked.add(row["index"])
                    outcomes.append((marked, assessed))
                (ordinary, ordinary_assessed), (marked, assessed) = outcomes
                words = {index for index, word in enumerate(work["words"]) if word[2] in marked}
                changed = set(work["changed"])
                overlap = len(words & changed)
                for layer in ("all", f"{work['genre']}/{work['family']}"):
                    result[layer][f"{variant}/{strategy}"].update(works=1, useful=bool(marked & targets),
                        exact=marked == targets, additional_wrong=bool(marked - targets), missed=not marked & targets,
                        assessed=bool(assessed), target_assessed=targets <= assessed, original_assessed=bool(ordinary_assessed),
                        ordinary_flags=bool(ordinary), useful_words=2*overlap >= len(changed) and 10*overlap >= 3*len(words | changed))
    return {tag: {key: dict(counts) for key, counts in outcomes.items()} for tag, outcomes in result.items()}


def affected_passages(packet):
    """Count response directions only on the exact altered sentence contexts."""
    outcomes = defaultdict(lambda: defaultdict(Counter))
    for work in packet["rows"]:
        for pair in work["pair"]:
            for variant in VARIANTS[1:]:
                states = []
                for row in pair["deliveries"][variant]["windows"]:
                    if not row["focus_changes"]:
                        continue
                    raw, adjusted = row["inputs"]
                    readings = [packet["readings"][check][index]
                                for check, index in (("primary", raw), ("primary", adjusted), ("second", adjusted))]
                    if any(value is None for value in readings):
                        states.append("unassessed")
                        continue
                    source, first, second = (value[0] >= .5 for value in readings)
                    if first != second:
                        states.append("different")
                    elif source == first:
                        states.append("consistent")
                    else:
                        states.append("toward_ai_like" if first else "away_from_ai_like")
                bucket = outcomes[variant][pair["role"]]
                bucket.update(works=1, windows=len(states))
                for state in set(states):
                    bucket.update({f"works_{state}": 1})
    return {variant: dict(arms) for variant, arms in outcomes.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="operation", required=True)
    freeze_cli = commands.add_parser("freeze")
    for field in ("shards", "parent", "output"):
        freeze_cli.add_argument(field, type=Path)
    freeze_cli.add_argument("--reference", type=Path)
    score_cli = commands.add_parser("score")
    for field in ("frozen", "primary", "second", "output"):
        score_cli.add_argument(field, type=Path)
    review_cli = commands.add_parser("review")
    review_cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.operation == "freeze":
        freeze(args.shards, args.parent, args.output, args.reference)
    elif args.operation == "score":
        score(args.frozen, args.primary, args.second, args.output)
    else:
        report = prior.read(args.receipt)
        result = recount(report)
        prior.require(report["results"] == result, "Held results differ on replay")
        print(json.dumps({"counts": result, "affected_passages": affected_passages(report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
