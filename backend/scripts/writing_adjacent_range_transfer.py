# SPDX-License-Identifier: MIT
"""Compare isolated and adjacent-sentence proposals on mixed English writing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import writing_attention_attribution as attention
from scripts import writing_range_context_transfer as context

pool, words, intake = attention.pool, attention.words, attention.intake
require, save, read = attention.require, attention.save, attention.read
SCHEMA = "chipmunk.writing-adjacent-range-transfer/1"
PINS = {
    "news": {
        "gemini-2.5-flash": (90392750, "5b37fb18828212242edeae339c5842754cc9ace3e622ff1780528c1552c502c7"),
        "gpt-5.4": (41072403, "5f49aec079641f21d1a9f93fbc57a49104d305a42b76d69a9bff0aeee8595755"),
    },
    "reports": {
        "gemini-2.5-flash": (118055124, "c57651c39e71cf76f0838ebb5bb1b26ce6aae5c4f8c47f4028e6d5902b9e72eb"),
        "gpt-5.4": (54161021, "22b2836026175261cd4fb6c42b9dba7afd99a4f3302ace9807e70c316aa594f0"),
    },
}
SOURCE_REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
SOURCES = {genre: {family: [count, pin] for family, (count, pin) in pins.items()} for genre, pins in PINS.items()}
SINGLES = tuple((index, 1) for index in range(1, 5))
PAIRS = tuple((index, 2) for index in range(1, 4))
SLOTS = SINGLES + PAIRS
CALIBRATION = 36
CAP = 3
STAGES = ("v1", "v2", "v3")
ARMS = ("single", "weaker_member", "read_together")
METHOD = {
    "scope": "Six consecutive English sentences in one native paragraph. Require exactly one or two adjacent "
             "interior edits, with all other sentences byte-for-byte unchanged. Use exact publisher character "
             "tags, independently aligned words and verified source segmentation. Freeze one variant per work.",
    "sources": "Use the pinned publisher development partition. Select 36 ordinary works per genre, then "
               "at most three held works per slot and generator, excluding known work hashes and normalized "
               "12-word overlap. Keep shortfalls visible.",
    "score": "Compare isolated interior sentences, adding pairs scored by the weaker member or by reading "
             "both sentences together. All arms select at most one exact source range. Original text enters "
             "calibration and label comparison only, never individual edited inference.",
    "calibration": "Normalize each position and width by its ordinary median and 90th-percentile gap, "
                   "with a .25 floor. Set each whole-file point at the previous tenth and quarter ordinary "
                   "budgets using leave-one-work-out ordinary maxima, with ties withheld.",
    "outcomes": "Measure exact changed-sentence boundaries separately from useful change-word overlap. "
                "Report wrong locations, wider or narrower overlaps, missed changes, ordinary flags and "
                "matched-original collisions. These comparisons do not estimate authorship probability.",
}


def positions(publisher):
    sentences = json.loads(publisher["sentences"])
    labels = json.loads(publisher["sentence_labels"])
    require(len(sentences) == len(labels) and set(labels) <= {0, 1} and all(sentences), "Invalid sentence truth")
    raw, gold = pool.truth_text(publisher)
    spacer = next((sep for sep in (" ", "\n", "\n\n") if sep.join(sentences) == raw), None)
    require(spacer is not None, "Publisher sentence inventory does not reconstruct the original")
    start = 0
    spans = []
    for sentence in sentences:
        stop = start + len(sentence)
        spans.append((start, stop))
        start = stop + len(spacer)
    require({span for span, label in zip(spans, labels, strict=True) if label} == gold, "Sentence and character truth disagree")
    return raw, sentences, spans, labels


def native_slice(raw, spans):
    from app.detectors.text.extract import _normalise, from_string
    from app.detectors.text.segment import segment

    doc = from_string(raw[spans[0][0]:spans[-1][1]])
    units = segment(doc)
    if len(doc.paragraphs) != 1 or not 72 <= doc.meta["words"] <= 360 or len(units) != 6:
        return None
    if [unit.text for unit in units] != [_normalise(raw[a:b])[0] for a, b in spans] or any(u.n_words < 8 for u in units):
        return None
    return {"text": doc.text, "sha256": pool.sha(doc.text), "spans": [[unit.start, unit.end] for unit in units]}


def examples(versions, wanted, *, calibrating):
    raw, sentences, spans, labels = positions(versions["v0"])
    require(not any(labels), "An ordinary original contains edited text")
    if calibrating:
        for offset in range(len(spans) - 5):
            if original := native_slice(raw, spans[offset:offset + 6]):
                yield (), "v0", original, None
                return
        return
    for stage in STAGES:
        if stage not in versions:
            continue
        try:
            edited, later, newer, marks = positions(versions[stage])
        except ValueError:
            continue
        if len(later) != len(sentences):
            continue
        same = [a == b for a, b in zip(sentences, later, strict=True)]
        for offset in range(len(spans) - 5):
            goals = tuple(index for index, value in enumerate(marks[offset:offset + 6]) if value)
            if goals not in wanted or any(equal == bool(mark) for equal, mark in zip(
                    same[offset:offset + 6], marks[offset:offset + 6], strict=True)):
                continue
            original = native_slice(raw, spans[offset:offset + 6])
            revision = native_slice(edited, newer[offset:offset + 6])
            if original and revision:
                changed = set().union(*(words.edit_words(original, revision | {"position": i}) for i in goals))
                if all(any(item[2] == i and j in changed for j, item in enumerate(words.items(revision))) for i in goals):
                    yield goals, stage, original, revision | {"changed": sorted(changed)}


def shard(path, genre, family, size, digest):
    require(path.stat().st_size == size and pool.file_hash(path) == digest, "Publisher bytes differ from the frozen release")
    csv.field_size_limit(1024 * 1024)
    table = defaultdict(dict)
    fields = ("text", "text_tagged", "sentences", "sentence_labels", "ai_spans_char")
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            require((row["domain"], row["generator"], row["split"], row["subset"]) == (genre, family, "dev", "main"),
                    "Publisher source metadata differs")
            version, work = row["version"], row["id"]
            if version == "v0" or version in STAGES:
                require(version not in table[work], "Duplicate benchmark work version")
                table[work][version] = {field: row[field] for field in fields}
    return table


def load_tables(genre, folder):
    tables = {}
    for family, (size, pin) in PINS[genre].items():
        name = f"{genre}_{family}.csv"
        if folder is not None:
            tables[family] = shard(folder / name, genre, family, size, pin)
            continue
        base = "https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/resolve"
        with TemporaryDirectory(prefix="chipmunk-writing-range-source-") as scratch:
            path = Path(scratch) / name
            subprocess.run(["curl", "-fLsS", "--max-time", "120", "--max-filesize", str(size), "--retry", "2",
                            "-o", str(path), f"{base}/{SOURCE_REVISION}/default/dev/{name}?download=true"], check=True)
            tables[family] = shard(path, genre, family, size, pin)
    return tables


def freeze(folders, exposures=None):
    known, commitments = context.abstract_exclusions(exposures)
    cases = []
    admitted = set()
    permitted = {tuple(range(start, start + length)) for start, length in SLOTS}
    for genre in PINS:
        tables = load_tables(genre, folders[genre])
        used = {phrase for table in tables.values() for key, variants in table.items()
                if "v0" in variants and pool.sha(key) in known for phrase in intake.fingerprints(variants["v0"]["text"])}
        for role, table in (("calibration", {key: values for bank in tables.values() for key, values in bank.items()}), *tables.items()):
            counts = Counter()
            for key in sorted(table, key=lambda value: pool.sha(f"{SCHEMA}/{genre}/{role}/{value}")):
                group = pool.sha(key)
                if group in known or "v0" not in table[key]:
                    continue
                fingerprints = intake.fingerprints(table[key]["v0"]["text"])
                if not fingerprints or not fingerprints.isdisjoint(used) or not fingerprints.isdisjoint(admitted):
                    continue
                available = permitted - {goals for goals, count in counts.items() if count == CAP}
                try:
                    variant = next(examples(table[key], available, calibrating=role == "calibration"), None)
                except ValueError:
                    continue
                if variant is None:
                    continue
                goals, stage, ordinary, revised = variant
                case = {"group": group, "genre": genre, "family": role, "stage": stage, "target": list(goals)}
                cases.append(case | {"delivery": "original", "changed": [], **ordinary})
                if revised is not None:
                    cases.append(case | {"delivery": "edited", **revised})
                admitted.update(fingerprints)
                known.add(group)
                counts[goals] += 1
                complete = counts[()] == CALIBRATION if role == "calibration" else all(counts[slot] == CAP for slot in permitted)
                if complete:
                    break
            print(f"{genre}/{role}: {dict(sorted(counts.items()))}", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "sources": SOURCES, "source_revision": SOURCE_REVISION,
            "exposures": commitments, "rows": cases}


def measure(row, model, tokenizer):
    import numpy as np

    require(pool.sha(row["text"]) == row["sha256"], "Native text commitment differs")
    responses = []
    for start, count in SLOTS:
        lower, upper = row["spans"][start][0], row["spans"][start + count - 1][1]
        ids = tokenizer.encode(row["text"][lower:upper]).ids
        if not 2 < len(ids) <= 512:
            return []
        input_ids = np.array([ids], dtype=np.int64)
        logits, = model.run(["logits"], {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)})
        responses.append(round(float(logits[0, 1]) - float(logits[0, 0]), 7))
    return responses


def readings(values):
    scores = dict(zip(SLOTS, values, strict=True))
    return {
        "single": {slot: scores[slot] for slot in SINGLES},
        "weaker_member": {slot: min(scores[slot[0], 1], scores[slot[0] + 1, 1]) for slot in PAIRS},
        "read_together": {slot: scores[slot] for slot in PAIRS},
    }


def profile(controls, scored, *, omit=None):
    records = defaultdict(list)
    for row in controls:
        values = scored[row["sha256"]]
        if row["group"] == omit or not values:
            continue
        for arm, options in readings(values).items():
            for slot, value in options.items():
                records[arm, slot].append(value)
    profiles = {}
    for key, values in records.items():
        median = statistics.median(values)
        profiles[key] = median, max(.25, sorted(values)[math.ceil(.9 * len(values)) - 1] - median)
    require(len(profiles) == 10, "Ordinary calibration has incomplete ranges")
    return profiles


def proposals(values, profiles):
    if not values:
        return dict.fromkeys(ARMS, (None, None))
    normalized = {}
    for arm, scores in readings(values).items():
        normalized[arm] = {slot: (value - profiles[arm, slot][0]) / profiles[arm, slot][1] for slot, value in scores.items()}
    output = {}
    for arm in ARMS:
        candidates = sorted((value, slot) for slot, value in (normalized["single"] | normalized[arm]).items())
        output[arm] = candidates[-1] if candidates[-1][0] > candidates[-2][0] else (None, None)
    return output


def calibrate(controls, scored):
    require(len(controls) == CALIBRATION and sum(bool(scored[r["sha256"]]) for r in controls) >= 24,
            "Not enough independently grouped ordinary calibration excerpts")
    values = [
        proposals(scored[row["sha256"]], profile(controls, scored, omit=row["group"]))
        for row in controls
    ]
    points = {}
    for arm in ARMS:
        maxima = sorted((row[arm][0] for row in values if row[arm][0] is not None), reverse=True)
        require(len(maxima) >= len(controls) // 4 + 1, "Ordinary maxima do not cover both budgets")
        points[arm] = {point: maxima[len(controls) // divisor] for point, divisor in attention.LEVELS.items()}
    return points


def evaluate(rows, scored):
    output, points = defaultdict(Counter), {}
    for genre in PINS:
        control = [r for r in rows if r["genre"] == genre and r["family"] == "calibration"]
        held = [r for r in rows if r["genre"] == genre and r["family"] != "calibration"]
        points[genre] = calibrate(control, scored)
        scale = profile(control, scored)
        picks = {r["sha256"]: proposals(scored[r["sha256"]], scale) for r in held}
        originals = {r["group"]: picks[r["sha256"]] for r in held if r["delivery"] == "original"}
        for row in held:
            ordinary = row["delivery"] == "original"
            found = picks[row["sha256"]]
            for level in attention.LEVELS:
                for arm, (value, slot) in found.items():
                    point = points[genre][arm][level]
                    selected = set(range(slot[0], sum(slot))) if slot and value > point else set()
                    label = set(row["target"]) if not ordinary else set()
                    span = {i for i, item in enumerate(row["words"]) if item[2] in selected}
                    base_value, base_slot = found["single"]
                    base_flag = base_slot and base_value > points[genre]["single"][level]
                    base_exact = bool(label) and base_flag and label == {base_slot[0]}
                    original_value, original_slot = originals[row["group"]][arm]
                    exact = bool(label) and selected == label
                    overlap = bool(label & selected)
                    for lane in ("all", row["family"]):
                        output[f"{genre}/{lane}/{len(row['target'])}/{level}/{arm}/" +
                               ("ordinary" if ordinary else "edited")].update(
                            files=1, scored=bool(scored[row["sha256"]]), flags=bool(selected), exact=exact,
                            useful=words.useful_words(span, set(row["changed"])),
                            wider_or_partial=overlap and not exact,
                            wrong=bool(selected and label) and not overlap,
                            added_exact=exact and not base_exact, lost_exact=bool(base_exact) and not exact,
                            matched_collision=exact and original_slot == slot and original_value > point,
                            ordinary_added=ordinary and bool(selected) and not base_flag,
                            ordinary_cleared=ordinary and bool(base_flag) and not selected,
                        )
    return points, output


def public(row):
    return {key: value for key, value in row.items() if key != "text"} | {"words": [list(item[:3]) for item in words.items(row)]}


def capture(plan, assets):
    model, tokenizer, runtime = attention.independent.load_classifier(assets)
    require(runtime == "1.24.3", "The classifier runtime differs")
    require(plan["schema"] == SCHEMA and plan["method"] == METHOD and plan["sources"] == SOURCES
            and plan["source_revision"] == SOURCE_REVISION, "The pre-inference source plan differs")
    rows = plan["rows"]
    controls = [r for r in rows if r["family"] == "calibration"]
    scored = {r["sha256"]: measure(r, model, tokenizer) for r in controls}
    fixed_points = {genre: calibrate([r for r in controls if r["genre"] == genre], scored) for genre in PINS}
    print(f"Fixed both ordinary points on {len(controls)} files before reading the held files.", flush=True)
    for n, row in enumerate((r for r in rows if r["family"] != "calibration"), 1):
        scored[row["sha256"]] = measure(row, model, tokenizer)
        if n % 36 == 0:
            print(f"Measured {n} held files.", flush=True)
    summarized = list(map(public, rows))
    points, summary = evaluate(summarized, scored)
    require(points == fixed_points, "Held outcomes changed the ordinary points")
    return plan | {"rows": summarized, "model": {"revision": attention.independent.RELEASE, "runtime": runtime,
                                                "assets": attention.independent.ASSETS},
                   "readings": scored, "points": points, "summary": summary}


def review(receipt):
    require(receipt["schema"] == SCHEMA and receipt["method"] == METHOD and receipt["sources"] == SOURCES
            and receipt["source_revision"] == SOURCE_REVISION
            and receipt["model"] == {"revision": attention.independent.RELEASE, "runtime": "1.24.3",
                                     "assets": attention.independent.ASSETS}, "The frozen study differs")
    context.abstract_exclusions(receipt["exposures"])
    rows = receipt["rows"]
    scored = receipt["readings"]
    require(len(rows) == len(scored) and set(scored) == {row["sha256"] for row in rows}, "Missing or duplicate readings")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
        require(row["genre"] in PINS and len(row["spans"]) == 6 and
                all(0 <= a < b for a, b in row["spans"]) and
                all(b <= c for (_, b), (c, _) in zip(row["spans"], row["spans"][1:])),
                "A native sentence extent differs")
        require(all(0 <= pos < 6 and row["spans"][pos][0] <= a < b <= row["spans"][pos][1]
                    for a, b, pos in row["words"]) and
                all(b <= c for (_, b, _), (c, _, _) in zip(row["words"], row["words"][1:])) and
                all(0 <= i < len(row["words"]) and row["words"][i][2] in row["target"] for i in row["changed"]),
                "A word lies outside its source sentence")
        values = scored[row["sha256"]]
        require(not values or len(values) == len(SLOTS) and all(math.isfinite(v) for v in values),
                "An incomplete model reading was used for evidence")
    for group in groups.values():
        original, *later = group
        require(original["delivery"] == "original" and not original["changed"] and (
            original["family"] == "calibration" and not later and not original["target"] or
            len(later) == 1 and later[0]["delivery"] == "edited" and bool(later[0]["changed"]) and
            original["family"] in PINS[original["genre"]] and original["stage"] in STAGES and
            (original["target"][0], len(original["target"])) in SLOTS and
            original["target"] == list(range(original["target"][0], original["target"][-1] + 1)) and
            all(original[key] == later[0][key] for key in ("family", "genre", "stage", "target"))),
            "An ordinary work was confused with its matched edit")
    points, summary = evaluate(rows, scored)
    require(points == receipt["points"] and summary == receipt["summary"], "The published outcomes do not reproduce")
    return summary


def diagnose(rows, scored):
    """Use held truth only to separate limitations of the range shape and selection."""
    result = defaultdict(Counter)
    for genre in PINS:
        references = [r for r in rows if r["genre"] == genre and r["family"] == "calibration"]
        scale = profile(references, scored)
        for row in rows:
            if row["genre"] != genre or row["delivery"] != "edited":
                continue
            target = set(row["target"])
            oracle = {index for index, (_a, _b, s) in enumerate(row["words"]) if s in target}
            disposition = result[f"{genre}/{len(target)}"]
            disposition.update(edits=1, useful_full_truth=words.useful_words(oracle, set(row["changed"])))
            for arm, (_, slot) in proposals(scored[row["sha256"]], scale).items():
                disposition[f"highest_exact_{arm}"] += bool(slot and slot == (min(target), len(target)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    source = modes.add_parser("freeze")
    source.add_argument("output", type=Path)
    for name in ("news", "reports"):
        source.add_argument(f"--{name}", type=Path, help="Use locally downloaded shards for this genre.")
    source.add_argument("--reference", type=Path)
    native = modes.add_parser("capture")
    for name in ("output", "plan", "assets"):
        native.add_argument(name, type=Path)
    portable = modes.add_parser("review")
    portable.add_argument("receipt", type=Path)
    portable.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze":
        exposures = read(args.reference)["exposures"] if args.reference else None
        save(args.output, freeze({genre: getattr(args, genre) for genre in PINS}, exposures))
    elif args.mode == "capture":
        receipt = capture(read(args.plan), args.assets)
        review(receipt)
        save(args.output, receipt)
    else:
        receipt = read(args.receipt)
        summary = review(receipt)
        output = diagnose(receipt["rows"], receipt["readings"]) if args.diagnostic else summary
        print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
