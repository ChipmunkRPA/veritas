# SPDX-License-Identifier: MIT
"""Test whether a sentence classifier's response can locate changed words."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts import writing_independent_range_transfer as parent

BASE = parent.previous.PANELS / "writing-independent-range-transfer-2026-11-15.json.gz"
SCHEMA = "chipmunk.writing-word-perturbation-transfer/1"
WIDTHS = (3, 6, 12, 18)
BATCH = 12
ARMS = ("delete", "mask", "both")
RULE = {
    "selection": "For each file nominate the single sentence selected by the previous independent absolute-rank rule. "
                 "Keep its original flag at each frozen point. Read the known edited sentence separately for diagnosis only.",
    "proposals": "Test contiguous spans of 3, 6, 12 and 18 source words. Use a stride of half the width or three words, "
                 "whichever is greater, including the final possible start. Keep at least four untouched words in the sentence.",
    "signals": "Measure the signed drop in the original logit after deleting the words or replacing all their pieces "
               "with mask tokens. Divide by the square root of the number of words. The joint arm requires both drops to "
               "be positive and ranks by the smaller normalized drop. A tie withholds the probe's range.",
    "flags": "A probe may replace the marked words only inside the already flagged sentence. Report unavailable "
             "and unchanged flags separately. Ordinary files remain flagged by the fixed sentence rule; also report "
             "which ordinary flags would receive a new word highlight and same-sentence matched alerts.",
    "evaluation": "Keep source and coordinate commitments. Measure the best alternative even with the correct sentence "
                  "supplied as a diagnostic, then evaluate the actual fixed sentence nomination separately. No new model fitting.",
}


def offsets(count):
    for width in WIDTHS:
        if count - width < 4:
            continue
        end = count - width
        for start in sorted(set(range(0, end + 1, max(3, width // 2))) | {end}):
            yield start, start + width


def mask_pieces(sentence, encoded, a, b, mask_id):
    ids = list(encoded.ids)
    touched = False
    for i, (left, right) in enumerate(encoded.offsets):
        if left == right or right <= a or b <= left:
            continue
        if (left < a and sentence[left:a].strip()) or (b < right and sentence[b:right].strip()):
            return None
        ids[i] = mask_id
        touched = True
    return ids if touched else None


def inference(model, inputs, pad):
    import numpy as np

    width = max(map(len, inputs))
    ids = np.full((len(inputs), width), pad, dtype=np.int64)
    valid = np.zeros_like(ids)
    for i, item in enumerate(inputs):
        ids[i, :len(item)] = item
        valid[i, :len(item)] = 1
    [logits] = model.run(["logits"], {"input_ids": ids, "attention_mask": valid})
    return [round(float(ai) - float(human), 7) for human, ai in logits]


def capture_sentence(model, tokenizer, row, index, words, expected, *, batch=BATCH):
    lower, upper = row["spans"][index]
    sentence = row["text"][lower:upper]
    encoded = tokenizer.encode(sentence)
    pad, mask_id = (tokenizer.token_to_id(special) for special in ("[PAD]", "[MASK]"))
    require = parent.previous.require
    require(pad is not None and mask_id is not None and 2 < len(encoded.ids) <= 512, "Token geometry differs")
    require(abs(inference(model, [encoded.ids], pad)[0] - expected) <= 0.000001, "Original sentence response changed")
    variants, slots = [], []
    locations = [[i, a, b] for i, (a, b, section, _) in enumerate(words) if section == index]
    output = []
    for start, stop in offsets(len(locations)):
        first, a, _ = locations[start]
        last, _, b = locations[stop - 1]
        left, right = a - lower, b - lower
        deleted = " ".join(part for part in (sentence[:left].rstrip(), sentence[right:].lstrip()) if part)
        masked = mask_pieces(sentence, encoded, left, right, mask_id)
        output.append([first, last + 1, None, None])
        for col, ids in enumerate((tokenizer.encode(deleted).ids, masked), 2):
            if ids is not None:
                variants.append(ids)
                slots.append((len(output) - 1, col))
    drift = 0.0
    batch_size = max(1, batch)
    for start in range(0, len(variants), batch_size):
        partition = variants[start:start + batch_size]
        if batch:
            anchor, *altered = inference(model, [encoded.ids, *partition], pad)
        else:
            anchor, altered = expected, inference(model, partition, pad)
        drift = max(drift, abs(anchor - expected))
        for value, (row_index, column) in zip(altered, slots[start:start + batch_size], strict=True):
            output[row_index][column] = round(anchor - value, 7)
    return {"words": locations, "responses": output, "largest_anchor_drift": round(drift, 7)}


def benchmark():
    baseline = parent.previous.read(BASE)
    parent.previous.require(parent.evaluate(baseline)["outcomes"] == baseline["outcomes"], "Sentence study differs")
    calibration, panels, points, _ = parent.reference()
    refs, limits = parent.calibrate(calibration, points, baseline["readings"])
    cases = {}
    for panel, (genre, rows) in panels.items():
        for row in rows:
            chosen = parent.winners(row, baseline["readings"], refs[genre])["independent"]
            cases[row["sha256"]] = panel, row, chosen, limits[genre]["independent"]
    return baseline, cases


def nominations(row, chosen, limits):
    real = chosen["sentence"] if chosen and chosen["score"] > limits["25pct"] else None
    diagnostic = row["position"] if row["family"] not in parent.previous.ORDINARY else None
    return sorted({index for index in (real, diagnostic) if index is not None})


def audit_sample(cases):
    ranked = defaultdict(list)
    for sha, (panel, row, chosen, limits) in cases.items():
        for index in nominations(row, chosen, limits):
            order = hashlib.sha256(f"{SCHEMA}/batch/{sha}/{index}".encode()).digest()
            ranked[panel].append((order, sha, index))
    return [(panel, sha, index) for panel, entries in ranked.items() for _, sha, index in sorted(entries)[:2]]


def reading_digest(reading):
    return hashlib.sha256(json.dumps(reading, sort_keys=True).encode()).hexdigest()


def capture(args):
    baseline, cases = benchmark()
    native = parent.native(args)
    model, tokenizer, runtime = parent.load_classifier(args.assets)
    jobs = [(sha, row, needed) for sha, (_panel, row, chosen, limits) in cases.items()
            if (needed := nominations(row, chosen, limits))]

    def score(job):
        sha, row, needed = job
        source = native[sha]
        parent.previous.require(parent.previous.pool.sha(source["text"]) == sha and source["spans"] == row["spans"],
                                "A source passage or range differs from the frozen nomination")
        words = parent.previous.words.items(source)
        output = {}
        for index in needed:
            reading = capture_sentence(model, tokenizer, source, index, words, baseline["readings"][sha][index - 1])
            validate(row, index, reading)
            output[str(index)] = reading
        return sha, output

    measured = {}
    journal = args.output.with_suffix(".native-readings.jsonl")
    with journal.open("x", buffering=1) as stored, ThreadPoolExecutor(max_workers=3) as worker:
        for i, (sha, output) in enumerate(worker.map(score, jobs), 1):
            measured[sha] = output
            stored.write(json.dumps({"source": sha, "readings": output}, sort_keys=True) + "\n")
            if i % 25 == 0:
                print(f"Measured {i}/{len(jobs)} source excerpts.", flush=True)
    return {"schema": SCHEMA, "rule": RULE, "baseline_sha256": parent.previous.digest(BASE),
            "runtime": runtime, "readings": measured}


def audit_batches(args):
    baseline, cases = benchmark()
    native = parent.native(args)
    model, tokenizer, runtime = parent.load_classifier(args.assets)
    counts = Counter()
    records, largest_difference = [], 0.
    for panel, sha, index in audit_sample(cases):
        source = native[sha]
        parent.previous.require(parent.previous.pool.sha(source["text"]) == sha and source["spans"] == cases[sha][1]["spans"],
                                "An audited source passage or range differs")
        words = parent.previous.words.items(source)
        expected = baseline["readings"][sha][index - 1]
        batched = capture_sentence(model, tokenizer, source, index, words, expected)
        individual = capture_sentence(model, tokenizer, source, index, words, expected, batch=0)
        batched_spans, individual_spans = select(batched), select(individual)
        counts.update({arm: batched_spans[arm] == individual_spans[arm] for arm in ARMS})
        difference = max((abs(x - y) for a, b in zip(batched["responses"], individual["responses"], strict=True)
                          for x, y in zip(a[2:], b[2:], strict=True) if x is not None and y is not None), default=0.)
        largest_difference = max(largest_difference, difference)
        records.append({"panel": panel, "source": sha, "sentence": index, "batched": batched_spans,
                        "individual": individual_spans, "largest_drop_difference": round(difference, 7),
                        "batched_reading_sha256": reading_digest(batched)})
    return {"schema": "chipmunk.writing-word-perturbation-batch-audit/1", "runtime": runtime,
            "preserved_selections": dict(counts), "audited_sentences": len(records),
            "largest_drop_difference": round(largest_difference, 7), "records": records}


def validate(row, index, reading):
    require = parent.previous.require
    a, b = row["spans"][index]
    words = reading["words"]
    if words:
        require(all(a <= start < stop <= b for _, start, stop in words)
                and [i for i, _, _ in words] == list(range(words[0][0], words[-1][0] + 1))
                and all(left[2] <= right[1] for left, right in zip(words, words[1:])), "Source word bounds differ")
    if row["assessed"]:
        exact = [[i, a, b] for i, (a, b, section, *_) in enumerate(row["words"]) if section == index]
        require(words == exact, "Perturbation words do not match the prior exact-source receipt")
    geometry = [[words[start][0], words[stop - 1][0] + 1] for start, stop in offsets(len(words))]
    require([entry[:2] for entry in reading["responses"]] == geometry
            and all(len(entry) == 4 and math.isfinite(entry[2]) and (entry[3] is None or math.isfinite(entry[3]))
                    for entry in reading["responses"]) and math.isfinite(reading["largest_anchor_drift"]), "Invalid perturbation responses")


def select(reading):
    proposals = defaultdict(list)
    for a, b, delete, mask in reading["responses"]:
        for key, drop in zip(ARMS, (delete, mask, min(delete, mask) if mask is not None else None), strict=True):
            if drop is not None and drop > 0.:
                proposals[key].append((round(drop / math.sqrt(b - a), 7), a, b))
    output = {}
    for key in ARMS:
        values = sorted(proposals[key], reverse=True)
        output[key] = values[0][1:] if values and (len(values) == 1 or values[0][0] > values[1][0]) else None
    return output


def evaluate(document):
    _baseline, cases = benchmark()
    readings = document["readings"]
    expected = {sha: needed for sha, (_panel, row, chosen, limits) in cases.items() if (needed := nominations(row, chosen, limits))}
    parent.previous.require(document["schema"] == SCHEMA and document["rule"] == RULE
                            and document["baseline_sha256"] == parent.previous.digest(BASE)
                            and set(readings) == set(expected), "The source or sentence nomination differs")
    outcomes, oracle = defaultdict(Counter), defaultdict(Counter)
    uniform_comparisons = defaultdict(Counter)
    variants = Counter()
    ordinary_sentences = defaultdict(set)
    useful_locations = []
    for sha, (panel, row, chosen, limits) in cases.items():
        measured = readings.get(sha, {})
        parent.previous.require(set(measured) == {str(index) for index in expected.get(sha, [])}, "Sentence scope differs")
        selections = {}
        for key, reading in measured.items():
            validate(row, int(key), reading)
            variants.update(sentences=1, deletions=len(reading["responses"]), masked=sum(entry[3] is not None for entry in reading["responses"]))
            selections[int(key)] = select(reading)
        ordinary = row["family"] in parent.previous.ORDINARY
        changed = set(row["changed"])
        if not ordinary:
            at = row["position"]
            original = next((target["words"] for target in row["by_sentence"] if target["sentence"] == at), [])
            result = oracle[panel]
            grid = measured[str(at)]["responses"]
            result.update(files=1, old_useful=parent.previous.words.useful_words(set(original), changed),
                           grid_could_locate=any(parent.previous.words.useful_words(set(range(a, b)), changed)
                                                 for a, b, *_ in grid))
            for arm, span in selections[at].items():
                result.update({f"{arm}_assessed": span is not None,
                               f"{arm}_useful": bool(span and parent.previous.words.useful_words(set(range(*span)), changed))})
                if span:
                    width = span[1] - span[0]
                    alternatives = [(a, b) for a, b, _, mask in grid
                                    if b - a == width and (arm == "delete" or mask is not None)]
                    useful_alternatives = sum(parent.previous.words.useful_words(set(range(a, b)), changed)
                                              for a, b in alternatives)
                    uniform_comparisons[panel][arm] += Fraction(useful_alternatives, len(alternatives))
        for level, point in limits.items():
            if not chosen or chosen["score"] <= point:
                continue
            old = parent.previous.words.useful_words(set(chosen["words"]), changed)
            for arm, span in selections[chosen["sentence"]].items():
                useful = bool(span and parent.previous.words.useful_words(set(range(*span)), changed))
                results = outcomes[f"{panel}/{level}/{arm}/{'ordinary' if ordinary else 'rewrite'}"]
                results.update(flags=1, word_highlights=span is not None, old_useful=old, useful=useful,
                               gained=useful and not old, lost=old and not useful,
                               highlight_with_no_prior_words=span is not None and not chosen["words"],
                               on_changed_sentence=chosen["sentence"] == row["position"] and span is not None)
                case = panel, level, arm, row["group"]
                if ordinary and span is not None:
                    ordinary_sentences[case].add(chosen["sentence"])
                if useful:
                    useful_locations.append((case, chosen["sentence"], results))
    for case, index, results in useful_locations:
        results.update(useful_with_matched_highlight=index in ordinary_sentences[case],
                       useful_without_matched_highlight=index not in ordinary_sentences[case])
    comparison = {panel: {arm: round(float(value), 6) for arm, value in totals.items()}
                  for panel, totals in uniform_comparisons.items()}
    return {"outcomes": dict(outcomes), "known_sentence": dict(oracle), "variants": dict(variants),
            "equal_width_uniform_expected_locations": comparison}


def review_audit(receipt, audit):
    _, cases = benchmark()
    require = parent.previous.require
    records = audit["records"]
    require(audit["schema"] == "chipmunk.writing-word-perturbation-batch-audit/1" and audit["runtime"] == receipt["runtime"]
            and len(records) == audit["audited_sentences"]
            and [(r["panel"], r["source"], r["sentence"]) for r in records] == audit_sample(cases), "Batch audit scope differs")
    preserved = Counter()
    for row in records:
        raw = receipt["readings"][row["source"]][str(row["sentence"])]
        expected = {key: list(span) if span else None for key, span in select(raw).items()}
        proposed = {tuple(span[:2]) for span in raw["responses"]}
        require(row["batched_reading_sha256"] == reading_digest(raw) and expected == row["batched"]
                and set(row["individual"]) == set(ARMS) and all(value is None or tuple(value) in proposed for value in row["individual"].values())
                and math.isfinite(row["largest_drop_difference"]) and row["largest_drop_difference"] >= 0,
                "An audited word span or its baseline differs")
        preserved.update({arm: row["individual"][arm] == expected[arm] for arm in ARMS})
    require(dict(preserved) == audit["preserved_selections"]
            and max(row["largest_drop_difference"] for row in records) == audit["largest_drop_difference"],
            "Batch audit totals differ")
    return {key: audit[key] for key in ("audited_sentences", "preserved_selections", "largest_drop_difference")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "audit-batch"):
        measure = phases.add_parser(phase)
        for name in ("assets", "reports", "matched", "additional", "dated", "elife", "fresh", "gemini", "gpt"):
            measure.add_argument(f"--{name}", type=Path, required=True)
        measure.add_argument("output", type=Path)
    review = phases.add_parser("review")
    review.add_argument("receipt", type=Path)
    review.add_argument("--batch-audit", type=Path)
    args = parser.parse_args()
    if args.phase != "review" and args.output.exists():
        parser.error("Choose a new output path for native inference.")
    if args.phase == "capture":
        result = capture(args)
        result.update(evaluate(result))
        with args.output.open("xb") as output:
            output.write(gzip.compress(json.dumps(result, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
        args.output.with_suffix(".native-readings.jsonl").unlink()
    elif args.phase == "audit-batch":
        result = audit_batches(args)
        with args.output.open("x") as output:
            output.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    else:
        receipt = parent.previous.read(args.receipt)
        result = evaluate(receipt)
        parent.previous.require(all(receipt[key] == entry for key, entry in result.items()), "Saved word measurements differ")
        if args.batch_audit:
            result["batch_audit"] = review_audit(receipt, parent.previous.read(args.batch_audit))
        print(json.dumps(result, sort_keys=True, indent=2))
