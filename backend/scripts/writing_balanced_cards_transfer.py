"""Test six-card writing evidence allocation on new source works."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import document_selection
from scripts import writing_sibling_paragraph_transfer as prior
from scripts.writing_context_pool_transfer import PANEL_DIR, file_hash, sha
from scripts.writing_editorial_worker_audit import covers
from scripts.writing_focus_inside_context import read, shard
from scripts.writing_interval_transfer import fingerprints
from scripts.writing_missed_passage_transfer import POINT
from scripts.writing_sentence_probe import SHA256, write

SCHEMA = "chipmunk.writing-balanced-cards/1"
SOURCES = (
    ("gpt-5.4-nano", "b1177c4cbda63ccd623dc903037aa8e3ac20b31fcb5e4fe8bcf93c32980f3045"),
    ("gemini-2.5-flash", "d4805b719907ea7e3e7215c0cbfd5d47e95f7bad394a51fcdc851d4bc029222e"),
)
TARGET_PER_FAMILY = 24
SECTION_SIZES = (3, 5, 3)
SOURCE_GAP = 80
SOURCE_BLOCKS = tuple(block for block, size in enumerate(SECTION_SIZES, 1) for _ in range(size))
METHOD = {
    "selection": "Retain disagreement reservation. In each selection step prefer the source block represented by the fewest cards. "
                 "Break ties in favor of a passage with classifier support, then by source order. Display at most six cards.",
    "comparison": "Use the same recorded direction and scope for both policies. Compare source blocks, unique source sentences, "
                  "retained disagreements, ordinary files with a highlight, and useful locations for the labeled rewrites.",
    "sources": "Use three disjoint excerpts of three, five and three sentences separated by at least 80 original characters. "
               "Use exactly one labeled interior rewrite in the central excerpt. Carry forward admitted pairs. "
               "Use older government reports as development context.",
}


def source_pins():
    return {path.name: file_hash(path) for path in sorted(PANEL_DIR.glob("writing-*-2026-11-14.json"))
            if not path.name.startswith("writing-balanced-cards-")}


def freeze(directory, output, *, previous):
    excluded = prior.source_exclusions(previous)
    rows, used = [], set()
    for generator, pin in SOURCES:
        table = shard(directory / f"essays_{generator}.csv", "essays", generator, pin)
        for source, versions in table.items():
            if sha(source) in excluded and "v0" in versions:
                used.update(fingerprints(versions["v0"]["text"]))
        positions = Counter()
        for source in sorted(table, key=lambda name: sha(f"{SCHEMA}/{generator}/{name}")):
            group = sha(source)
            if group in excluded or any(v not in table[source] for v in ("v0", *prior.pool.VARIANTS)):
                continue
            original = table[source]["v0"]["text"]
            if not 1200 <= len(original) <= 65000:
                continue
            phrases = fingerprints(original)
            if phrases & used:
                continue
            order = sorted((1, 2, 3), key=lambda position: (positions[position + SECTION_SIZES[0]], position))
            try:
                result = prior.pair(source, table[source], order, flank_size=SECTION_SIZES[0],
                                    separation=SOURCE_GAP, min_words=120)
            except ValueError:
                continue
            if result is None:
                continue
            offsets, cases = result
            positions[offsets["position"]] += 1
            for text, spans, variant in cases:
                rows.append(offsets | {"text": text, "spans": spans, "sha256": sha(text), "group": group, "role": "held",
                                       "truth": spans[offsets["position"]] if variant else None, "generator": generator,
                                       "spaces": [i for i, char in enumerate(text) if char.isspace()]})
            excluded.add(group)
            used.update(phrases)
            if positions.total() == TARGET_PER_FAMILY:
                break
        print(generator, positions.total(), flush=True)
        if not positions:
            raise ValueError("The essay source arm has no eligible works")
    write(output, {"schema": SCHEMA, "method": METHOD, "earlier_panels": previous, "publishers": SOURCES, "rows": rows})


def plans(row):
    spans = [tuple(pair) for pair in row["spans"]]
    candidates, disagreements, classifier, blocks, scopes = [], set(), set(), {}, {}
    for i, (span, prob, (ratio, _, expanded)) in enumerate(zip(spans, row["classifier"], row["ratio"], strict=True)):
        cls_flag, ratio_flag = prob >= .5, ratio is not None and ratio < POINT
        if not (cls_flag or ratio_flag):
            continue
        candidates.append(span)
        blocks[span] = SOURCE_BLOCKS[i]
        if (ratio_flag and not cls_flag) or (cls_flag and ratio is not None and not ratio_flag):
            disagreements.add(span)
        if cls_flag:
            classifier.add(span)
        scopes[span] = span if ratio_flag and not expanded else (spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1])
    return candidates, disagreements, classifier, blocks, scopes


def compare(rows):
    totals = defaultdict(Counter)
    for row in rows:
        candidates, disagreements, classifier, blocks, scopes = plans(row)
        truth, spaces = row["truth"], set(row["spaces"])
        bucket = f"{row['generator']}/{'rewritten' if truth else 'ordinary'}"
        located = {}
        for mode, mapping in (("source_order", {}), ("balanced", blocks)):
            selected, record = document_selection.select(candidates, disagreements, classifier, [], mapping)
            readings = [scopes[span] for span in selected]
            tally = totals[f"{bucket}/{mode}"]
            tally["total"] += 1
            tally["capped"] += int(len(candidates) > 6)
            tally["balanced"] += int(record is not None)
            tally["blocks_displayed"] += len({blocks[span] for span in selected})
            tally["disagreements_displayed"] += len(set(selected) & disagreements)
            tally["focus_ranges"] += len(selected)
            tally["sentences_in_measured_spans"] += sum(
                any(a <= x < y <= b for a, b in readings) for x, y in row["spans"]
            )
            located[mode] = covers(readings, truth, spaces) if truth else bool(readings)
            tally["usefully_located_or_flagged_ordinary"] += int(located[mode])
            tally["wrong_location_only"] += int(bool(truth and readings and not located[mode]))
        if truth:
            paired = totals[f"{bucket}/paired"]
            paired["gained"] += int(located["balanced"] and not located["source_order"])
            paired["lost"] += int(located["source_order"] and not located["balanced"])
    return dict(totals)


def analyze(roster, records, output):
    observed = {row["phase"]: row for row in map(read, records)}
    pin = file_hash(roster)
    if set(observed) != {"classifier", "ratio"} or any(row["roster_sha256"] != pin for row in observed.values()):
        raise ValueError("Both measurement phases must use the frozen source roster")
    data = read(roster)
    rows = [{key: value for key, value in row.items() if key != "text"}
            | {phase: record["scores"][row["sha256"]] for phase, record in observed.items()}
            for row in data["rows"]]
    receipt = data | {"rows": rows, "roster_sha256": pin, "outcomes": compare(rows)}
    review(receipt)
    write(output, receipt)


def review(receipt):
    if (receipt["schema"] != SCHEMA or receipt["method"] != METHOD
            or receipt["publishers"] != [list(source) for source in SOURCES]):
        raise ValueError("Unexpected study definition")
    earlier = prior.source_exclusions(receipt["earlier_panels"])
    cohorts = defaultdict(list)
    for row in receipt["rows"]:
        spans, position, anchors = row["spans"], row["position"], row["source_anchors"]
        count = len(SOURCE_BLOCKS)
        if (not SHA256.fullmatch(row["sha256"]) or not SHA256.fullmatch(row["group"]) or row["group"] in earlier
                or row["role"] != "held" or len(spans) != count or position not in (4, 5, 6)
                or row["truth"] not in (None, spans[position]) or any(not 0 <= a < b for a, b in spans)
                or len(anchors) != 3 or any(not 0 <= a < b for a, b in anchors)
                or any(a[1] + SOURCE_GAP > b[0] for a, b in zip(anchors, anchors[1:]))
                or len(row["classifier"]) != count or len(row["ratio"]) != count
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row["classifier"])
                or any(r is not None and (not math.isfinite(r) or r <= 0 or not n) for r, n, _ in row["ratio"])):
            raise ValueError("A source or measurement is missing or invalid")
        cohorts[row["group"]].append(row)
    if any(len(rows) != 2 or sum(row["truth"] is None for row in rows) != 1
           or any(rows[0][key] != rows[1][key] for key in ("generator", "position", "source_anchors"))
           for rows in cohorts.values()):
        raise ValueError("Held source pairs differ")
    outcomes = compare(receipt["rows"])
    if outcomes != receipt["outcomes"]:
        raise ValueError("The replay changed the held outcomes")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "analyze", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--earlier", type=Path)
    parser.add_argument("--parts", type=Path, nargs=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        previous = read(args.earlier)["earlier_panels"] if args.earlier else source_pins()
        freeze(args.shards, args.path, previous=previous)
    elif args.phase == "analyze":
        analyze(args.path, args.parts, args.output)
    else:
        print(json.dumps(review(read(args.path)), sort_keys=True))


if __name__ == "__main__":
    main()
