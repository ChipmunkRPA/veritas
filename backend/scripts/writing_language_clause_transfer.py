# SPDX-License-Identifier: Apache-2.0
"""Compare exact within-sentence language witnesses on disjoint writing and book sources."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.writing_language_window_transfer import WORD, foreign_bank, read, sha, write

SCHEMA = "chipmunk.writing-language-clause-transfer/1"
DEVELOPMENT = "94eccaf39a4dc869272b29dd9d1bbc657401db2ab00bb8918f2fe7e4585ea5e2"
HELD = "45b3065d5ea5242a3ce52f89467074e4b9b7ed30f519c341b9b468de98f06491"
ROSTER = {
    "es": ("64465", "3676c97acd2e73557c8474572864d2e2414334fc1fdd531f68bc703b729ac9f5"),
    "fr": ("70891", "9fcc1f06668d953410c96b00fc5e421a3cc5498abdfce12be835420ef9a19879"),
    "de": ("74008", "464e776735b91a336d40fa2b96e6b9f88d20e20bbdadc59e3290cfc127e97907"),
    "it": ("36379", "72c9e0ce9975f7a6c9b11381fc0ef516129691819cefc0413962e1fc15161592"),
}
TITLES = {"es": "Novelas ejemplares y amorosas", "fr": "Notre-Dame de Paris",
          "de": "Der Schimmelreiter", "it": "La famiglia Bonifazio"}
HELD_CASES = ("clean", "short_title", "substantial_clause", "short_clause", "separate_sentence")
FOREIGN_CASES = {"clause", "substantial_clause", "short_clause", "separate_sentence"}
ROUTES = ("narrow", "paired")


def ngrams(text):
    words = WORD.findall(text.casefold())
    return set(zip(*(words[i:] for i in range(12)), strict=False))


def held_deliveries(row, ordinal, fragment, language):
    text = row["text"]
    assert sha(text) == row["sha256"]
    index = 1 + ((ordinal // 4 + ordinal) % (len(row["spans"]) - 2))
    start, end = row["spans"][index]
    words = list(WORD.finditer(text[start:end]))
    assert len(words) >= 4
    # Different source positions exercise all three portions of the sentence.
    internal = start + words[len(words) * (1 + ordinal % 3) // 4].start()
    short_words = list(WORD.finditer(fragment))
    width = 9 + (ordinal // 4 % 4)
    left = (len(short_words) - width) // 2
    short = fragment[short_words[left].start():short_words[left + width - 1].end()]
    name = f"“{TITLES[language]}”"
    for case, offset, inserted in (
        ("clean", 0, ""),
        ("short_title", internal, name),
        ("substantial_clause", internal, fragment[:-1]),
        ("short_clause", internal, short),
        ("separate_sentence", start, fragment),
    ):
        source = text if not inserted else text[:offset] + inserted + " " + text[offset:]
        yield {
            "group": row["group"], "role": row["delivery"], "genre": row["genre"], "case": case,
            "language": language, "source": source, "sha256": sha(source),
            "truth": [offset, offset + len(inserted)] if case in FOREIGN_CASES else None,
            "truth_sha256": sha(inserted) if case in FOREIGN_CASES else None,
        }


def freeze(development, held, books, destination, phase):
    from app.detectors.text.extract import from_string

    assert sha(development.read_bytes()) == DEVELOPMENT and sha(held.read_bytes()) == HELD
    earlier = read(development)["inputs"]
    held_rows = read(held)["rows"]
    dev_rows = [r for r in earlier if r["case"] == "clean"]
    dev_groups = {r["group"] for r in dev_rows}
    held_groups = sorted({r["group"] for r in held_rows})
    assert len(dev_groups) == 112 and len(held_groups) == 84 and not dev_groups.intersection(held_groups)
    dev_sequences = set().union(*(ngrams(r["source"]) for r in dev_rows))
    assert not any(ngrams(r["text"]) & dev_sequences for r in held_rows)
    if phase == "development":
        cases = [r for r in earlier if r["case"] in {"clean", "title", "clause"}]
        deliveries = [{k: r[k] for k in ("group", "role", "genre", "case", "language", "source", "sha256")} | {
            "truth": r["foreign_range"], "truth_sha256": (
                sha(r["source"][slice(*r["foreign_range"])]) if r["foreign_range"] else None),
        } for r in cases]
    else:
        bank = foreign_bank(books, ROSTER, count=21)
        languages = tuple(ROSTER)
        positions = {group: index for index, group in enumerate(held_groups)}
        deliveries = []
        for r in held_rows:
            ordinal = positions[r["group"]]
            language = languages[ordinal % len(languages)]
            fragment = bank[language][ordinal // len(languages)]
            deliveries.extend(held_deliveries(r, ordinal, fragment, language))
    for row in deliveries:
        if row["truth"]:
            assert sha(from_string(row["source"]).text[slice(*row["truth"])]) == row["truth_sha256"]
    write(destination, {"schema": SCHEMA, "phase": phase, "inputs": deliveries})


def measure(row):
    from app.detectors.text import language_clauses, language_scope
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string(row["source"])
    sentences = segment(doc)
    baseline = language_scope.classify(doc.text, sentences)
    matches = language_clauses.candidates(sentences, single=True) if baseline == "english" else dict.fromkeys(ROUTES)
    return {k: v for k, v in row.items() if k != "source"} | {
        "chars": len(doc.text), "baseline": baseline, "matches": matches,
    }


def matches_truth(proposal, truth, language):
    if proposal is None or truth is None:
        return False
    a, b = truth
    start, end = proposal["start"], proposal["end"]
    overlap = max(0, min(b, end) - max(a, start))
    return (proposal["language"] == language and overlap >= 0.5 * (end - start)
            and overlap >= 0.3 * (max(b, end) - min(a, start)))


def tally(rows):
    result = {}
    for case in sorted({r["case"] for r in rows}):
        counts = Counter()
        for row in (r for r in rows if r["case"] == case):
            counts["total"] += 1
            counts["baseline_english"] += row["baseline"] == "english"
            for route, proposal in row["matches"].items():
                counts[route + "_added_stops"] += proposal is not None
                counts[route + "_correct_locations"] += matches_truth(proposal, row["truth"], row["language"])
        result[case] = dict(counts)
    return result


def check(receipt):
    assert receipt["schema"] == SCHEMA
    expected = ("clean", "title", "clause") if receipt["phase"] == "development" else HELD_CASES
    groups = {r["group"] for r in receipt["rows"]}
    assert len(groups) == (112 if receipt["phase"] == "development" else 84)
    assert {(r["group"], r["role"], r["case"]) for r in receipt["rows"]} == {
        (group, role, case) for group in groups for role in ("original", "edited") for case in expected}
    assert len(receipt["rows"]) == len(groups) * 2 * len(expected)
    for row in receipt["rows"]:
        assert "source" not in row and bool(row["truth"]) == (row["case"] in FOREIGN_CASES)
        assert set(row["matches"]) == set(ROUTES)
        for route, proposal in row["matches"].items():
            if proposal is None:
                continue
            assert row["baseline"] == "english" and proposal["language"] != "en"
            assert 0 <= proposal["start"] < proposal["end"] <= row["chars"]
            if route == "paired":
                assert row["matches"]["narrow"] is not None
                assert 0 <= proposal["context_start"] <= proposal["start"]
                assert proposal["end"] <= proposal["context_end"] <= row["chars"]
    results = tally(receipt["rows"])
    assert results == receipt["results"]
    return results


def capture(frozen, output):
    plan = read(frozen)
    assert plan["schema"] == SCHEMA
    rows = []
    for r in plan["inputs"]:
        assert sha(r["source"]) == r["sha256"]
        rows.append(measure(r))
    receipt = {"schema": SCHEMA, "phase": plan["phase"], "freeze_sha256": sha(frozen.read_bytes()),
               "rows": rows, "results": tally(rows)}
    check(receipt)
    write(output, receipt)
    print(json.dumps(receipt["results"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    freeze_args = mode.add_parser("freeze")
    freeze_args.add_argument("phase", choices=("development", "held"))
    for name in ("development", "held", "books", "output"):
        freeze_args.add_argument(name, type=Path)
    capture_args = mode.add_parser("capture")
    for name in ("frozen", "output"):
        capture_args.add_argument(name, type=Path)
    mode.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.development, args.held, args.books, args.output, args.phase)
    elif args.mode == "capture":
        capture(args.frozen, args.output)
    else:
        print(json.dumps(check(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
