"""Test exact archive comparison after one changed word next to a lost paragraph break."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.evidence import document_archive as archive
from app.evidence import document_archive_layout as layout
from app.evidence.document_revision import tokens
from scripts.writing_archive_joined_layout_transfer import digest, fingerprint, read, save
from scripts.writing_reference_archive_transfer import VOLUMES, source_rows

SCHEMA = "chipmunk.writing-archive-changed-edge/1"
DESIGN = {
    "selection": "Up to four unused official-source works per pinned volume in hash order, distinct senders within the held panel and no earlier or cross-work exact 12-word overlap. Historical sender identities may recur.",
    "candidate": "Recover at least three consecutive source paragraphs of equal token counts. Every token matches except exactly one substitution within eight tokens of an internal paragraph edge. Require unique alignment and independent existing fingerprint support.",
    "qualification": "At least 12 distinct held source groups, at least 90% exact changed-word results in each direction, no incorrect source or changed token on controls, and all unchanged copies still match.",
    "scope": "Plain-text archive version comparison. No authorship contribution.",
}
TARGETS = {"before_break", "after_break"}


def edits(parts):
    """Freeze one earlier-token and one later-token alteration at different boundaries."""
    word_rows = [tokens(text, [(0, len(text))]) for text in parts]
    if not 3 <= len(parts) <= 24 or any(len(row) < 24 for row in word_rows):
        return None
    choices = []
    for index, tail in ((0, True), (2, False)):
        section = word_rows[index][-8:] if tail else word_rows[index][:8]
        target = next((row for row in section if row[0].isalpha() and len(row[0]) >= 4
                       and sum(row[0] == value for value, _, _ in word_rows[index]) == 1), None)
        if target is None:
            return None
        choices.append((index, target[1], target[2], "reconsidered" if target[0].casefold() != "reconsidered" else "adjusted"))
    return choices


def source_panel(parent, previous, catalog):
    history = [read(parent / f"{name}.json") for name in ("originals", "unknown", "earlier")]
    old_cases = read(previous / "freeze.json")["cases"]
    used = {item["group"] for rows in history for item in rows} | {case[0] for case in old_cases}
    known = set().union(*(fingerprint([p["text"] for p in catalog[group]["paragraphs"]]) for group in used if group in catalog))
    raw_volumes = {}
    for volume, pins in VOLUMES.items():
        path = parent / "raw" / f"{volume}.xml"
        assert digest(path) == pins[0]
        raw_volumes[volume] = source_rows(path, volume)
    works, seen_senders = [], set()
    for volume, raw in raw_volumes.items():
        admitted = 0
        ordered = sorted((group for group, row in catalog.items() if row["volume"] == volume),
                         key=lambda group: text_hash(f"changed-edge/1/{group}"))
        for group in ordered:
            if group in used:
                continue
            parts = catalog[group]["paragraphs"]
            origin = raw[catalog[group]["doc_id"]]
            sender = origin["sender_key"]
            text = [part["text"] for part in parts]
            signature = fingerprint(text)
            changes = edits(text)
            if sender in seen_senders or known & signature or changes is None or ">" in "".join(text):
                continue
            assert all(part["text"] == origin["paragraphs"][part["ordinal"]]["text"] for part in parts)
            assert all(b["ordinal"] == a["ordinal"] + 1 for a, b in zip(parts, parts[1:]))
            seen_senders.add(sender)
            known |= signature
            works.append({"source": group, "sender": sender, "parts": text, "ordinals": [p["ordinal"] + 1 for p in parts],
                          "edits": changes})
            admitted += 1
            if admitted == 4:
                break
    assert len(works) >= 12, len(works)
    return works


def replace(parts, edit, value=None):
    index, a, b, word = edit
    word = word if value is None else value
    copies = parts[:]
    copies[index] = parts[index][:a] + word + parts[index][b:]
    start = sum(len(part) + 1 for part in parts[:index]) + a
    return copies, [start, start + len(word)]


def fixtures(works):
    for i, work in enumerate(works):
        parts, group = work["parts"], work["source"]
        left, right = work["edits"]
        before, a = replace(parts, left)
        after, b = replace(parts, right)
        twice, _ = replace(before, right)
        insertion, _ = replace(parts, left, f"new wording {left[3]}")
        foreign = works[(i + 1) % len(works)]["parts"][1]
        variants = [
            ("original", parts, group, None),
            ("before_break", before, group, a),
            ("after_break", after, group, b),
            ("second_substitution", twice, None, None),
            ("inserted_words", insertion, None, None),
            ("wrong_order", [before[1], before[0], *before[2:]], None, None),
            ("other_source", [before[0], foreign, *before[2:]], None, None),
            ("prefixed", ["An editor added this separate note.", *before], None, None),
            ("quoted", ["> " + before[0], *before[1:]], None, None),
            ("ambiguous", before, None, None),
        ]
        for role, text, expected, truth in variants:
            yield {"group": group, "role": role, "text": " ".join(text), "expected": expected, "truth": truth}


def decision(case, catalog, index, edges):
    text, role = case["text"], case["role"]
    doc = from_string(text)
    assert doc.text == text and doc.paragraphs == [(0, len(text))]
    baseline = archive.compare(doc, text_hash(text), catalog, index)
    source = (baseline["reference"] or {}).get("id")
    aligned, changes = baseline["aligned_paragraphs"], baseline["changes"]
    prior = source
    if not source and ">" not in text:
        if role == "ambiguous":
            catalog = catalog | {case["group"] + "/duplicate": catalog[case["group"]]}
            index, edges = archive.build_index(catalog), layout.index_edges(catalog)
        recovered = layout.joined_spans_one_edge_change(text, doc.paragraphs[0], catalog, edges)
        if recovered is not None:
            key, first, spans = recovered
            status, _, _, match, pairs = archive._anchors(text, spans, index)
            if status == "reference_bound" and match == key and pairs == [(i, first+i) for i in range(len(spans))]:
                source = key
                aligned, changes = archive._changes(text, [(1, span) for span in spans], catalog[key]["paragraphs"],
                                                     [(i, j, "fingerprint_anchor") for i, j in pairs])
    return {"baseline": prior, "source": source,
            "aligned": [[r["archive_paragraph"], *r["submitted_range"]] for r in aligned],
            "submitted": [list(token) for change in changes for token in change["submitted"]["changed_tokens"]],
            "earlier": [list(token) for change in changes for token in change["archive"]["changed_tokens"]]}


def result(cases, catalog, index):
    edges = layout.index_edges(catalog)
    counts, rows = {}, []
    for case in cases:
        row = decision(case, catalog, index, edges)
        cell = counts.setdefault(case["role"], Counter())
        cell.update(total=1, baseline=bool(row["baseline"]), matched=bool(row["source"]),
                    incorrect=row["source"] is not None and row["source"] != case["expected"],
                    exact=case["truth"] is not None and row["submitted"] == [case["truth"]],
                    changed=bool(row["submitted"]))
        rows.append(row)
    passed = (counts["original"]["total"] >= 12 and counts["original"]["total"] == counts["original"]["matched"]
              and not counts["original"]["changed"] and not any(cell["incorrect"] for cell in counts.values())
              and all(counts[k]["exact"] * 10 >= counts[k]["total"] * 9 for k in TARGETS))
    return {"counts": counts, "passed": passed, "decisions": rows}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "measure", "replay"))
    for name in ("parent", "previous", "run"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    catalog, index = archive.library()
    works = source_panel(args.parent, args.previous, catalog)
    cases = list(fixtures(works))
    lock = {"schema": SCHEMA, "design": DESIGN, "archive": archive.ARCHIVE_SHA256,
            "earlier_layout_freeze": digest(args.previous / "freeze.json"),
            "implementation": {path.name: digest(path) for path in (Path(__file__), Path(layout.__file__))},
            "cases": [{**case, "text": text_hash(case["text"])} for case in cases]}
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        save(args.run / "freeze.json", lock)
        save(args.run / "private-sources.json", works)
        save(args.run / "private-cases.json", cases)
        print(f"Frozen {len(works)} source groups and {len(cases)} cases")
        return
    assert read(args.run / "freeze.json") == lock and read(args.run / "private-cases.json") == cases
    outcomes = result(cases, catalog, index)
    if args.phase == "measure":
        save(args.run / "result.json", outcomes)
    else:
        assert read(args.run / "result.json") == outcomes
    print(json.dumps({key: outcomes[key] for key in ("counts", "passed")}, indent=2))


if __name__ == "__main__":
    main()
