"""Freeze, measure and review source matching after paragraph breaks are lost."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.evidence import document_archive as archive
from app.evidence import document_archive_layout as layout
from scripts.writing_reference_archive_transfer import VOLUMES, source_rows

SCHEMA = "chipmunk.writing-archive-joined-layout-transfer/1"
METHOD = {
    "scope": "One extracted plain-text paragraph, exact first and last eight tokens for each consecutive indexed paragraph.",
    "attribution": "Unique complete sequence of three or more paragraph edges, followed by the unchanged archive fingerprint checks on every recovered span.",
    "known_edit": "Replace one interior alphabetic token in the third paragraph with a distinct word already present in the same document.",
    "selection": "Up to five work groups per pinned volume in deterministic hash order, excluding earlier queried works, exact 12-word overlaps and repeated sender keys within this panel. Prefer sender keys absent from prior queries. One extra archive-absent group per volume where eligible.",
    "controls": "Same-work reversed paragraphs, one paragraph from another source, added unrelated introduction, quote marker, out-of-index and duplicated library entries.",
    "advance": "At least 12 independent indexed work groups, at least 75% of all exact normalized originals and known edits, and zero incorrect attributions or differences on identical flattened copies.",
}
WORD = re.compile(r"[A-Za-z]{4,}\b")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def fingerprint(parts):
    output = set()
    for text in parts:
        words = [m[0].casefold() for m in archive.WORD.finditer(text)]
        output.update(tuple(words[i:i + 12]) for i in range(len(words) - 11))
    return output


def selected_edit(parts):
    center = parts[2]
    matches = [m for m in WORD.finditer(center) if len(center) * .4 < m.start() < len(center) * .6]
    for match in matches:
        alternatives = [word[0] for word in WORD.finditer(parts[0])
                        if len(word[0]) == len(match[0]) and word[0].casefold() != match[0].casefold()]
        if alternatives:
            return match.start(), match.end(), alternatives[0].lower()
    return None


def create_cases(parent, catalog):
    pins = {volume: digest(parent / "raw" / f"{volume}.xml") for volume in VOLUMES}
    assert all(pins[volume] == VOLUMES[volume][0] for volume in VOLUMES)
    raw = {v: source_rows(parent / "raw" / f"{v}.xml", v) for v in VOLUMES}
    previous = read(parent / "originals.json") + read(parent / "unknown.json") + read(parent / "earlier.json")
    used_groups = {row["group"] for row in previous}
    prior_senders = {row["sender_key"] for row in previous}
    used_senders = set()
    used_text = set().union(*(fingerprint([p["text"] for p in row["paragraphs"]]) for row in previous))
    indexed, absent = {}, {}
    for volume, rows in raw.items():
        counted_index = counted_absent = 0
        ordered = sorted(rows.values(), key=lambda item: (item["sender_key"] in prior_senders,
                                                         text_hash("joined-layout-1/" + item["group"])))
        for row in ordered:
            group, sender = row["group"], row["sender_key"]
            if group in used_groups or sender in used_senders:
                continue
            if group in catalog and counted_index < 5:
                parts = [p["text"] for p in catalog[group]["paragraphs"]]
                if not 4 <= len(parts) <= 6 or ">" in "".join(parts) or selected_edit(parts) is None:
                    continue
                signatures = fingerprint(parts)
                if signatures & used_text:
                    continue
                # Bind every selected text to the raw official volume by ordinal.
                assert all(p["text"] == row["paragraphs"][p["ordinal"]]["text"] for p in catalog[group]["paragraphs"])
                indexed[group] = parts
                counted_index += 1
            elif group not in catalog and counted_absent < 1:
                nonempty = next((parts for i in range(len(row["paragraphs"]) - 3)
                                 if all(parts := [p["text"] for p in row["paragraphs"][i:i + 4]])), None)
                if not nonempty or ">" in "".join(nonempty):
                    continue
                signatures = fingerprint(nonempty)
                if signatures & used_text:
                    continue
                absent[group] = nonempty
                counted_absent += 1
            else:
                continue
            used_senders.add(sender)
            used_text |= signatures
    assert len(indexed) >= 12 and len(absent) >= 3, (len(indexed), len(absent))

    cases = []
    indexed_rows = list(indexed.items())
    for i, (group, parts) in enumerate(indexed_rows):
        base = " ".join(parts)
        a, b, value = selected_edit(parts)
        target = sum(len(p) + 1 for p in parts[:2]) + a
        edited = base[:target] + value + base[target + b - a:]
        donor = indexed_rows[(i + 1) % len(indexed_rows)][1]
        cases.extend((
            (group, "native_original", "\n\n".join(parts), group, None),
            (group, "joined_original", base, group, None),
            (group, "joined_edit", edited, group, [target, target + len(value)]),
            (group, "reordered", " ".join(parts[::-1]), None, None),
            (group, "mixed_sources", " ".join([*parts[:2], donor[2], *parts[3:]]), None, None),
            (group, "intro_added", "A separate note accompanies this archival material. " + base, None, None),
            (group, "quoted", "> " + base, None, None),
        ))
        if i < 4:
            cases.append((group, "duplicate_index", base, None, None))
    cases.extend((group, "out_of_index", " ".join(parts), None, None) for group, parts in absent.items())

    for row in read(parent / "outcomes.json"):
        if row["kind"] not in {"qwen", "smol2"}:
            continue
        source = parent / "fixtures" / row["file"]
        assert digest(source) == row["file_sha256"]
        text = source.read_text()
        a, b = row["truth"]
        ranges = [len(text[:n].replace("\n\n", " ")) for n in (a, b)]
        cases.append((row["group"], "earlier_generated_reflow", text.replace("\n\n", " "), row["source"], ranges))
    return pins, cases


def freeze(parent, catalog):
    pins, cases = create_cases(parent, catalog)
    implementation = (Path(archive.__file__), Path(layout.__file__), Path(__file__))
    binding = {"schema": SCHEMA, "method": METHOD, "archive_sha256": archive.ARCHIVE_SHA256,
               "parent_outcomes_sha256": digest(parent / "outcomes.json"), "volumes": pins,
               "implementation_sha256": {p.name: digest(p) for p in implementation},
               "cases": [[g, role, text_hash(text), source, target] for g, role, text, source, target in cases]}
    return binding, cases


def run_joined(doc, catalog, index, edges):
    if len(doc.paragraphs) != 1 or ">" in doc.text:
        return None, [], []
    result = layout.joined_spans(doc.text, doc.paragraphs[0], edges)
    if result is None:
        return None, [], []
    source, first, spans = result
    status, _, _, found, pairs = archive._anchors(doc.text, spans, index)
    if status != "reference_bound" or found != source or pairs != [(i, first + i) for i in range(len(spans))]:
        return None, [], []
    return source, *archive._changes(doc.text, [(1, span) for span in spans], catalog[source]["paragraphs"],
                                     [(i, j, "fingerprint_anchor") for i, j in pairs])


def measure(cases, catalog, index):
    edges = layout.index_edges(catalog)
    counts = {}
    decisions = []
    for group, role, text, expected, truth in cases:
        doc = from_string(text)
        assert doc.text == text
        base = archive.compare(doc, text_hash(text), catalog, index)
        source = (base.get("reference") or {}).get("id")
        aligned, changes = base["aligned_paragraphs"], base["changes"]
        prior = source
        if source is None and role == "duplicate_index":
            doubled = catalog | {group + "/duplicate": catalog[group]}
            source, aligned, changes = run_joined(doc, doubled, archive.build_index(doubled), layout.index_edges(doubled))
        elif source is None:
            source, aligned, changes = run_joined(doc, catalog, index, edges)
        hits = [list(token) for change in changes for token in change["submitted"]["changed_tokens"]]
        useful = bool(truth and any(a < truth[1] and b > truth[0] for a, b in hits))
        exact = bool(truth and hits == [truth])
        counter = counts.setdefault(role, Counter())
        counter.update(total=1, baseline=int(prior is not None), matched=int(source is not None),
                       false_attribution=int(source is not None and source != expected),
                       useful=int(useful), exact=int(exact), has_changes=int(bool(changes)))
        decisions.append({"group": group, "role": role, "source": source, "baseline": prior, "expected": expected,
                          "truth": truth, "aligned": [[r["archive_paragraph"], *r["submitted_range"]] for r in aligned],
                          "changes": hits, "useful": useful, "exact": exact})
    original, edit = (counts[key] for key in ("joined_original", "joined_edit"))
    advance = (original["total"] >= 12 and original["matched"] * 4 >= original["total"] * 3
               and edit["exact"] * 4 >= edit["total"] * 3 and not original["has_changes"]
               and not sum(row["false_attribution"] for row in counts.values()))
    return {"counts": counts, "advance": advance, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "replay"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    catalog, index = archive.library()
    binding, cases = freeze(args.parent, catalog)
    lock = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir()
        save(lock, binding)
        print(f"Frozen {len(cases)} inputs")
        return
    assert read(lock) == binding
    result = measure(cases, catalog, index)
    if args.phase == "measure":
        save(args.run / "results.json", result)
    else:
        assert read(args.run / "results.json") == result
    print(json.dumps({key: result[key] for key in ("counts", "advance")}, indent=2))


if __name__ == "__main__":
    main()
