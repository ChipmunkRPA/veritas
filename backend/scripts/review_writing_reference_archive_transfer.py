"""Replay source attribution, archive glyph changes and exact model locations."""

from __future__ import annotations

import argparse
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import extract, from_string
from app.detectors.text.segment import segment
from app.evidence.document_revision import TOKENS
from scripts import writing_reference_archive_transfer as study
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import text_hash
from scripts.writing_focus_v3 import digest


def word_keys(value):
    words = [unicodedata.normalize("NFKC", match[0]).casefold().replace("’", "'") for match in study.WORD.finditer(value)]
    return [tuple(words[i:i + 8]) for i in range(len(words) - 7)], len(words)


def body_ranges(doc):
    return [[a, b] for (a, b), row in zip(doc.paragraphs, doc.source_map, strict=True)
            if row.get("quote_role", "body") == "body" and not doc.text[a:b].lstrip().startswith(">")]


def source_tables(catalog):
    indexed = {}
    documents = defaultdict(set)
    for source, row in catalog.items():
        by_paragraph = indexed[source] = []
        for paragraph in row["paragraphs"]:
            key_set = set(word_keys(paragraph["text"])[0])
            by_paragraph.append(key_set)
            for key in key_set:
                documents[key].add(source)
    permitted = {key for key, sources in documents.items() if len(sources) <= 4}
    return {source: [keys & permitted for keys in parts] for source, parts in indexed.items()}


def independent_reference(doc, index):
    body = body_ranges(doc)
    if not 3 <= len(body) <= 24:
        return "out_of_scope", None, []
    query = [word_keys(doc.text[a:b]) for a, b in body]
    total = sum(n for _, n in query)
    candidates = []
    for source, paragraphs in index.items():
        anchors = []
        for j, (keys, size) in enumerate(query):
            relevant = [(k, n, position) for k, position in enumerate(paragraphs)
                        if (n := len({i for start, key in enumerate(keys) if key in position
                                    for i in range(start, start + 8)})) >= 20 and n / size >= .55]
            relevant.sort(key=lambda item: -item[1])
            if relevant and (len(relevant) == 1 or relevant[0][1] != relevant[1][1]):
                k, covered, _ = relevant[0]
                anchors.append((j, k, covered))
        if len(anchors) >= 3 and all(a[1] < b[1] for a, b in zip(anchors, anchors[1:])):
            votes = sum(row[2] for row in anchors)
            if votes >= 60:
                candidates.append((votes, source, anchors))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] / total < .65:
        return "no_qualified_reference", None, []
    if len(candidates) > 1 and (candidates[0][0] - candidates[1][0]) / total < .10:
        return "ambiguous_reference", None, []
    return "reference_bound", candidates[0][1], [(a, b) for a, b, _ in candidates[0][2]]


def replay_hunks(old, new, pos, proof):
    old_cursor = new_cursor = 0
    for record in proof:
        a, b = record["earlier"]
        u, v = [mark - pos for mark in record["later"]]
        require(old_cursor <= a <= b <= len(old) and new_cursor <= u <= v <= len(new),
                "Edit coordinates outside the two paragraph ranges")
        require(TOKENS.findall(old[old_cursor:a]) == TOKENS.findall(new[new_cursor:u])
                and TOKENS.findall(old[a:b]) != TOKENS.findall(new[u:v]), "An unreported or artificial token change exists")
        old_tokens = {(m.start(), m.end()) for m in TOKENS.finditer(old[a:b])}
        new_tokens = {(m.start(), m.end()) for m in TOKENS.finditer(new[u:v])}
        require(all((left - a, right - a) in old_tokens for left, right in record["earlier_changed_tokens"])
                and all((left - pos - u, right - pos - u) in new_tokens for left, right in record["later_changed_tokens"]),
                "Changed-token coordinates do not select exact source words or punctuation")
        old_cursor, new_cursor = b, v
    require(TOKENS.findall(old[old_cursor:]) == TOKENS.findall(new[new_cursor:]), "Unreported trailing edit")
    return sum(len(item["earlier_changed_tokens"]) + len(item["later_changed_tokens"]) for item in proof)


def overlap(text, ranges, truth):
    first, last = truth
    target = {i for i in range(first, last) if not text[i].isspace()}
    claim = {i for start, stop in ranges for i in range(start, stop) if not text[i].isspace()}
    return bool(target) and len(target & claim) / len(target | claim) >= .30 and len(target & claim) / len(target) >= .50


def audit_fixture(run, case, output, catalog, full_index, measurements):
    path = run / "fixtures" / case["file"]
    doc = extract(path, path.suffix)
    require(digest(path) == case["file_sha256"] and text_hash(doc.text) == case["text_sha256"],
            "Frozen query fixture differs")
    archive = output["source_comparison"]
    local = catalog
    if case["kind"] == "duplicate_ambiguity":
        source = case["duplicate_source"]
        local = catalog | {source + "/duplicate": catalog[source]}
        index = source_tables(local)
    else:
        index = full_index
    status, source, strong = independent_reference(doc, index)
    require((status, source) == (archive["status"], archive["source"])
            and (source == case["source"]) is output["source_correct"], "Independent archive attribution differs")
    if source:
        body = body_ranges(doc)
        matched = [(i, j) for i, j, role in archive["matched"] if role == "fingerprint_anchor"]
        require(matched == strong, "Independent strong reference anchors differ")
        records = {(row["query_paragraph"], row["source_paragraph"]): row for row in archive["edits"]}
        token_checks = 0
        for i, j, role in archive["matched"]:
            require(role in {"fingerprint_anchor", "bounded_pair"} and (i, j) in strong or role == "bounded_pair",
                    "Unexpected main-study alignment")
            a, b = body[i]
            before = local[source]["paragraphs"][j]["text"]
            after = doc.text[a:b]
            record = records.get((i, j))
            if record:
                token_checks += replay_hunks(before, after, a, record["changes"])
                source_ordinal = local[source]["paragraphs"][j]["ordinal"]
                require(record["source_ordinal"] == source_ordinal, "Reference source paragraph number differs")
            else:
                require(TOKENS.findall(before) == TOKENS.findall(after), "Aligned paragraphs contain an undisplayed difference")
    else:
        token_checks = 0
        require(not archive["edits"] and not archive["matched"], "An abstention contains source differences")
    native = []
    if case["windows"]:
        sentences = segment(doc)
        require(len(sentences) == len(case["windows"]), "Incomplete native comparator windows")
        for i, evidence in enumerate(case["windows"]):
            sequence = [[unit.start, unit.end] for unit in sentences[max(0, i - 1):i + 2]]
            pin = text_hash(" ".join(doc.text[a:b] for a, b in sequence))
            require(evidence == {"spans": sequence, "sha256": pin}, "Comparator model input or span differs")
            value = measurements[pin]
            cov = value["coverage"]
            require(type(value["score"]) is float and math.isfinite(value["score"]) and 0 <= value["score"] <= 1
                    and cov["token_windows"] == 1 and 0 < cov["tokens"] <= 512 and not cov.get("omitted_tokens", 0)
                    and cov.get("covered_tokens", cov["tokens"]) == cov["tokens"], "Comparator input coverage is incomplete")
            native.append((sequence, value["score"]))
    current = [span for span, p in native if p >= .5][:6]
    target = case["truth"]
    left = bool(target) and any(overlap(doc.text, spans, target) for spans in current)
    right = bool(target) and source == case["source"] and any(
        overlap(doc.text, [row["query_envelope"]], target) for row in archive["edits"])
    require(left is output["current_useful"] and right is output["archive_useful"]
            and bool(current) is output["current_flag"], "Independent location outcome differs")
    return token_checks, len(native)


def boundary_probes(catalog, sources, unknown):
    index = study.build_index(catalog)
    counts = Counter()
    donor = unknown[0]["paragraphs"][1]["text"]
    for row in sources[:4]:
        group = row["group"]
        p = [part["text"] for part in row["paragraphs"]]
        probes = {
            "human_paragraph_insertion": (p[:2] + [donor] + p[2:], "bounded_insert"),
            "human_paragraph_deletion": (p[:2] + p[3:], "bounded_delete"),
            "human_paragraph_replacement": (p[:2] + [donor] + p[3:], "bounded_pair"),
            "first_three_paragraphs_only": (p[:3], None),
            "reversed_paragraph_order": (p[::-1], None),
        }
        for kind, (parts, relation) in probes.items():
            counts[f"{kind}/files"] += 1
            result = study.query_evidence(from_string(study.body_text(parts)), catalog, index)
            if kind == "reversed_paragraph_order":
                good = result["source"] is None and not result["edits"]
            elif kind == "first_three_paragraphs_only":
                good = result["source"] == group and not result["edits"]
            else:
                same = [record for record in result["edits"] if record["relation"] == relation]
                good = result["source"] == group and len(result["edits"]) == len(same) == 1
                if good and kind == "human_paragraph_deletion":
                    good = same[0]["query_envelope"] == [study.offset(parts, 2)] * 2
                elif good:
                    start = study.offset(parts, 2)
                    good = same[0]["query_envelope"][0] >= start and same[0]["query_envelope"][1] <= start + len(donor)
            counts[f"{kind}/expected"] += good
    return dict(counts)


def audit(run):
    expected = study.raw_contract(run)
    require(read(run / "source-freeze.json") == expected, "Frozen publisher rights or input identities differ")
    catalog = read(run / "index.json")
    raw = {volume: study.source_rows(run / "raw" / f"{volume}.xml", volume) for volume in study.VOLUMES}
    source_count = 0
    for row in catalog.values():
        genuine = raw[row["volume"]][row["doc_id"]]["paragraphs"]
        require(all(part == genuine[part["ordinal"]] for part in row["paragraphs"]), "Indexed body differs from pinned source")
        source_count += len(row["paragraphs"])
    probe = next(row for row in read(run / "earlier.json") if row.get("probe"))
    genuine_older = study.source_rows(run / "raw" / f"{study.TYPO_VOLUME}.typo-parent.xml", study.TYPO_VOLUME)["d585"]
    require(all(part == genuine_older["paragraphs"][part["ordinal"]] for part in probe["earlier"]), "Earlier publisher body differs")
    model = read(run / "primary.json")
    lock, outcomes = read(run / "input-lock.json"), read(run / "outcomes.json")
    require(model["binding"] == digest(run / "input-lock.json") and lock["freeze_sha256"] == digest(run / "freeze.json")
            and set(lock["inputs"]) == model["responses"].keys()
            and model["forwards"] == (len(lock["inputs"]) + 3) // 4, "Frozen model packet binding differs")
    index = source_tables(catalog)
    native_checks = token_checks = 0
    for case, item in zip(lock["cases"], outcomes, strict=True):
        require(all(item[key] == value for key, value in case.items()), "Outcome lost its case binding")
        tokens, native = audit_fixture(run, case, item, catalog, index, model["responses"])
        token_checks += tokens
        native_checks += native
    report = study.summary(outcomes, lock["admission"], catalog)
    require(report == read(run / "aggregate.json"), "Exact aggregate differs")
    names = ("source-freeze.json", "source-receipt.json", "freeze.json", "index.json", "input-lock.json", "primary.json", "outcomes.json", "aggregate.json")
    return {"schema": "chipmunk.writing-reference-archive-audit/1", "verified": True, "fixtures": len(outcomes),
            "verified_archive_paragraphs": source_count, "exact_edited_token_offsets": token_checks,
            "native_window_span_bindings": native_checks, "independent_source_attributions": len(outcomes),
            "boundary_probes": boundary_probes(catalog, read(run / "originals.json"), read(run / "unknown.json")),
            "commitments": {name: digest(run / name) for name in names}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    args = cli.parse_args()
    result = audit(args.run)
    write(args.run / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
