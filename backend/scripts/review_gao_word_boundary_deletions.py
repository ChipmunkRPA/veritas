"""Rebuild held GAO Word omissions and controls, then verify their native positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

SCHEMA = "chipmunk.gao-word-boundary-deletions/1"
PACK_SHA = "671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e"
SOURCE_COUNT = 24
POSITIVE = {"one_word", "three_words"}
CONTROLS = {"reflow", "empty_paragraph", "table", "quote", "hidden"}
STATUSES = {"reference_bound", "out_of_scope", "no_qualified_reference", "ambiguous_reference"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fixtures(transcript, stream):
    for index in range(len(stream) // 2, len(stream) - 20):
        parts = stream[index - 1:index + 4]
        if not all(word.isalpha() for word, _, _ in parts) or len({word for word, _, _ in parts}) != 5:
            continue
        if any(transcript[a[2]:b[1]] != " " for a, b in zip(parts, parts[1:])):
            continue
        _, start, one = parts[1]
        three = parts[3][2]
        left = "  " + transcript[:start].rstrip()
        first, rest = "\t " + transcript[one:].lstrip(), "\t " + transcript[three:].lstrip()
        yield "one_word", left, first, (start, one)
        yield "three_words", left, rest, (start, three)
        yield "reflow", left, "\t " + transcript[start:], None
        for role in ("empty_paragraph", "table", "quote", "hidden"):
            yield role, left, first, (start, one)
        return


def _word(role, left, right, omitted):
    from docx import Document

    word = Document()
    paragraph = word.add_paragraph(style="Quote" if role == "quote" else None)
    paragraph.add_run(left + "  ")
    if role == "hidden":
        paragraph.add_run(omitted).font.hidden = True
    if role == "empty_paragraph":
        word.add_paragraph()
    if role == "table":
        word.add_table(1, 1).cell(0, 0).text = omitted
    word.add_paragraph(right)
    source = BytesIO()
    word.save(source)
    canonical = BytesIO()
    with ZipFile(source) as packed, ZipFile(canonical, "w") as result:
        for name in sorted(packed.namelist()):
            result.writestr(ZipInfo(name), packed.read(name))
    return canonical.getvalue()


def replay(pack, excluded):
    from docx import Document

    from app.evidence import document_archive_word, document_gao_archive

    reports, index = document_gao_archive.library(pack, PACK_SHA)
    cases = []
    ordered = sorted(reports, key=lambda name: hashlib.sha256(f"{SCHEMA}/{name}".encode()).digest())
    for name in ordered:
        transcript = reports[name]["text"]
        if name in excluded or reports[name]["split"] != "test" or not transcript.isascii():
            continue
        stream = document_gao_archive.token_stream(transcript)
        if not 100 <= len(stream) <= 650:
            continue
        fixtures = list(_fixtures(transcript, stream))
        if not fixtures:
            continue
        for role, left, right, target in fixtures:
            data = _word(role, left, right, transcript[slice(*target)] if target else "")
            digest = hashlib.sha256(data).hexdigest()
            previous = document_archive_word.compare_gao(data, digest, PACK_SHA, reports, index, allow_boundary_deletion=False)
            current = document_archive_word.compare_gao(data, digest, PACK_SHA, reports, index)
            reference = (current["reference"] or {}).get("id")
            _require(reference in {name, None}, f"Wrong source attribution for {name} {role}")
            found, transcript_range = [], None
            if current["changes"]:
                _require(role in POSITIVE and len(current["changes"]) == 1, f"Unexpected change for {name} {role}")
                change, = current["changes"]
                submitted, archive = change["submitted"], change["archive"]
                transcript_range = [archive["start"], archive["end"]]
                _require(transcript_range == list(target) and transcript[slice(*target)] == archive["text"],
                         f"Wrong omitted archive span for {name} {role}")
                _require(submitted["start"] == submitted["end"] == len(left.strip()) + 2
                         and submitted["text"] == "" and not submitted["changed_tokens"], "Unexpected submitted text")
                original = Document(BytesIO(data))
                for location in submitted["docx_locations"]:
                    paragraph = original.paragraphs[location["paragraph"] - 1]
                    offset = (len(paragraph.text.rstrip()) if location["paragraph"] == 1
                              else len(paragraph.text) - len(paragraph.text.lstrip()))
                    expected = [offset, offset]
                    _require(location["paragraph_range"] == expected and location["runs_complete"]
                             and not location["changed_tokens"], f"Wrong native endpoint for {name} {role}")
                    found.append([location["paragraph"], location["body_index"], *expected])
            _require(current["authorship_confidence_contribution"] == 0, "Unexpected authorship weight")
            cases.append({"source": name, "role": role, "word_sha256": digest, "previous": previous["status"],
                          "status": current["status"], "reference": reference, "omitted_transcript_range": transcript_range,
                          "native_points": found})
        if len(cases) == SOURCE_COUNT * (len(POSITIVE) + len(CONTROLS)):
            break
    return {"schema": SCHEMA, "pack_sha256": PACK_SHA, "excluded_sources": sorted(excluded), "cases": cases}


def recount(receipt):
    _require(receipt["schema"] == SCHEMA and receipt["pack_sha256"] == PACK_SHA, "Unexpected panel")
    rows = receipt["cases"]
    sources = {row["source"] for row in rows}
    pairs = {(row["source"], row["role"]) for row in rows}
    _require(len(sources) == SOURCE_COUNT and len(rows) == len(pairs)
             and pairs == {(name, role) for name in sources for role in POSITIVE | CONTROLS}
             and not sources.intersection(receipt["excluded_sources"]), "Incomplete or overlapping panel")
    counts = {role: Counter() for role in sorted(POSITIVE | CONTROLS)}
    for row in rows:
        role = row["role"]
        found = row["native_points"]
        start, end = row["omitted_transcript_range"] or (0, 0)
        _require(row["previous"] in STATUSES and row["status"] in STATUSES and re.fullmatch(r"[a-f0-9]{64}", row["word_sha256"])
                 and bool(found) == (0 <= start < end) and bool(row["reference"]) == (row["status"] == "reference_bound")
                 and row["reference"] in {row["source"], None}, "Inconsistent source or omission")
        _require(not found or role in POSITIVE and row["status"] == "reference_bound"
                 and [(p, i) for p, i, _, _ in found] == [(1, 0), (2, 1)]
                 and all(0 <= lo == hi for _, _, lo, hi in found), "Invalid native endpoints")
        counts[role].update(deliveries=1, old_matches=row["previous"] == "reference_bound",
                            matches=row["status"] == "reference_bound", exact_omissions=bool(found))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--exclude", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    exclusions = None
    if args.exclude:
        exclusions = {name for path in args.exclude
                      for name in re.findall(r"GAO-\d+-\d+(?:-SP|-T|-R)?", path.read_text().upper())}
        _require(len(exclusions) >= 50, "Missing earlier source identities")
    if args.pack:
        _require(exclusions is not None, "Pass the preceding GAO ledgers to freeze source exclusions")
        result = replay(args.pack, exclusions)
        result["summary"] = recount(result)
        if args.write:
            with args.receipt.open("x") as stream:
                stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _require(result == json.loads(args.receipt.read_text()), "Native replay differs from the receipt")
    else:
        _require(not args.write, "--write requires the pinned pack and preceding ledgers")
        result = json.loads(args.receipt.read_text())
        if exclusions is not None:
            _require(result["excluded_sources"] == sorted(exclusions), "Recorded exclusions differ")
        _require(result["summary"] == recount(result), "Recorded counts differ")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
