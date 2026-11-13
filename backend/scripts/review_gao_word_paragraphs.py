"""Rebuild a source-held Word paragraph comparison panel from the pinned public GAO pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from docx import Document

from app.evidence import document_archive_word, document_gao_archive

PACK_SHA = "671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e"
SELECTION = "gao-word-paragraph-replay/1"
SOURCE_COUNT = 24


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _word(parts):
    document = Document()
    for part in parts:
        middle = len(part) // 2
        paragraph = document.add_paragraph()
        paragraph.add_run(part[:middle])
        paragraph.add_run(part[middle:])
    source = BytesIO()
    document.save(source)
    # Fix ZIP metadata so the replay also checks the exact submitted bytes.
    canonical = BytesIO()
    with ZipFile(source) as packed, ZipFile(canonical, "w") as result:
        for name in sorted(packed.namelist()):
            result.writestr(ZipInfo(name), packed.read(name))
    return canonical.getvalue()


def _fixtures(text):
    words = list(re.finditer(r"[^\W\d_]+", text))
    pair = next(((a, b) for a, b in zip(words[len(words) // 2:], words[len(words) // 2 + 1:])
                 if text[a.end():b.start()] == " " and b.end() - a.start() < 48), None)
    if pair is None:
        return
    left, right = pair
    start, split, end = left.start(), right.start(), right.end()
    before, after = text[:start], text[end:]
    before_right = text[:split].rstrip()
    from_right = text[split:]
    yield "reflow", (before_right, from_right), ()
    yield "replacement", (before + "amended", "schedule" + after), ((len(before), len(before) + 7), (0, 8))
    yield "split_word", (before + "amend", "ment" + after), ((len(before), len(before) + 5), (0, 4))
    yield "boundary_deletion", (before.rstrip(), text[split:]), ()
    yield "quote", ("> " + before_right, from_right), ()


def _locations(original, change):
    word = Document(BytesIO(original))
    found = []
    excerpt = change["submitted"]
    for location in excerpt["docx_locations"]:
        paragraph = word.paragraphs[location["paragraph"] - 1]
        start, end = location["paragraph_range"]
        native = paragraph.text[start:end]
        tokens = location["changed_tokens"]
        _require(len(tokens) == 1 and tokens[0]["paragraph_range"] == [start, end], "Native marked span differs")
        token = tokens[0]
        a, b = token["extracted_range"]
        _require(excerpt["text"][a - excerpt["start"]:b - excerpt["start"]] == native, "Word text differs at the marked range")
        restored = "".join(paragraph.runs[item["run"] - 1].text[slice(*item["range"])] for item in token["runs"])
        _require(restored == native, "Word run text differs at the marked range")
        found.append([start, end])
    return found


def replay(pack, excluded):
    reports, index = document_gao_archive.library(pack, PACK_SHA)
    cases = []
    selected = sorted(reports, key=lambda name: hashlib.sha256(f"{SELECTION}/{name}".encode()).digest())
    for name in selected:
        if name in excluded or reports[name]["split"] != "test":
            continue
        transcript = reports[name]["text"]
        if not 100 <= len(document_gao_archive.token_stream(transcript)) <= 650 or not transcript.isascii():
            continue
        fixtures = list(_fixtures(transcript))
        if not fixtures:
            continue
        for role, parts, target in fixtures:
            data = _word(parts)
            digest = hashlib.sha256(data).hexdigest()
            current = document_archive_word.compare_gao(data, digest, PACK_SHA, reports, index)
            previous = document_archive_word.compare_gao(data, digest, PACK_SHA, reports, index, allow_multi_paragraph=False)
            source = (current["reference"] or {}).get("id")
            _require(source in {None, name}, "A comparison matched a different source")
            found = []
            for change in current["changes"]:
                if change["submitted"].get("docx_locations"):
                    found.extend(_locations(data, change))
            if found:
                _require(found == [list(span) for span in target], "Comparison marked the wrong native positions")
            _require(current["authorship_confidence_contribution"] == 0, "Unexpected authorship weight")
            cases.append({"source": name, "role": role, "word_sha256": digest, "previous": previous["status"],
                          "status": current["status"], "reference": source, "changes": len(current["changes"]), "locations": found})
        if len(cases) == 5 * SOURCE_COUNT:
            break
    _require(len(cases) == 5 * SOURCE_COUNT, "Incomplete public panel")
    return {"schema": SELECTION, "pack_sha256": PACK_SHA, "excluded_sources": sorted(excluded), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("previous_ledger", nargs="+", type=Path)
    args = parser.parse_args()
    excluded = {name for path in args.previous_ledger
                for name in re.findall(r'GAO-\d+-\d+(?:-SP|-T|-R)?', path.read_text().upper())}
    _require(len(excluded) >= 20, "Missing earlier source identities")
    print(json.dumps(replay(args.pack, excluded), indent=2))


if __name__ == "__main__":
    main()
