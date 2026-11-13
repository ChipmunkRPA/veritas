"""Rebuild public PLOS excerpts and verify mixed-format draft evidence."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import pymupdf

from app.evidence import document_revision
from scripts import rebuild_writing_word_boundary_sources as publisher
from scripts.review_document_revision_export import review

FORMATS = (".txt", ".docx", ".pdf")


def pdf(text):
    with pymupdf.open() as document:
        page = document.new_page()
        font = pymupdf.Font("helv")
        if any(not character.isspace() and not font.has_glyph(ord(character), fallback=False) for character in text):
            raise ValueError("The fixed PDF font lacks a character from the publisher excerpt")
        page.insert_font(fontname="ReplayText", fontbuffer=font.buffer)
        bottom = page.insert_textbox((54, 60, 558, 740), text, fontname="ReplayText", fontsize=10)
        if bottom < 0:
            raise ValueError("Publisher excerpt does not fit the fixed native PDF page")
        if document_revision.TOKENS.findall(text) != document_revision.TOKENS.findall(page.get_text()):
            raise ValueError("PDF rendering changed the publisher excerpt's words or punctuation")
        return document.tobytes(deflate=True, no_new_id=True)


def rendition(text, extension):
    if extension == ".txt":
        return text.encode()
    if extension == ".docx":
        return publisher.docx(text)
    return pdf(text)


def save(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Cached rendition changed: {path.name}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def roles(record, side):
    return [change[side] for change in record["changes"] if change[side] is not None]


def changed_tokens(record, side):
    return [[row["text"][start - row["start"]:end - row["start"]] for start, end in row["changed_tokens"]]
            for row in roles(record, side)]


def result(cache, work, *, download):
    word_cache = work / "word"
    qualified = publisher.build(cache, word_cache, download=download)
    rows = []
    for source in qualified["native_drafts"]:
        group = publisher.DOI.fullmatch(source["doi"])[1]
        original, ordinary = (document_revision._docx_source((word_cache / source["drafts"][role]["filename"]).read_bytes())[0]
                              for role in ("earlier", "ordinary"))
        baseline = document_revision.compare(original.encode(), ordinary.encode(), ".txt")
        if not baseline["changes"]:
            raise ValueError("Known source wording change disappeared")
        paths = {}
        for role, text in (("original", original), ("ordinary", ordinary)):
            for extension in FORMATS:
                path = work / f"{group}-{role}{extension}"
                save(path, rendition(text, extension))
                paths[role, extension] = path
        for old_format, new_format in permutations(FORMATS, 2):
            earlier = paths["original", old_format]
            for role in ("original", "ordinary"):
                later = paths[role, new_format]
                comparison, *_ = document_revision.compare_with_sources(
                    earlier.read_bytes(), later.read_bytes(), new_format, earlier_extension=old_format,
                )
                if role == "original":
                    if comparison["changes"]:
                        raise ValueError(f"Ordinary formatting produced a wording difference: {group}")
                elif any(changed_tokens(comparison, side) != changed_tokens(baseline, side) for side in ("earlier", "later")):
                    raise ValueError(f"Cross-format comparison missed or changed the wording ranges: {group}")
                export = work / f"{group}-{role}-{old_format[1:]}-{new_format[1:]}.json"
                serialized = json.dumps(comparison, ensure_ascii=False, sort_keys=True).encode()
                save(export, serialized)
                verified = review(export, earlier, later)
                rows.append({"doi": source["doi"], "authors": source["authors"], "license": source["license"],
                             "source_group": group, "role": role, "earlier_format": old_format, "later_format": new_format,
                             "changes": len(comparison["changes"]), "earlier_sha256": comparison["earlier"]["sha256"],
                             "later_sha256": comparison["later"]["sha256"], "comparison_sha256": publisher.digest(serialized),
                             "verification": verified})
    return {"schema": "chipmunk.document-mixed-formats-replay/1", "source_work_count": len(qualified["native_drafts"]),
            "pymupdf": pymupdf.VersionBind, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(mode=0o700, parents=True, exist_ok=True)
    print(json.dumps(result(args.cache, args.work, download=args.download), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
