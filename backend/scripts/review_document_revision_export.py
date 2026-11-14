"""Replay a draft-comparison JSON using the two original source files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app.evidence.document_revision import (
    FORMATS,
    HTML_COORDINATES,
    MAX_BYTES,
    RELOCATION_SCHEMA,
    compare_with_sources,
)
from app.evidence.document_revision_word_alignment import SCHEMA as ALIGNED_WORD_TABLES_SCHEMA
from app.evidence.document_revision_word_tables import SCHEMA as WORD_TABLES_SCHEMA

MAX_EXPORT = 2 * 1024 * 1024
MAX_PDF_EXPORT = 4 * 1024 * 1024


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError("An input is empty or exceeds the verification limit")
    return data


def _unique(fields):
    result = {}
    for name, value in fields:
        if name in result:
            raise ValueError("The exported JSON contains a duplicate field")
        result[name] = value
    return result


def review(export: Path, earlier: Path, later: Path) -> dict:
    extension = later.suffix.lower()
    old_extension = earlier.suffix.lower()
    if extension not in FORMATS or old_extension not in FORMATS:
        raise ValueError("Use the original files in TXT, DOCX, PDF or EML format")
    received = json.loads(_read(export, MAX_PDF_EXPORT if ".pdf" in (extension, old_extension) else MAX_EXPORT), object_pairs_hook=_unique)
    if "paired_check" in received and not isinstance(received["paired_check"], dict):
        raise ValueError("The optional paired writing check has invalid fields")
    paired = received.pop("paired_check", None)
    if paired is not None and old_extension != extension:
        raise ValueError("Paired writing responses require both drafts to have the same format")
    expected, old_text, new_text, word_layouts = compare_with_sources(
        _read(earlier, MAX_BYTES), _read(later, MAX_BYTES), extension, earlier_extension=old_extension,
        email_body_only=paired is not None, pdf_paired_text_only=paired is not None,
        include_word_tables=(received.get("schema") in {WORD_TABLES_SCHEMA, ALIGNED_WORD_TABLES_SCHEMA}
                             or received.get("schema") == RELOCATION_SCHEMA and "word_tables" in received),
        strict_word_tables=received.get("schema") == WORD_TABLES_SCHEMA,
        include_relocations=received.get("schema") == RELOCATION_SCHEMA,
    )
    if json.dumps(received, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("The comparison does not replay from these source bytes and decoded positions")
    roles = Counter(row["role"] for row in expected["changes"])
    excerpts = [row[side] for row in expected["changes"] for side in ("earlier", "later") if row[side] is not None]
    result = {"source_hashes_verified": 2, "coordinate_space": expected["coordinate_space"], "changes_by_role": dict(roles),
              "changed_token_ranges_verified": sum(len(excerpt["changed_tokens"]) for excerpt in excerpts)}
    if passages := expected.get("relocated_passages"):
        result["relocated_passages_verified"] = len(passages)
        excerpts.extend(passage[side] for passage in passages for side in ("earlier", "later"))
    if ".docx" in (extension, old_extension):
        result["body_paragraph_ranges_verified"] = sum(len(excerpt.get("paragraphs", ())) for excerpt in excerpts)
    if ".pdf" in (extension, old_extension):
        pages = [page for excerpt in excerpts for page in excerpt.get("pdf_pages", ())]
        result["pdf_word_regions_verified"] = sum(len(page["word_regions"]) for page in pages)
        result["pdf_previews_verified"] = sum("preview" in page for page in pages)
    if expected["coordinate_space"] == HTML_COORDINATES:
        result["decoded_html_ranges_verified"] = sum(
            len(token["source_ranges"]) for excerpt in excerpts for token in excerpt["html_tokens"]
        )
    if tables := expected.get("word_tables"):
        result["word_tables_compared"] = len(tables["compared"])
        result["word_tables_skipped"] = len(tables["skipped"])
        result["word_cell_token_ranges_verified"] = sum(len(change[side]["changed_tokens"])
            for table in tables["compared"] for change in table["changes"] for side in ("earlier", "later"))
    if paired is not None:
        from app.evidence.document_revision_writing import verify

        result["paired_scope_and_saved_response_arithmetic"] = verify(
            expected, old_text, new_text, paired, word_layouts=word_layouts,
        )
        if "context_paragraphs" in paired.get("earlier", {}):
            result["paired_word_sentence_ranges_verified"] = sum(
                len(paired[side]["context_paragraphs"]) for side in ("earlier", "later")
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("export", "earlier", "later"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    try:
        result = review(args.export, args.earlier, args.later)
    except (OSError, ValueError, TypeError, KeyError, LookupError) as error:
        print(f"Draft comparison did not verify. {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
