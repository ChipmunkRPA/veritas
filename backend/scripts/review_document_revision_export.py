"""Replay a draft-comparison JSON using the two original source files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app.evidence.document_revision import (
    EDITED_RELOCATION_SCHEMA,
    FORMATS,
    HTML_COORDINATES,
    MAX_BYTES,
    RELOCATION_SCHEMA,
    compare_with_sources,
)
from app.evidence.document_revision_word_alignment import SCHEMA as ALIGNED_WORD_TABLES_SCHEMA
from app.evidence.document_revision_word_tables import SCHEMA as WORD_TABLES_SCHEMA
from app.evidence.document_revision_writing import (
    ANCHORED_DRAFT_SCHEMA,
    BLOCK_DRAFT_SCHEMA,
    INDEPENDENT_SCHEMA,
    MIXED_DRAFT_SCHEMA,
    PDF_CONTINUATION_SCHEMA,
    PDF_INDEPENDENT_SCHEMA,
    refine_wording,
)

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
        raise ValueError("Use the original files in TXT, DOCX, ODT, PDF or EML format")
    received = json.loads(_read(export, MAX_PDF_EXPORT if ".pdf" in (extension, old_extension) else MAX_EXPORT), object_pairs_hook=_unique)
    if "paired_check" in received and not isinstance(received["paired_check"], dict):
        raise ValueError("The optional paired writing check has invalid fields")
    paired = received.pop("paired_check", None)
    schema = received.get("schema")
    anchored = schema == ANCHORED_DRAFT_SCHEMA
    relocated = schema in (RELOCATION_SCHEMA, EDITED_RELOCATION_SCHEMA)
    if (anchored and paired is not None and paired.get("status") != "outside_scope"
            and paired.get("schema") != INDEPENDENT_SCHEMA):
        raise ValueError("Anchored wording changes cannot carry statistical sentence responses")
    blocks = paired is not None and paired.get("schema") == BLOCK_DRAFT_SCHEMA
    mixed_paired = paired is not None and (paired.get("schema") == MIXED_DRAFT_SCHEMA or (blocks and old_extension != extension))
    if paired is not None and mixed_paired != (old_extension != extension):
        raise ValueError("Paired writing responses use different source formats from the supplied drafts")
    pdf_independent = (paired is not None and (paired.get("schema") in (PDF_INDEPENDENT_SCHEMA, PDF_CONTINUATION_SCHEMA)
                       or (blocks and old_extension == extension == ".pdf")))
    expected, old_text, new_text, native_layouts = compare_with_sources(
        _read(earlier, MAX_BYTES), _read(later, MAX_BYTES), extension, earlier_extension=old_extension,
        email_body_only=paired is not None, pdf_paired_text_only=paired is not None,
        pdf_independent=pdf_independent, mixed_paired=mixed_paired,
        include_word_tables=(schema in {WORD_TABLES_SCHEMA, ALIGNED_WORD_TABLES_SCHEMA}
                             or relocated and "word_tables" in received),
        strict_word_tables=schema == WORD_TABLES_SCHEMA,
        include_relocations=anchored or relocated,
        include_edited_relocations=schema == EDITED_RELOCATION_SCHEMA,
        include_values="numeric_comparison" in received,
    )
    rebuilt = refine_wording(expected, old_text, new_text) if anchored else expected
    if json.dumps(received, sort_keys=True, allow_nan=False) != json.dumps(rebuilt, sort_keys=True, allow_nan=False):
        raise ValueError("The comparison does not replay from these source bytes and decoded positions")
    roles = Counter(row["role"] for row in rebuilt["changes"])
    excerpts = [row[side] for row in rebuilt["changes"] for side in ("earlier", "later") if row[side] is not None]
    result = {"source_hashes_verified": 2, "coordinate_space": expected["coordinate_space"], "changes_by_role": dict(roles)}
    if passages := expected.get("relocated_passages"):
        result["relocated_passages_verified"] = len(passages)
        excerpts.extend(passage[side] for passage in passages for side in ("earlier", "later"))
    if values := expected.get("numeric_comparison"):
        result["figures_compared"] = values["compared"]
        result["changed_figures_located"] = len(values["changes"])
        excerpts.extend(row[side] for row in values["changes"] for side in ("earlier", "later"))
    result["changed_token_ranges_verified"] = sum(len(excerpt["changed_tokens"]) for excerpt in excerpts)
    if {".docx", ".odt"} & {extension, old_extension}:
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
            expected, old_text, new_text, paired,
            word_layouts=native_layouts if extension == ".docx" else None,
            pdf_layouts=native_layouts if pdf_independent else None,
            mixed_layouts=native_layouts if mixed_paired else None,
        )
        word_ranges = sum(len(revision.get(side, {}).get("context_paragraphs", ()))
                          for revision in (paired, *paired.get("revisions", ())) for side in ("earlier", "later"))
        if word_ranges:
            result["paired_word_sentence_ranges_verified"] = word_ranges
        if pdf_independent or (mixed_paired and ".pdf" in (extension, old_extension)):
            result["paired_pdf_word_regions_verified"] = sum(
                len(page["word_regions"]) for revision in paired.get("revisions", ()) for side in ("earlier", "later")
                for page in revision[side].get("pdf_pages", ())
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
