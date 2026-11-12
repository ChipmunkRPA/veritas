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
    SUPPORTED_DRAFTS,
    compare_with_text,
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
    if extension not in FORMATS or earlier.suffix.lower() != extension:
        raise ValueError(f"Use the two original versions as {SUPPORTED_DRAFTS}")
    received = json.loads(_read(export, MAX_PDF_EXPORT if extension == ".pdf" else MAX_EXPORT), object_pairs_hook=_unique)
    if "paired_check" in received and not isinstance(received["paired_check"], dict):
        raise ValueError("The optional paired writing check has invalid fields")
    paired = received.pop("paired_check", None)
    expected, old_text, new_text = compare_with_text(_read(earlier, MAX_BYTES), _read(later, MAX_BYTES), extension,
                                                    email_body_only=paired is not None)
    if json.dumps(received, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("The comparison does not replay from these source bytes and decoded positions")
    roles = Counter(row["role"] for row in expected["changes"])
    excerpts = [row[side] for row in expected["changes"] for side in ("earlier", "later") if row[side] is not None]
    result = {"source_hashes_verified": 2, "coordinate_space": expected["coordinate_space"], "changes_by_role": dict(roles),
              "changed_token_ranges_verified": sum(len(excerpt["changed_tokens"]) for excerpt in excerpts)}
    if extension == ".docx":
        result["body_paragraph_ranges_verified"] = sum(len(excerpt["paragraphs"]) for excerpt in excerpts)
    if extension == ".pdf":
        pages = [page for excerpt in excerpts for page in excerpt["pdf_pages"]]
        result["pdf_word_regions_verified"] = sum(len(page["word_regions"]) for page in pages)
        result["pdf_previews_verified"] = sum("preview" in page for page in pages)
    if expected["coordinate_space"] == HTML_COORDINATES:
        result["decoded_html_ranges_verified"] = sum(
            len(token["source_ranges"]) for excerpt in excerpts for token in excerpt["html_tokens"]
        )
    if paired is not None:
        from app.evidence.document_revision_writing import verify

        result["paired_scope_and_saved_response_arithmetic"] = verify(expected, old_text, new_text, paired)
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
