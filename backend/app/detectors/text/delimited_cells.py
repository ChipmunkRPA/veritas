"""Read writing from bounded delimited-text records with native cell positions."""

from __future__ import annotations

import csv
import io
import re
from itertools import islice
from pathlib import Path

from app.evidence.document_native_locations import CSV_DECLARED_SEPARATORS

MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 1_024
MAX_FIELDS = 1_000_000
MAX_TEXT_CELLS = 8_000
SAMPLE_ROWS = 24
_SEPARATOR_DECLARATION = re.compile(r'\A(?:sep=([^\r\n"])|"sep=([^\r\n"])")(?:\r\n|\r|\n|\Z)', re.IGNORECASE)
_SPLIT = "The table exceeds the local reader limit. Split it into smaller files."
_QUOTING = "The table has invalid quoting. Export it again as CSV or TSV with quoted text fields."
_FORMULA = re.compile(
    r"(?:=|[+@-]\s*(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(|\$?[A-Za-z]{1,3}\$?\d+\b|\d[\d.,]*\s*[-+/*^&=]))"
)
_NUMERIC = re.compile(r"[+−-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:[eE][+-]?\d+)?\s*%?")
_LINK = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.-]*://\S+|www\.\S+|[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+)", re.I)


def _widths(text: str, delimiter: str):
    """Validate field boundaries before the CSV reader allocates a record."""
    punctuation = re.compile(r'\r\n|["\r\n' + re.escape(delimiter) + "]")
    state = "empty"
    previous = record_start = 0
    columns = 1
    for match in punctuation.finditer(text):
        if match.start() > previous:
            if state == "closed":
                raise ValueError(_QUOTING)
            if state == "empty":
                state = "unquoted"
        previous = match.end()
        token = match.group()
        if token == '"':
            if state == "unquoted":
                raise ValueError(_QUOTING)
            state = "closed" if state == "quoted" else "quoted"
        elif state != "quoted":
            state = "empty"
            if token == delimiter:
                columns += 1
                if columns > MAX_COLUMNS:
                    raise ValueError(_SPLIT)
            else:
                yield columns if match.start() > record_start else 0
                columns, record_start = 1, previous
    if state == "quoted" or (state == "closed" and previous < len(text)):
        raise ValueError(_QUOTING)
    if record_start < len(text):
        yield columns


def _sample_widths(text: str, delimiter: str) -> tuple[int, ...]:
    try:
        return tuple(width for width in islice(_widths(text, delimiter), SAMPLE_ROWS) if width)
    except ValueError:
        return ()


def _delimiter(text: str, extension: str) -> str:
    if extension == ".tsv":
        return "\t"
    if ";" not in text:
        return ","
    semicolon = _sample_widths(text, ";")
    if len(semicolon) < 2 or semicolon[0] < 2 or len(set(semicolon)) != 1:
        return ","
    comma = _sample_widths(text, ",")
    if not comma:
        return ";"
    if comma[0] > 1:
        return ","
    # An unquoted one-column CSV can contain semicolons as ordinary prose.
    # Without stronger quote evidence, require a short table heading.
    heading = re.match(r"[\r\n]*([^\r\n]*)", text)[1]
    try:
        header = next(csv.reader((heading,), delimiter=";", strict=True))
    except csv.Error:
        return ","
    return ";" if len(header) == semicolon[0] and all(
        0 < len(field.strip()) <= 64 and not field.strip().endswith((".", "?", "!")) for field in header
    ) else ","


def _table_text(path: Path) -> str:
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(_SPLIT)
    try:
        text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Export the table as UTF-8 or UTF-16 with a byte order mark.") from error
    if "\x00" in text:
        raise ValueError("The table contains binary data. Export it again as CSV or TSV.")
    return text


def _writing(field: str) -> bool:
    return bool(field and not _FORMULA.match(field) and not _NUMERIC.fullmatch(field)
                and not _LINK.fullmatch(field) and any(char.isalpha() for char in field))


def read(path: Path, extension: str, *, max_chars: int,
         honor_separator_declaration: bool = True) -> tuple[list[tuple[str, dict]], str | None]:
    """Return mapped writing fields and the separator declared in the file, if used."""
    if extension not in {".csv", ".tsv"}:
        raise ValueError("Choose a CSV or TSV file.")
    text = _table_text(path)
    declaration = _SEPARATOR_DECLARATION.match(text) if extension == ".csv" and honor_separator_declaration else None
    declared_separator = (declaration[1] or declaration[2]) if declaration else None
    first_record = 1
    if declaration and declared_separator in CSV_DECLARED_SEPARATORS:
        text = text[declaration.end():]
        # Keep source record numbers aligned with the file, including its opening line.
        first_record = 2
    else:
        declared_separator = None
    delimiter = declared_separator or _delimiter(text, extension)
    fields = 0
    for row, width in enumerate(_widths(text, delimiter), first_record):
        fields += width
        if row > MAX_ROWS or fields > MAX_FIELDS:
            raise ValueError(_SPLIT)
    parts = []
    characters = raw_characters = 0
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        for row, fields_here in enumerate(reader, first_record):
            for column, raw in enumerate(fields_here, 1):
                if not _writing(raw.strip()):
                    continue
                characters += len(" ".join(raw.split())) + (2 if parts else 0)
                raw_characters += len(raw)
                if characters > max_chars or raw_characters > 4 * max_chars or len(parts) >= MAX_TEXT_CELLS:
                    raise ValueError(_SPLIT)
                parts.append((raw, {"kind": "markup", "paragraph_index": len(parts), "part_kind": "delimited_cell",
                                    "row": row, "column": column}))
    except csv.Error as error:
        message = _SPLIT if str(error).startswith("field larger than field limit") else _QUOTING
        raise ValueError(message) from error
    return parts, declared_separator
