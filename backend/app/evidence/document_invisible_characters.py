"""Locate selected invisible and look-alike Unicode characters in the source."""
from __future__ import annotations

from heapq import merge
from itertools import chain
from unicodedata import category, name

from app.core.canonical import canonical_hash
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import EMAIL_HTML_COORDINATES

KEY = "invisible_characters"
COMMITMENT = "invisible_characters_sha256"
SCHEMA = "document-invisible-characters/1"
GLYPH_SCHEMA = "document-invisible-characters/2"
TITLE = "Invisible characters in the source text"
SCOPE = ("Checks decoded plain text, mapped Word body and table paragraphs, and the selected plain or source-mapped HTML "
         "email body. Reports direction overrides, embedded byte order marks, and selected invisible breaks or joiners "
         "inside ASCII letters or digits.")
LIMITATION = ("These characters can come from ordinary formatting, accessibility tools or copy and paste. "
              "This observation carries zero AI-authorship weight.")
MAX_LOCATIONS = 24
OVERRIDES = frozenset({"\u202d", "\u202e"})
IN_WORD = frozenset({"\u00ad", "\u200b", "\u2060"})
ENCODED_FORMATTING = (OVERRIDES | IN_WORD | frozenset("\ufeff\u061c\u200c\u200d\u200e\u200f"
                                                   "\u202a\u202b\u202c\u2066\u2067\u2068\u2069"))
NAMES = {"U+202D": "Left-to-right override", "U+202E": "Right-to-left override", "U+FEFF": "Embedded byte order mark",
         "U+00AD": "Soft hyphen", "U+200B": "Zero-width space", "U+2060": "Word joiner"}
_RANGES = {"plain": "decoded_text", "python-docx": "word_paragraph"}

# Unicode 17.0 UTS #39 confusables.txt subset, Cyrillic and Greek letters
# with a single ASCII letter as their listed confusable skeleton.
# Copyright © 1991–2026 Unicode, Inc. See data/unicode-license-v3.txt.
LOOKALIKES = {
    0x0405: "S", 0x0406: "l", 0x0408: "J", 0x0410: "A", 0x0412: "B", 0x0415: "E",
    0x041A: "K", 0x041C: "M", 0x041D: "H", 0x041E: "O", 0x0420: "P", 0x0421: "C",
    0x0422: "T", 0x0423: "Y", 0x0425: "X", 0x042C: "b", 0x0430: "a", 0x0433: "r",
    0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0443: "y", 0x0445: "x",
    0x0448: "w", 0x0455: "s", 0x0456: "i", 0x0458: "j", 0x0461: "w", 0x0474: "V",
    0x0475: "v", 0x04AE: "Y", 0x04AF: "y", 0x04BB: "h", 0x04BD: "e", 0x04C0: "l",
    0x04CF: "l", 0x0501: "d", 0x050C: "G", 0x051B: "q", 0x051C: "W", 0x051D: "w",
    0xA647: "i",
    0x037A: "i", 0x037F: "J", 0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0396: "Z",
    0x0397: "H", 0x0399: "l", 0x039A: "K", 0x039C: "M", 0x039D: "N", 0x039F: "O",
    0x03A1: "P", 0x03A4: "T", 0x03A5: "Y", 0x03A7: "X", 0x03B1: "a", 0x03B3: "y",
    0x03B9: "i", 0x03BD: "v", 0x03BF: "o", 0x03C1: "p", 0x03C3: "o", 0x03C5: "u",
    0x03D2: "Y", 0x03DC: "F", 0x03F1: "p", 0x03F2: "c", 0x03F3: "j", 0x03F8: "p",
    0x03F9: "C", 0x03FA: "M", 0x1D26: "r", 0x1FBE: "i",
}
GLYPHS = {f"U+{code:04X}": equivalent for code, equivalent in LOOKALIKES.items()}
GLYPH_TITLE = "Invisible and look-alike characters in the source text"
GLYPH_SCOPE = (SCOPE + " Also checks listed Greek and Cyrillic look-alikes in words of basic Latin letters and "
               "look-alike letters, with at least three basic Latin letters and more Latin than look-alike letters.")
GLYPH_LIMITATION = ("Ordinary multilingual text, technical notation and copy and paste can introduce these characters. "
                    "This observation carries zero AI-authorship weight.")


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def _ascii_word(character):
    return character.isascii() and character.isalnum()


def _coordinate(method, email):
    if method != "email":
        return _RANGES.get(method)
    if email.get("body_content_type") == "text/plain":
        return "decoded_email"
    if email.get("body_content_type") == "text/html" and email.get("body_coordinates") == EMAIL_HTML_COORDINATES:
        return "decoded_email_html"
    return None


def _candidates(text):
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in OVERRIDES or ch == "\ufeff":
            yield i, ch
        elif ch in IN_WORD:
            start = i
            while i < len(text) and text[i] in IN_WORD:
                i += 1
            if start and i < len(text) and _ascii_word(text[start - 1]) and _ascii_word(text[i]):
                for position in range(start, i):
                    yield position, text[position]
            continue
        i += 1


def _lookalikes(text):
    latin = 0
    foreign = []
    start = 0
    for i, ch in enumerate(chain(text, " ")):
        if ch.isascii() and ch.isalpha():
            latin += 1
        elif ord(ch) in LOOKALIKES:
            foreign.append((i, ch))
        elif ch.isalpha() or (not ch.isascii() and category(ch).startswith("M")):
            # Leave a mixed run unassessed if it contains other letters or combining marks.
            continue
        else:
            if 0 < len(foreign) < latin and latin >= 3 and i - start == latin + len(foreign):
                yield from foreign
            latin, start = 0, i + 1
            foreign.clear()


def measure(text, mapping, meta, *, schema=SCHEMA):
    method = meta.get("method")
    email = meta.get("email") or {}
    coordinate = _coordinate(method, email)
    if coordinate is None:
        return None
    kind = {"plain": "text", "python-docx": "docx", "email": "email"}[method]
    blocks = chars = count = 0
    locations = []
    for block in mapping:
        ranges = block.get("source_ranges")
        if block.get("kind") != kind or not ranges or (kind == "email" and block.get("mime_part") != email.get("body_part")):
            continue
        start, end = block["start"], block["end"]
        blocks += 1
        chars += end - start
        range_index = 0
        contents = text[start:end]
        candidates = _candidates(contents)
        if schema == GLYPH_SCHEMA:
            candidates = merge(candidates, _lookalikes(contents))
        for index, character in candidates:
            offset = start + index
            while range_index < len(ranges) and ranges[range_index][1] <= offset:
                range_index += 1
            native = clip_source_ranges(ranges[range_index:range_index + 1], offset, offset + 1)
            if not native:
                continue
            count += 1
            if len(locations) == MAX_LOCATIONS:
                continue
            [(lo, hi)] = native
            place = {"text_offset": offset, "source_start": lo, "source_end": hi, "codepoint": f"U+{ord(character):04X}"}
            if schema == GLYPH_SCHEMA and ord(character) in LOOKALIKES:
                place["looks_like"] = LOOKALIKES[ord(character)]
            if kind == "docx":
                place["body_path"] = block["body_path"]
            elif kind == "email":
                place["mime_part"] = block["mime_part"]
            locations.append(place)
    return {"schema": schema, "coordinate_space": coordinate, "checked_blocks": blocks, "checked_chars": chars,
            "matches": count, "locations": locations, "authorship_confidence_contribution": 0}


def validate(document, expected):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None and expected is None:
        return
    source = document["source"]
    method = source.get("method")
    email = document["coverage"].get("email") or {}
    coordinate = _coordinate(method, email)
    if (not isinstance(record, dict) or coordinate is None or record.get("schema") not in {SCHEMA, GLYPH_SCHEMA}
            or record.get("coordinate_space") != coordinate or type(record.get("authorship_confidence_contribution")) is not int
            or record["authorship_confidence_contribution"] != 0 or canonical_hash(record) != expected
            or set(record) != {"schema", "coordinate_space", "checked_blocks", "checked_chars", "matches", "locations",
                               "authorship_confidence_contribution"}):
        raise ValueError("Invisible-character evidence differs from its committed scope")
    blocks, chars, count = (record[k] for k in ("checked_blocks", "checked_chars", "matches"))
    locations = record["locations"]
    if (any(type(value) is not int for value in (blocks, chars, count))
            or not 0 <= blocks <= chars <= source["chars"] or not 0 <= count <= chars
            or not isinstance(locations, list) or len(locations) != min(count, MAX_LOCATIONS)):
        raise ValueError("Invalid invisible-character coverage")
    previous = -1
    for row in locations:
        keys = {"text_offset", "codepoint", "source_start", "source_end"}
        equivalent = GLYPHS.get(row.get("codepoint")) if record["schema"] == GLYPH_SCHEMA else None
        if equivalent is not None:
            keys.add("looks_like")
        if method == "email":
            keys.add("mime_part")
        elif method == "python-docx":
            keys.add("body_path")
        if (set(row) != keys or (row["codepoint"] not in NAMES and equivalent is None)
                or row.get("looks_like") != equivalent
                or any(type(row[k]) is not int for k in ("text_offset", "source_start", "source_end"))
                or not previous < row["text_offset"] < source["chars"] or row["text_offset"] < 0
                or not 0 <= row["source_start"] < row["source_end"]
                or (coordinate != "decoded_email_html" and row["source_end"] - row["source_start"] != 1)):
            raise ValueError("Invalid invisible-character location")
        if method == "email" and row["mime_part"] != email.get("body_part"):
            raise ValueError("Invisible character has a different email source part")
        if method == "python-docx":
            from app.evidence.document_word_visibility import _native_path

            _native_path(row["body_path"])
        previous = row["text_offset"]


def verify_source(document, text, mapping, meta):
    record = (document.get("coverage") or {}).get(KEY)
    validate(document, document["source"].get(COMMITMENT))
    if record is not None and measure(text, mapping, meta, schema=record["schema"]) != record:
        raise ValueError("Character evidence differs from the source text")


def rebuild(text, mapping, meta, commitment):
    current = measure(text, mapping, meta, schema=GLYPH_SCHEMA)
    if current is not None and canonical_hash(current) == commitment:
        return current
    return measure(text, mapping, meta)


def character_name(row):
    if "looks_like" in row:
        return name(chr(int(row["codepoint"][2:], 16))).capitalize()
    return NAMES[row["codepoint"]]


def location_label(row, coordinate):
    if "body_path" in row:
        path = " / ".join(map(str, row["body_path"]))
        return f"Word body path {path}, paragraph characters"
    if "mime_part" in row:
        label = "decoded HTML source characters" if coordinate == "decoded_email_html" else "decoded source characters"
        return f"Email part {row['mime_part']}, {label}"
    return "Decoded source characters"
