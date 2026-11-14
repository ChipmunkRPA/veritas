"""Page locations for invisible PDF text and discrepancies in copyable text."""

from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher

KEY = "pdf_text_visibility"
LEGACY_MAX_PAGES = 12
MAX_BOXES = 4
LEGACY_SCHEMA = "document-pdf-text-visibility/1"
FULL_PAGE_SCHEMA = "document-pdf-text-visibility/2"
SCHEMA = "document-pdf-text-visibility/3"
ASSESSMENT_HEADLINE = "PDF writing assessment limited by invisible text"
INCOMPLETE_HEADLINE = "PDF writing assessment limited by incomplete text visibility checks"
REPLACEMENT_HEADLINE = "PDF writing assessment limited by differing text layers"
LIMITATION = ("Searchable scans and accessibility overlays can contain invisible text. "
              "The recorded locations describe the PDF text layer and carry zero AI-authorship weight.")
REPLACEMENT_LIMITATION = ("PDF accessibility and OCR corrections can legitimately change copyable text. "
                          "This comparison uses the PDF's glyph text and copyable text. It carries zero AI-authorship weight.")
MAX_DIFF_CHARS = 768
LIGATURES = str.maketrans({char: unicodedata.normalize("NFKD", char) for char in "ﬀﬁﬂﬃﬄﬅﬆ"})


def inspect_page(page, spans):
    """Record the geometry of invisible words without copying their contents."""
    visible = {span["chars"] for span in spans if span["type"] in (0, 1) and span["opacity"] > 0}
    locations, count = [], 0
    for span in spans:
        # MuPDF emits separate entries for fill and stroke. Either can display the same glyphs.
        if (span["type"] != 3 and span["opacity"] != 0) or span["chars"] in visible:
            continue
        if not any(0 <= char[0] <= 0x10ffff and chr(char[0]).isalnum() for char in span["chars"]):
            continue
        box = _page_box(page, span["bbox"])
        if box is None:
            continue
        count += 1
        if len(locations) < MAX_BOXES:
            locations.append(box)
    if count:
        return {"page": page.number + 1, "invisible_spans": count, "boxes": locations}
    return None


def _glyphs(blocks):
    for block in blocks:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                yield from span["chars"]


def _significant(text):
    return "".join(char for char in unicodedata.normalize("NFD", text.translate(LIGATURES))
                   if not char.isspace() and unicodedata.category(char) != "Cf")


def _page_box(page, bounds):
    x0, y0, x1, y1 = bounds
    if not all(math.isfinite(side) for side in bounds):
        return None
    box = [round(side, 4) for side in (max(0, x0), max(0, y0), min(page.cropbox.width, x1), min(page.cropbox.height, y1))]
    return box if box[0] < box[2] and box[1] < box[3] else None


def inspect_replacement(page, copyable, glyph_blocks):
    """Compare copyable and glyph text with normalized spacing and ligatures."""
    stored = _significant("".join(glyph["c"] for glyph in _glyphs(copyable)))
    painted = _significant("".join(glyph["c"] for glyph in _glyphs(glyph_blocks)))
    if stored == painted:
        return None
    record = {"page": page.number + 1, "boxes": [], "locations_incomplete": True}
    # Match only the changed extent. Large changes remain findings with page-level coverage.
    prefix = 0
    size = min(len(painted), len(stored))
    while prefix < size and painted[prefix] == stored[prefix]:
        prefix += 1
    suffix = 0
    while suffix < size - prefix and painted[-1 - suffix] == stored[-1 - suffix]:
        suffix += 1
    first_end, second_end = len(painted) - suffix, len(stored) - suffix
    if max(first_end, second_end) - prefix > MAX_DIFF_CHARS:
        return record

    # Normalize each glyph separately for source positions. If the page's Unicode
    # normalization reorders marks across glyphs, retain the page-only finding.
    glyph_parts = [(_significant(glyph["c"]), glyph["bbox"]) for glyph in _glyphs(glyph_blocks)]
    if "".join(part for part, _ in glyph_parts) != painted:
        return record
    positions = [bbox for part, bbox in glyph_parts for _ in part]
    difference = SequenceMatcher(None, painted[prefix:first_end], stored[prefix:second_end], autojunk=False)
    boxes = []
    complete = True
    for operation, start, end, _, _ in difference.get_opcodes():
        if operation == "equal":
            continue
        start, end = prefix + start, prefix + end
        # Inserted copy text has no corresponding glyph. Save the nearest glyph
        # as an approximate location and disclose that the locations are incomplete.
        if start == end:
            complete = False
            start = min(start, len(positions) - 1)
            end = start + 1
        region = [box for bounds in positions[start:end] if (box := _page_box(page, bounds)) is not None]
        if not region:
            complete = False
            continue
        box = [min(row[0] for row in region), min(row[1] for row in region),
               max(row[2] for row in region), max(row[3] for row in region)]
        if box not in boxes:
            boxes.append(box)
    return {**record, "boxes": boxes[:MAX_BOXES], "locations_incomplete": not complete or len(boxes) > MAX_BOXES}


def build(pages, examined, locations, replacement_examined, replacement_differences):
    return {"schema": SCHEMA, "total_pages": pages, "examined_pages": examined, "locations": locations,
            "replacement_examined_pages": replacement_examined, "replacement_differences": replacement_differences,
            "coordinate_space": "unrotated-page-points", "authorship_confidence_contribution": 0}


def _valid_pages(pages, total):
    return (isinstance(pages, list) and all(type(page) is int and 1 <= page <= total for page in pages)
            and pages == sorted(set(pages)))


def _valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def validate(document):
    record = document.get("coverage", {}).get(KEY)
    if record is None:
        return
    total, examined = record["total_pages"], record["examined_pages"]
    schema = record["schema"]
    if (document["source"]["method"] != "pymupdf" or schema not in (LEGACY_SCHEMA, FULL_PAGE_SCHEMA, SCHEMA)
            or record["coordinate_space"] != "unrotated-page-points"
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or type(total) is not int or total < 1 or not _valid_pages(examined, total)
            or schema == LEGACY_SCHEMA and any(page > LEGACY_MAX_PAGES for page in examined)):
        raise ValueError("Invalid PDF text visibility coverage")
    previous = 0
    for row in record["locations"]:
        page, count, boxes = row["page"], row["invisible_spans"], row["boxes"]
        if (type(page) is not int or page not in examined or page <= previous
                or type(count) is not int or count < 1 or len(boxes) != min(count, MAX_BOXES)
                or not all(_valid_box(box) for box in boxes)):
            raise ValueError("Invalid invisible PDF text location")
        previous = page
    if schema == SCHEMA:
        examined = record["replacement_examined_pages"]
        differences = record["replacement_differences"]
        if not _valid_pages(examined, total) or not isinstance(differences, list):
            raise ValueError("Invalid PDF replacement text coverage")
        previous = 0
        for row in differences:
            page, boxes, incomplete = row["page"], row["boxes"], row["locations_incomplete"]
            if (type(page) is not int or page not in examined or page <= previous
                    or type(incomplete) is not bool or not isinstance(boxes, list)
                    or len(boxes) > MAX_BOXES or not (boxes or incomplete) or not all(_valid_box(box) for box in boxes)):
                raise ValueError("Invalid PDF replacement text location")
            previous = page
    elif "replacement_examined_pages" in record or "replacement_differences" in record:
        raise ValueError("Unexpected PDF replacement text coverage")


def verify_source(document, meta, *, frozen=False):
    validate(document)
    record = document.get("coverage", {}).get(KEY)
    source = meta.get(KEY)
    if not frozen and record is not None and record["schema"] in (LEGACY_SCHEMA, FULL_PAGE_SCHEMA) and source and source["schema"] == SCHEMA:
        # Replay the records the older extractors could produce with the same source text.
        source = {**source, "schema": record["schema"]}
        del source["replacement_examined_pages"], source["replacement_differences"]
        if record["schema"] == LEGACY_SCHEMA:
            source["examined_pages"] = [page for page in source["examined_pages"] if page <= LEGACY_MAX_PAGES]
            source["locations"] = [row for row in source["locations"] if row["page"] <= LEGACY_MAX_PAGES]
    if (record is not None or frozen) and record != source:
        raise ValueError("PDF visibility evidence differs from the source extraction")
