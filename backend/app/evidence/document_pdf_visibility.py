"""Page locations for invisible PDF text and discrepancies in copyable text."""

from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher
from itertools import groupby

KEY = "pdf_text_visibility"
LEGACY_MAX_PAGES = 12
MAX_BOXES = 4
LEGACY_SCHEMA = "document-pdf-text-visibility/1"
FULL_PAGE_SCHEMA = "document-pdf-text-visibility/2"
REPLACEMENT_SCHEMA = "document-pdf-text-visibility/3"
SCHEMA = "document-pdf-text-visibility/4"
ASSESSMENT_HEADLINE = "PDF writing assessment limited by invisible text"
INCOMPLETE_HEADLINE = "PDF writing assessment limited by incomplete text visibility checks"
REPLACEMENT_HEADLINE = "PDF writing assessment limited by differing text layers"
COVERED_HEADLINE = "PDF writing assessment limited by covered text"
LIMITATION = ("Searchable scans and accessibility overlays can contain invisible text. "
              "The recorded locations describe the PDF text layer and carry zero AI-authorship weight.")
REPLACEMENT_LIMITATION = ("PDF accessibility and OCR corrections can legitimately change copyable text. "
                          "This comparison uses the PDF's glyph text and copyable text. It carries zero AI-authorship weight.")
COVERED_LIMITATION = ("The check looks for copyable text behind simple opaque rectangular fills in the page content. "
                      "It carries zero AI-authorship weight. A page without a finding is not verified as securely redacted.")
MAX_DIFF_CHARS = 768
MAX_PAINT_OPERATIONS = 8192
MAX_RECTANGLE_CANDIDATES = 96
MAX_RENDER_PIXELS = 1_000_000
LIGATURES = str.maketrans({char: unicodedata.normalize("NFKD", char) for char in "ﬀﬁﬂﬃﬄﬅﬆ"})


def covered_notice(findings):
    pages = ", ".join(str(row["page"]) for row in findings)
    subject = "page" if len(findings) == 1 else "pages"
    return (f"The PDF contains copyable text behind a solid cover on {subject} {pages}. "
            "The affected pages were excluded from writing checks.")


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


def _contains(outer, inner, padding=0):
    x0, y0, x1, y1 = outer
    a0, b0, a1, b1 = inner
    return x0 + padding <= a0 < a1 <= x1 - padding and y0 + padding <= b0 < b1 <= y1 - padding


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _union(boxes):
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _copyable_words(span, copy_origins):
    chars = [(chr(code) if 0 <= code <= 0x10ffff else " ", origin, bbox)
             for code, _, origin, bbox in span["chars"]]
    for space, group in groupby(chars, key=lambda char: char[0].isspace()):
        if space:
            continue
        word = list(group)
        origins = [(char, round(x, 2), round(y, 2)) for char, (x, y), _ in word if char.isalnum()]
        if len(origins) >= 2 and all(origin in copy_origins for origin in origins):
            yield _union([bbox for _, _, bbox in word])


def inspect_covered(page, spans, copyable):
    """Check paint order and rendered pixels for covered, copyable words."""
    import pymupdf

    text = [span for span in spans if span["type"] == 0 and span["opacity"] > 0]
    if not text:
        return None
    operations = page.get_bboxlog()
    if len(operations) > MAX_PAINT_OPERATIONS or len(text) > MAX_PAINT_OPERATIONS:
        raise ValueError("PDF paint inspection limit exceeded")
    first_text = min(span["seqno"] for span in text)
    if not any(op == "fill-path" for op, _ in operations[first_text + 1:]):
        return None
    drawings = page.get_cdrawings()
    covers = []
    for path in drawings:
        if (path["type"] not in ("f", "fs") or path["fill_opacity"] != 1 or path.get("layer")
                or path["seqno"] <= first_text or len(path["items"]) != 1 or path["items"][0][0] != "re"):
            continue
        box = path["items"][0][1]
        color = path["fill"]
        if len(color) != 3 or not all(0 <= value <= 1 for value in color):
            continue
        covers.append((path["seqno"], box, [round(value * 255) for value in color]))
    if len(covers) > MAX_RECTANGLE_CANDIDATES:
        raise ValueError("PDF rectangle inspection limit exceeded")
    if not covers:
        return None
    copy_origins = {(glyph["c"], round(glyph["origin"][0], 2), round(glyph["origin"][1], 2))
                    for glyph in _glyphs(copyable) if glyph["c"].isalnum()}
    display = None
    render_pixels = 0
    locations = set()
    page_box = (0, 0, page.cropbox.width, page.cropbox.height)
    for span in text:
        candidates = [(box, rgb) for seqno, box, rgb in covers
                      if seqno > span["seqno"] and _overlaps(box, span["bbox"])]
        if not candidates:
            continue
        words = list(_copyable_words(span, copy_origins))
        for cover, rgb in candidates:
            retained = [word for word in words if _contains(cover, word, padding=1) and _contains(page_box, word)]
            if not retained:
                continue
            box = _page_box(page, _union(retained))
            if box is None or tuple(box) in locations:
                continue
            clip = pymupdf.Rect(box) * page.rotation_matrix
            width, height = math.ceil(clip.width * 2) + 2, math.ceil(clip.height * 2) + 2
            render_pixels += width * height
            if render_pixels > MAX_RENDER_PIXELS:
                raise ValueError("PDF cover rendering limit exceeded")
            if display is None:
                display = page.get_displaylist(annots=False)
            pix = display.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip)
            if pix.width and pix.height and pix.is_unicolor and all(
                    abs(actual - expected) <= 2 for actual, expected in zip(pix.pixel(0, 0), rgb, strict=True)):
                locations.add(tuple(box))
    if locations:
        return {"page": page.number + 1, "covered_spans": len(locations),
                "boxes": [list(box) for box in sorted(locations)[:MAX_BOXES]]}
    return None


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


def build(pages, examined, locations, replacement_examined, replacement_differences, *, covered=None):
    record = {"schema": REPLACEMENT_SCHEMA, "total_pages": pages, "examined_pages": examined, "locations": locations,
              "replacement_examined_pages": replacement_examined, "replacement_differences": replacement_differences,
              "coordinate_space": "unrotated-page-points", "authorship_confidence_contribution": 0}
    if covered is not None:
        checked, findings = covered
        record.update(schema=SCHEMA, covered_examined_pages=checked, covered_locations=findings)
    return record


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
    if (document["source"]["method"] != "pymupdf" or schema not in (LEGACY_SCHEMA, FULL_PAGE_SCHEMA, REPLACEMENT_SCHEMA, SCHEMA)
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
    if schema in (REPLACEMENT_SCHEMA, SCHEMA):
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
    if schema == SCHEMA:
        covered = record["covered_examined_pages"]
        findings = record["covered_locations"]
        if not _valid_pages(covered, total) or not set(covered).issubset(record["examined_pages"]) or not isinstance(findings, list):
            raise ValueError("Invalid PDF covered-text coverage")
        previous = 0
        for row in findings:
            page, count, boxes = row["page"], row["covered_spans"], row["boxes"]
            if (set(row) != {"page", "covered_spans", "boxes"} or type(page) is not int
                    or page not in covered or page <= previous or type(count) is not int or count < 1
                    or not isinstance(boxes, list) or len(boxes) != min(count, MAX_BOXES)
                    or not all(_valid_box(box) for box in boxes) or boxes != sorted(boxes)
                    or len({tuple(box) for box in boxes}) != len(boxes)):
                raise ValueError("Invalid covered PDF text location")
            previous = page
    elif "covered_examined_pages" in record or "covered_locations" in record:
        raise ValueError("Unexpected PDF covered text coverage")


def verify_source(document, meta, *, frozen=False):
    validate(document)
    record = document.get("coverage", {}).get(KEY)
    source = meta.get(KEY)
    if (not frozen and record is not None and record["schema"] in (LEGACY_SCHEMA, FULL_PAGE_SCHEMA)
            and source and source["schema"] == REPLACEMENT_SCHEMA):
        # Replay the records the older extractors could produce with the same source text.
        source = {**source, "schema": record["schema"]}
        del source["replacement_examined_pages"], source["replacement_differences"]
        if record["schema"] == LEGACY_SCHEMA:
            source["examined_pages"] = [page for page in source["examined_pages"] if page <= LEGACY_MAX_PAGES]
            source["locations"] = [row for row in source["locations"] if row["page"] <= LEGACY_MAX_PAGES]
    if (record is not None or frozen) and record != source:
        raise ValueError("PDF visibility evidence differs from the source extraction")
