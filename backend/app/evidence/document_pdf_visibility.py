"""Page locations for explicitly non-rendering PDF text operations."""

from __future__ import annotations

import math

KEY = "pdf_text_visibility"
MAX_PAGES = 12
MAX_BOXES = 4
SCHEMA = "document-pdf-text-visibility/1"
ASSESSMENT_HEADLINE = "PDF writing assessment limited by invisible text"
LIMITATION = ("Searchable scans and accessibility overlays can contain invisible text. "
              "The recorded locations describe the PDF text layer and carry zero AI-authorship weight.")


def inspect_page(page, spans):
    """Record the geometry of invisible words without copying their contents."""
    width, height = page.cropbox.width, page.cropbox.height
    visible = {span["chars"] for span in spans if span["type"] in (0, 1) and span["opacity"] > 0}
    locations, count = [], 0
    for span in spans:
        # MuPDF emits separate entries for fill and stroke. Either can display the same glyphs.
        if (span["type"] != 3 and span["opacity"] != 0) or span["chars"] in visible:
            continue
        if not any(0 <= char[0] <= 0x10ffff and chr(char[0]).isalnum() for char in span["chars"]):
            continue
        x0, y0, x1, y1 = span["bbox"]
        if not all(math.isfinite(side) for side in (x0, y0, x1, y1)):
            continue
        x0, y0, x1, y1 = (round(side, 4) for side in (max(0, x0), max(0, y0), min(width, x1), min(height, y1)))
        if x0 >= x1 or y0 >= y1:
            continue
        count += 1
        if len(locations) < MAX_BOXES:
            locations.append([x0, y0, x1, y1])
    if count:
        return {"page": page.number + 1, "invisible_spans": count, "boxes": locations}
    return None


def build(pages, examined, locations):
    return {"schema": SCHEMA, "total_pages": pages, "examined_pages": examined, "locations": locations,
            "coordinate_space": "unrotated-page-points", "authorship_confidence_contribution": 0}


def validate(document):
    record = document.get("coverage", {}).get(KEY)
    if record is None:
        return
    total, examined = record["total_pages"], record["examined_pages"]
    if (document["source"]["method"] != "pymupdf" or record["schema"] != SCHEMA
            or record["coordinate_space"] != "unrotated-page-points"
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or type(total) is not int or total < 1 or not isinstance(examined, list)
            or any(type(page) is not int or not 1 <= page <= min(total, MAX_PAGES) for page in examined)
            or examined != sorted(set(examined))):
        raise ValueError("Invalid PDF text visibility coverage")
    previous = 0
    for row in record["locations"]:
        page, count, boxes = row["page"], row["invisible_spans"], row["boxes"]
        if (type(page) is not int or page not in examined or page <= previous
                or type(count) is not int or count < 1 or len(boxes) != min(count, MAX_BOXES)
                or any(len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                       or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]) for box in boxes)):
            raise ValueError("Invalid invisible PDF text location")
        previous = page


def verify_source(document, meta, *, frozen=False):
    validate(document)
    record = document.get("coverage", {}).get(KEY)
    if (record is not None or frozen) and record != meta.get(KEY):
        raise ValueError("PDF visibility evidence differs from the source extraction")
