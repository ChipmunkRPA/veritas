"""Native coordinates for separate writing passages in mapped document formats."""

from __future__ import annotations

from math import isfinite

from app.core.canonical import FLOAT_DP
from app.evidence import document_pdf_visibility, document_word_visibility
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import _complete_pdf_fragments, mapped_email_html

KINDS = {"python-docx": "docx", "email": "email", "pymupdf": "pdf"}
PDF_POINTS = "unrotated-page-points"
MAX_BOXES = 3000


def eligible_source(method, metadata, word_review=None):
    if method == "python-docx":
        visibility = metadata.get(document_word_visibility.KEY) or {}
        return bool(visibility) and not visibility.get("hidden_runs") and (
            not word_review or word_review.get("status") == "no_match"
        )
    if method == "email":
        email = metadata.get("email") or {}
        return bool(email.get("body_part")) and (email.get("body_content_type") == "text/plain" or mapped_email_html(email))
    if method == "pymupdf":
        visibility = metadata.get(document_pdf_visibility.KEY) or {}
        pages = visibility.get("total_pages")
        if type(pages) is not int or pages < 1:
            return False
        for key in ("examined_pages", "replacement_examined_pages", "covered_examined_pages"):
            checked = visibility.get(key)
            if not isinstance(checked, list) or len(checked) != pages or any(page != i for i, page in enumerate(checked, 1)):
                return False
        return (not any(visibility.get(key) for key in ("locations", "replacement_differences", "covered_locations"))
                and (metadata.get("ocr") or {}).get("status") == "not_needed"
                and not (metadata.get("unmapped_pages") or metadata.get("partially_checked_pages")))
    return False


def checked_end(method, text, mapping, metadata, word_review=None):
    """Return the length of the supported, contiguous writing input."""
    if not eligible_source(method, metadata, word_review) or not mapping:
        return None
    kind = KINDS[method]
    previous = 0
    body_end = 0
    trailing_email = False
    for block in mapping:
        start, end = block["start"], block["end"]
        if (block.get("kind") != kind or start != (previous + 2 if previous else 0) or not start < end <= len(text)):
            return None
        previous = end
        if kind == "docx":
            if (len(block.get("body_path", ())) != 1 or not block.get("run_mapping_complete")
                    or not block.get("source_ranges")):
                return None
        elif kind == "email":
            role = block.get("quote_role")
            if block.get("mime_part") != metadata["email"]["body_part"] or not block.get("source_ranges"):
                return None
            if role in ("quoted", "signature"):
                trailing_email = True
            elif role == "body" and not trailing_email:
                body_end = end
            else:
                return None
        elif (block.get("extraction") != "text_layer" or block.get("page_rotation") != 0
              or not _complete_pdf_fragments(block, text, start, end)):
            return None
    if previous != len(text):
        return None
    if kind == "pdf" and len({block["page"] for block in mapping}) != metadata[document_pdf_visibility.KEY]["total_pages"]:
        return None
    return (body_end or None) if kind == "email" else previous


def locate(method, mapping, a, b):
    places = []
    box_count = 0
    for block in mapping:
        start, end = max(a, block["start"]), min(b, block["end"])
        if start >= end:
            continue
        place = {"start": start, "end": end}
        if method == "pymupdf":
            boxes = [[round(v, FLOAT_DP) for v in fragment["bbox"]]
                     for fragment in block["fragments"] if fragment["start"] < end and start < fragment["end"]]
            box_count += len(boxes)
            if not boxes or box_count > MAX_BOXES:
                return []
            place.update(page=block["page"], coordinate_space=PDF_POINTS, boxes=boxes)
        else:
            ranges = clip_source_ranges(block["source_ranges"], start, end)
            if not ranges:
                return []
            place["ranges"] = ranges
            if method == "python-docx":
                place.update(paragraph=block["paragraph_index"] + 1, body_path=block["body_path"])
            else:
                place["mime_part"] = block["mime_part"]
        places.append(place)
    return places


def _positive(value):
    return type(value) is int and value > 0


def _valid_spans(ranges):
    previous = 0
    if not isinstance(ranges, list) or not ranges:
        return False
    for span in ranges:
        if (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1]):
            return False
        previous = span[1]
    return True


def _valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and isfinite(v) for v in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def validate(method, metadata, places, start, end):
    if not isinstance(places, list) or not places:
        raise ValueError("Writing passage lacks an original document location")
    previous = start
    native_position = -1
    box_count = 0
    for place in places:
        if (not isinstance(place, dict) or type(place.get("start")) is not int or type(place.get("end")) is not int
                or place["start"] != previous or not previous < place["end"] <= end):
            raise ValueError("Original locations do not cover the writing passage")
        previous = place["end"] + 2
        if method == "pymupdf":
            page, boxes = place.get("page"), place.get("boxes")
            total = metadata[document_pdf_visibility.KEY]["total_pages"]
            if (set(place) != {"start", "end", "page", "coordinate_space", "boxes"} or not _positive(page)
                    or not native_position <= page <= total or place["coordinate_space"] != PDF_POINTS
                    or not isinstance(boxes, list) or not boxes or not all(_valid_box(box) for box in boxes)):
                raise ValueError("Invalid PDF writing passage location")
            native_position = page
            box_count += len(boxes)
        elif not _valid_spans(place.get("ranges")):
            raise ValueError("Invalid decoded writing passage location")
        elif method == "python-docx":
            path, paragraph = place.get("body_path"), place.get("paragraph")
            if (set(place) != {"start", "end", "ranges", "paragraph", "body_path"} or not _positive(paragraph)
                    or not isinstance(path, list) or len(path) != 1 or type(path[0]) is not int or path[0] <= native_position):
                raise ValueError("Invalid Word paragraph location")
            native_position = path[0]
        elif (set(place) != {"start", "end", "ranges", "mime_part"}
              or place["mime_part"] != metadata["email"]["body_part"]
              or place["ranges"][0][0] <= native_position):
            raise ValueError("Invalid decoded email body location")
        else:
            native_position = place["ranges"][-1][1]
    if previous - 2 != end or box_count > MAX_BOXES:
        raise ValueError("Original locations do not cover the writing passage")
