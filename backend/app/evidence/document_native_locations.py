"""Locate inspected writing in its original decoded text or Word paragraph."""

from __future__ import annotations

from app.evidence.document_scope import EMAIL_HTML_COORDINATES

KEY = "original_text_locations"
SCHEMA = "document-original-text-locations/1"
MAX_RANGES = 16


def _clip(ranges, start, end):
    """Clip one mapped interval and preserve any gaps in original HTML."""
    if not 0 <= start < end:
        return []
    cursor = start
    clipped = []
    for a, b, lo, hi in ranges:
        if b <= cursor:
            continue
        if a > cursor or b <= a or not 0 <= lo < hi:
            return []
        stop = min(b, end)
        if b - a == hi - lo:
            lo, hi = lo + cursor - a, lo + stop - a
        elif (cursor, stop) != (a, b):
            return []
        if clipped and clipped[-1][1] == lo:
            clipped[-1][1] = hi
        elif clipped and clipped[-1][1] > lo:
            return []
        else:
            clipped.append([lo, hi])
            if len(clipped) > MAX_RANGES:
                return []
        cursor = stop
        if cursor == end:
            return clipped
    return []


def _coordinate(source, coverage, anchor):
    method = source.get("method")
    kind = anchor.get("kind")
    if method == "plain" and kind == "text":
        return "decoded_text"
    if method == "python-docx" and kind == "docx":
        return "word_paragraph"
    if method == "email" and kind == "email":
        email = coverage.get("email") or {}
        if anchor.get("mime_part") != email.get("body_part"):
            return None
        if email.get("body_content_type") == "text/plain":
            return "decoded_email"
        if email.get("body_content_type") == "text/html" and email.get("body_coordinates") == EMAIL_HTML_COORDINATES:
            return "decoded_email_html"
    return None


def build(document):
    coverage = document.get("coverage") or {}
    locations = {}
    for packet in document["packets"]:
        rows = []
        for index, anchor in enumerate(packet["anchors"]):
            coordinate = _coordinate(document["source"], coverage, anchor)
            if not coordinate:
                continue
            start, end = anchor["selected_start"], anchor["selected_end"]
            ranges = _clip(anchor.get("source_ranges") or [], start, end)
            if ranges:
                rows.append({"anchor_index": index, "coordinate": coordinate, "ranges": ranges})
        if rows:
            locations[packet["id"]] = rows
    return {"schema": SCHEMA, "packets": locations} if locations else None


def validate(document):
    record = (document.get("coverage") or {}).get(KEY)
    if record is not None and record != build(document):
        raise ValueError("Original text locations differ from their mapped evidence spans")
