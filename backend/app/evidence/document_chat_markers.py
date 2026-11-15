"""Locate complete assistant-style turn markers in mapped document text."""

from __future__ import annotations

import re
from bisect import bisect_right
from itertools import pairwise

from app.core.canonical import FLOAT_DP, canonical_hash
from app.evidence.document_native_locations import MAX_RANGES, clip_source_ranges
from app.evidence.document_repeat import block_anchor, validate_block_anchor
from app.evidence.document_scope import mapped_email_html

KEY = "chat_format_markers"
COMMITMENT = "chat_format_markers_sha256"
SCHEMA = "document-chat-format-markers/1"
TITLE = "Chat formatting markers"
SCOPE = "Checks intact opening and closing markers around an assistant-style message in mapped source text."
LIMITATION = "Anyone can type or copy these markers. They do not identify a provider or establish who wrote the surrounding text. This observation carries zero AI-authorship weight."
MAX_TURNS = 8
MAX_BODY = 16_384
METHODS = {"plain": "text", "python-docx": "docx", "pymupdf": "pdf", "email": "email"}

# Exact opening and closing forms documented in the provider-format inventory.
OPENINGS = {
    "<|start|>assistant<|channel|>final<|message|>": {"<|end|>", "<|return|>"},
    "<|im_start|>assistant": {"<|im_end|>"},
    "<|start_header_id|>assistant<|end_header_id|>": {"<|eot_id|>"},
    "<start_of_turn>model": {"<end_of_turn>"},
    "<|turn>model": {"<turn|>"},
    "<｜Assistant｜>": {"<｜end▁of▁sentence｜>"},
    "]~b]ai": {"[e~["},
    "<|im_assistant|>assistant<|im_middle|>": {"<|im_end|>"},
}
REQUIRES_SPACE = {"<|im_start|>assistant", "<start_of_turn>model", "<|turn>model", "]~b]ai"}
BLOCKERS = {"<|start|>", "<|im_start|>", "<|start_header_id|>", "<start_of_turn>", "<|turn>", "<｜User｜>",
            "<｜tool▁calls▁begin｜>", "<｜tool▁outputs▁begin｜>", "<|call|>", "<|tool_call>", "]~b]", "<minimax:tool_call>",
            "<|im_assistant|>", "<|im_user|>", "<|im_system|>", "<|tool_calls_section_begin|>"}
TOKENS = re.compile("|".join(map(re.escape, sorted(BLOCKERS | OPENINGS.keys() | set.union(*OPENINGS.values()),
                                                  key=lambda s: (-len(s), s)))))
FENCES = re.compile(r"^[ \t]{0,3}(\x60{3,}|~{3,})([^\n]*)$", re.M)


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def outside_fences(text):
    cursor, active = 0, ""
    for match in FENCES.finditer(text):
        marker, suffix = match.groups()
        if not active:
            yield cursor, match.start()
            active = marker
        elif marker[0] == active[0] and len(marker) >= len(active) and not suffix.strip():
            cursor, active = match.end(), ""
    if not active:
        yield cursor, len(text)


def pairs(text):
    for start, stop in outside_fences(text):
        for first, last in pairwise(TOKENS.finditer(text, start, stop)):
            opening = first[0]
            if (last[0] not in OPENINGS.get(opening, ()) or last.start() - first.end() > MAX_BODY
                    or opening in REQUIRES_SPACE and not text[first.end()].isspace()):
                continue
            if any(character.isalnum() for character in text[first.end():last.start()]):
                yield first, last


def supported(method, email):
    return method in METHODS and (method != "email" or email.get("body_content_type") == "text/plain" or mapped_email_html(email))


def _location(match, block, anchor):
    start, end = match.span()
    if not block["start"] <= start < end <= block["end"]:
        return None
    location = {"literal": match[0], "start": start, "end": end, "anchor": anchor}
    if anchor["kind"] != "pdf":
        ranges = clip_source_ranges(block.get("source_ranges", ()), start, end)
        if not ranges:
            return None
        location["ranges"] = ranges
    return location


def measure(text, mapping, meta):
    method, email = meta.get("method"), meta.get("email") or {}
    if not supported(method, email):
        return None
    all_starts = [block["start"] for block in mapping]
    anchors = {}
    for index, block in enumerate(mapping):
        if block.get("kind") != METHODS[method] or method == "email" and block.get("mime_part") != email.get("body_part"):
            continue
        if anchor := block_anchor(block):
            if anchor["kind"] == "pdf":
                anchor["bbox"] = [round(value, FLOAT_DP) for value in anchor["bbox"]]
            anchors[index] = anchor
    coverage = sum(mapping[i]["end"] - mapping[i]["start"] for i in anchors)
    if not coverage:
        return None

    def locate(token):
        index = bisect_right(all_starts, token.start()) - 1
        return _location(token, mapping[index], anchors[index]) if index in anchors else None

    turns, locations = 0, []
    for first, last in pairs(text):
        opener, closer = locate(first), locate(last)
        if opener is None or closer is None or opener["anchor"].get("page") != closer["anchor"].get("page"):
            continue
        turns += 1
        if len(locations) < MAX_TURNS:
            locations.append({"opener": opener, "closer": closer})
    return {"schema": SCHEMA, "scope": SCOPE, "limitation": LIMITATION, "checked_chars": coverage,
            "turns": turns, "locations": locations, "authorship_confidence_contribution": 0}


def _valid_ranges(ranges, anchor, width, *, html):
    if (not isinstance(ranges, list) or not 1 <= len(ranges) <= MAX_RANGES
            or any(not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                   or not anchor.get("source_start", 0) <= span[0] < span[1] <= anchor.get("source_end", float("inf"))
                   for span in ranges)
            or any(left[1] >= right[0] for left, right in pairwise(ranges))):
        return False
    length = sum(b - a for a, b in ranges)
    return len(ranges) == 1 and length == width if not html else length >= width


def validate(document, expected):
    coverage = document.get("coverage") or {}
    record = coverage.get(KEY)
    source = document["source"]
    if record is None and expected is None and source.get(COMMITMENT) is None:
        return
    method, email = source.get("method"), coverage.get("email") or {}
    if (not isinstance(record, dict) or not supported(method, email) or canonical_hash(record) != expected
            or source.get(COMMITMENT) != expected or record.keys() != {"schema", "scope", "limitation", "checked_chars",
                                                                      "turns", "locations", "authorship_confidence_contribution"}
            or (record["schema"], record["scope"], record["limitation"]) != (SCHEMA, SCOPE, LIMITATION)
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0):
        raise ValueError("Chat-format markers differ from the committed document scope")
    checked, turns, locations = record["checked_chars"], record["turns"], record["locations"]
    if (type(checked) is not int or not 0 < checked <= source["chars"] or type(turns) is not int or not 0 <= turns <= checked
            or not isinstance(locations, list) or len(locations) != min(turns, MAX_TURNS)):
        raise ValueError("Invalid chat-format marker coverage")
    previous = 0
    for pair in locations:
        if not isinstance(pair, dict) or pair.keys() != {"opener", "closer"}:
            raise ValueError("Invalid chat-format marker pair")
        opening, closing = pair["opener"], pair["closer"]
        for marker in (opening, closing):
            fields = {"literal", "start", "end", "anchor"}
            if method != "pymupdf":
                fields.add("ranges")
            if not isinstance(marker, dict) or marker.keys() != fields:
                raise ValueError("Invalid chat-format marker fields")
            literal, start, end, anchor = (marker[key] for key in ("literal", "start", "end", "anchor"))
            if (not isinstance(literal, str) or type(start) is not int or type(end) is not int
                    or not previous <= start < end <= source["chars"] or end - start != len(literal)
                    or not isinstance(anchor, dict) or anchor.get("kind") != METHODS[method]):
                raise ValueError("Invalid chat-format marker location")
            validate_block_anchor(anchor, len(literal))
            if (method == "email" and anchor.get("mime_part") != email.get("body_part")
                    or method != "pymupdf" and not _valid_ranges(marker["ranges"], anchor, len(literal), html=mapped_email_html(email))):
                raise ValueError("Invalid native chat-format marker position")
            previous = end
        if (closing["literal"] not in OPENINGS.get(opening["literal"], ())
                or not 0 < closing["start"] - opening["end"] <= MAX_BODY
                or opening["anchor"].get("page") != closing["anchor"].get("page")):
            raise ValueError("Invalid chat-format marker sequence")


def verify_source(document, text, mapping, meta):
    record = (document.get("coverage") or {}).get(KEY)
    validate(document, document["source"].get(COMMITMENT))
    if record is not None and measure(text, mapping, meta) != record:
        raise ValueError("Chat-format markers differ from the source text")


def location_label(marker):
    anchor = marker["anchor"]
    if anchor["kind"] == "pdf":
        return f"PDF page {anchor['page']}, text block {anchor['block']}"
    target = "Decoded source"
    if anchor["kind"] == "email":
        target = f"Decoded email part {anchor['mime_part']}"
    elif anchor["kind"] == "docx":
        target = "Word body path " + " / ".join(map(str, anchor["body_path"]))
    return target + " · characters " + ", ".join(f"[{a}, {b})" for a, b in marker["ranges"])
