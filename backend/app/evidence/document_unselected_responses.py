"""Source-bound classifier responses outside selected passage cards."""

from __future__ import annotations

import hashlib
import math
import re
from bisect import bisect_right

from app.core.canonical import canonical_hash
from app.evidence.document_scope import check_range

KEY = "unselected_passage_responses"
SCHEMA = "document-unselected-passage-responses/1"
MAX_ROWS = 12
MAX_INPUT_CHARS = 1800
REPORT_EXCERPT_CHARS = 200
DIRECTIONS = {"supports_ai": "AI-like response", "opposes_ai": "Response leaned away from AI in this window"}
SOURCE_FIELDS = {
    "text": {"paragraph"},
    "docx": {"paragraph"},
    "pdf": {"page", "extraction"},
    "email": {"mime_part", "quote_role"},
}
SHA = re.compile(r"[a-f0-9]{64}")
CHECK = re.compile(r"S[0-9]{2,6}")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _text(spans: list[dict], text: str) -> str:
    return " ".join(text[span["start"]:span["end"]] for span in spans)


def _locations(spans, mapping, starts):
    sources = []
    for span in spans:
        index = bisect_right(starts, span["start"]) - 1
        if index < 0 or mapping[index]["end"] < span["end"]:
            return None
        block = mapping[index]
        kind = block.get("kind")
        if kind not in SOURCE_FIELDS:
            return None
        if kind == "pdf":
            locator = {"page": block.get("page"), "extraction": block.get("extraction", "text_layer")}
        elif kind == "email":
            locator = {"mime_part": block.get("mime_part"), "quote_role": block.get("quote_role")}
        else:
            paragraph = block.get("paragraph_index")
            locator = {"paragraph": paragraph + 1 if type(paragraph) is int else block.get("paragraph", index + 1)}
        source = {"source_block": index + 1, "kind": kind, **locator}
        if not _valid_location(source):
            return None
        sources.append(source)
    return sources


def _complete(window):
    proof = window.get("token_coverage") or {}
    tokens = proof.get("tokens")
    return (type(tokens) is int and tokens > 0 and not proof.get("omitted_tokens", 0)
            and proof.get("covered_tokens", tokens) == tokens)


def _value(window, key, default=None):
    stored = window.get(key, default)
    if exact := window.get(f"{key}_exact_hex"):
        return float.fromhex(exact)
    return stored


def build(result: dict, text: str, mapping: list[dict], packets: list[dict]) -> dict | None:
    selected = {(packet["focus"]["start"], packet["focus"]["end"]) for packet in packets}
    starts = [block["start"] for block in mapping]
    seen_signals, seen_responses = set(), set()
    eligible = []
    for signal in result.get("signals", []):
        record = (signal.get("details") or {}).get("text_evidence") or {}
        label = signal.get("display_id")
        if (signal.get("status") != "ok" or not isinstance(label, str) or not CHECK.fullmatch(label)
                or record.get("version") not in {"scored-context/1", "scored-context/2"}):
            continue
        key = (signal.get("detector_id", signal.get("id")), canonical_hash(record))
        if key in seen_signals:
            continue
        seen_signals.add(key)
        for window in record.get("windows", []):
            if window.get("score_kind", "classifier_probability") != "classifier_probability" or not _complete(window):
                continue
            focus, spans = window["focus"], window["spans"]
            a, b = check_range(focus, len(text))
            if (a, b) in selected or not 1 <= len(spans) <= 3:
                continue
            size = sum(end - start for start, end in (check_range(span, len(text)) for span in spans)) + len(spans) - 1
            if not 0 < size <= MAX_INPUT_CHARS or (sources := _locations(spans, mapping, starts)) is None:
                continue
            input_text = _text(spans, text)
            pin = window["input_sha256"]
            _require(_digest(input_text) == pin and focus in spans, "Response input differs from the frozen text")
            response = _value(window, "score")
            boundary = _value(window, "threshold", 0.5)
            _require(type(response) in {int, float} and type(boundary) in {int, float}
                     and math.isfinite(response) and 0 <= response <= 1 and math.isfinite(boundary)
                     and boundary > 0 and window.get("ai_when", "gte") == "gte", "Invalid classifier response")
            response_key = (a, b, label, pin)
            if response_key in seen_responses:
                continue
            seen_responses.add(response_key)
            eligible.append({"focus": focus, "spans": spans, "sources": sources, "check": label, "input_sha256": pin,
                             "direction": "supports_ai" if response >= boundary else "opposes_ai", "response": round(response, 6),
                             "text": input_text})
    if not eligible:
        return None
    eligible.sort(key=lambda row: (row["focus"]["start"], row["focus"]["end"], row["check"], row["input_sha256"]))
    rows = [{"id": f"U{index:02d}", **row} for index, row in enumerate(eligible[:MAX_ROWS], 1)]
    return {"schema": SCHEMA, "affects_overall_confidence": False, "eligible_windows": len(eligible),
            "omitted_windows": len(eligible) - len(rows), "rows": rows}


def _valid_location(source):
    kind = source["kind"]
    if (kind not in SOURCE_FIELDS or set(source) != {"kind", "source_block", *SOURCE_FIELDS[kind]}
            or type(source["source_block"]) is not int or source["source_block"] < 1):
        return False
    if kind == "pdf":
        return type(source["page"]) is int and source["page"] >= 1 and source["extraction"] in {"text_layer", "ocr"}
    if kind == "email":
        return (isinstance(source["mime_part"], str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", source["mime_part"]) is not None
                and source["quote_role"] in {"body", "quoted", "signature"})
    return type(source["paragraph"]) is int and source["paragraph"] >= 1


def validate(document: dict, *, published_checks: set[str] | None = None) -> None:
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return
    size = document["source"]["chars"]
    rows, count, omitted = record["rows"], record["eligible_windows"], record["omitted_windows"]
    _require(set(record) == {"schema", "affects_overall_confidence", "eligible_windows", "omitted_windows", "rows"}
             and record["schema"] == SCHEMA and record["affects_overall_confidence"] is False
             and isinstance(rows, list) and type(count) is type(omitted) is int and 1 <= len(rows) == min(count, MAX_ROWS)
             and omitted == count - len(rows), "Invalid unselected passage response summary")
    selected = {(packet["focus"]["start"], packet["focus"]["end"]) for packet in document["packets"]}
    previous = (-1, -1, "", "")
    for index, row in enumerate(rows, 1):
        a, b = check_range(row["focus"], size)
        spans, pin, source, label = (row[key] for key in ("spans", "input_sha256", "sources", "check"))
        intervals = [check_range(span, size) for span in spans]
        response, text = row["response"], row["text"]
        key = (a, b, label, pin)
        _require(set(row) == {"id", "focus", "spans", "sources", "check", "input_sha256", "direction", "response", "text"}
                 and row["id"] == f"U{index:02d}" and key > previous and (a, b) not in selected
                 and len(spans) == len(source) in {1, 2, 3} and (a, b) in intervals
                 and all(set(span) == {"start", "end"} for span in [row["focus"], *spans])
                 and all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
                 and CHECK.fullmatch(label) and (published_checks is None or label in published_checks)
                 and SHA.fullmatch(pin) and isinstance(text, str) and _digest(text) == pin
                 and len(text) == sum(end - start for start, end in intervals) + len(spans) - 1 <= MAX_INPUT_CHARS
                 and row["direction"] in DIRECTIONS and type(response) in {int, float} and 0 <= response <= 1,
                 "Invalid unselected passage response")
        _require(all(_valid_location(location) for location in source), "Invalid response source location")
        previous = key


def verify_source(document: dict, text: str, mapping: list[dict], *, result: dict | None = None) -> int:
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return 0
    validate(document)
    starts = [block["start"] for block in mapping]
    for row in record["rows"]:
        _require(row["text"] == _text(row["spans"], text) and row["sources"] == _locations(row["spans"], mapping, starts),
                 "Unselected passage response differs from its original text or location")
    if result is not None:
        _require(record == build(result, text, mapping, document["packets"]),
                 "Unselected passage responses differ from the frozen classifier records")
    return len(record["rows"])
