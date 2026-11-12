"""Project retained local responses onto mapped document text."""

from __future__ import annotations

from app.detectors.text.extract import text_hash
from app.evidence.document_scope import (
    PASSAGE_COVERAGE_KEY as KEY,
)
from app.evidence.document_scope import (
    PASSAGE_COVERAGE_MAX_RANGES as MAX_RANGES,
)
from app.evidence.document_scope import (
    PASSAGE_COVERAGE_SCHEMA as SCHEMA,
)
from app.evidence.document_scope import (
    PASSAGE_COVERAGE_STATES as STATES,
)
from app.evidence.document_scope import (
    check_range,
)
from app.evidence.document_scope import (
    validate_passage_coverage as validate,
)

LABELS = {
    "no_local_record": "No exact local record",
    "context_only": "Read as context only",
    "response_range": "Local response recorded",
}


def _location(block: dict, number: int) -> dict:
    location = {"source_block": number}
    for field in ("page", "mime_part", "quote_role"):
        if block.get(field) is not None:
            location[field] = block[field]
    if block.get("paragraph_index") is not None:
        location["paragraph"] = block["paragraph_index"] + 1
    elif block.get("paragraph") is not None:
        location["paragraph"] = block["paragraph"]
    return location


def _fill(mask: bytearray, start: int, end: int) -> None:
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(mask):
        raise ValueError("Invalid recorded text coverage coordinates")
    mask[start:end] = b"\x01" * (end - start)


def _complete(window: dict) -> bool:
    coverage = window.get("token_coverage") or {}
    tokens = coverage.get("tokens")
    return (type(tokens) is int and tokens > 0 and not coverage.get("omitted_tokens", 0)
            and coverage.get("covered_tokens", tokens) == tokens)


def _marks(result: dict, text: str) -> tuple[bytearray, bytearray]:
    size = len(text)
    context, response = bytearray(size), bytearray(size)
    source_hash = text_hash(text)
    for signal in result.get("signals", []):
        record = (signal.get("details") or {}).get("text_evidence") or {}
        if signal.get("status") != "ok" or record.get("version") not in {"scored-context/1", "scored-context/2"}:
            continue
        if record.get("extracted_text_sha256") != source_hash or record.get("offset_unit") != "unicode_codepoint":
            raise ValueError("Text coverage refers to another extraction")
        contexts = {item["id"]: item for item in record.get("contexts", [])}
        used = set()
        for window in record.get("windows", []):
            if window.get("score_kind") == "perplexity_ratio":
                scored = set(window["scored_token_indices"])
                for key in window["context_ids"]:
                    item = contexts[key]
                    source_start, _ = item["source_token_range"]
                    for index, (start, end) in enumerate(item["token_offsets"], source_start):
                        if key not in used:
                            _fill(context, start, end)
                        if index in scored:
                            _fill(response, start, end)
                    used.add(key)
                continue
            if not _complete(window):
                continue
            for span in window["spans"]:
                _fill(context, *check_range(span, size))
            _fill(response, *check_range(window["focus"], size))
        isolated = (record.get("isolated_paragraph_review") or {}).get("inputs", [])
        for window in isolated:
            if not _complete(window) or len(window["spans"]) != 1:
                continue
            start, end = check_range(window["spans"][0], size)
            if text_hash(text[start:end]) != window["input_sha256"]:
                raise ValueError("Isolated paragraph coverage refers to another text input")
            _fill(response, start, end)
    return context, response


def build(result: dict, text: str, mapping: list[dict]) -> dict:
    context, response = _marks(result, text)
    ranges: list[dict] = []
    counts = dict.fromkeys(STATES, 0)
    total = 0
    for number, block in enumerate(mapping, 1):
        start, end = check_range(block, len(text))
        current = None
        for index in range(start, end):
            if text[index].isspace():
                continue
            state = STATES[2 if response[index] else context[index]]
            counts[state] += 1
            if current is not None and current["state"] == state:
                current["end"] = index + 1
                current["characters"] += 1
                continue
            current = {"start": index, "end": index + 1, "state": state, "characters": 1,
                       **_location(block, number)}
            total += 1
            if len(ranges) < MAX_RANGES:
                ranges.append(current)
    return {"schema": SCHEMA, "count_unit": "non_whitespace_unicode_codepoint",
            "mapped_characters": sum(counts.values()), "counts": counts, "ranges": ranges,
            "omitted_ranges": total - len(ranges)}


def verify_source(document: dict, text: str, mapping: list[dict], *, result: dict | None = None) -> None:
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return
    validate(document)
    if result is not None and record != build(result, text, mapping):
        raise ValueError("Document passage coverage differs from the frozen responses")
    blocks = [check_range(block, len(text)) for block in mapping]
    if record["mapped_characters"] != sum(not char.isspace() for start, end in blocks for char in text[start:end]):
        raise ValueError("Document passage coverage differs from the extracted character count")
    for row in record["ranges"]:
        number, start, end = (row[key] for key in ("source_block", "start", "end"))
        if number > len(blocks):
            raise ValueError("Document passage coverage has an unknown source block")
        left, right = blocks[number - 1]
        expected = _location(mapping[number - 1], number)
        if (not left <= start < end <= right or set(row) != {*expected, "start", "end", "characters", "state"}
                or any(row.get(key) != value for key, value in expected.items())
                or row["characters"] != sum(not char.isspace() for char in text[start:end])):
            raise ValueError("Document passage coverage differs from the extracted source positions")
