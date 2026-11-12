"""Source ranges actually measured by the checks behind a writing review card."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

VERSION = "document-evidence-5"
MATCHED_VERSION = "document-evidence-6"
PARAGRAPH_PRIORITY_VERSION = "document-evidence-7"
REPEAT_VERSIONS = frozenset({VERSION, MATCHED_VERSION, PARAGRAPH_PRIORITY_VERSION})
SCOPE_VERSIONS = frozenset({"document-evidence-4", *REPEAT_VERSIONS})
CONTEXT_VERSIONS = frozenset({"document-evidence-3", *SCOPE_VERSIONS})
VERSIONS = frozenset({"document-evidence-1", "document-evidence-2", *CONTEXT_VERSIONS})

PASSAGE_COVERAGE_KEY = "local_passage_coverage"
PASSAGE_COVERAGE_SCHEMA = "document-local-passage-coverage/1"
PASSAGE_COVERAGE_MAX_RANGES = 256
PASSAGE_COVERAGE_STATES = ("no_local_record", "context_only", "response_range")
CHECKPOINT_AGREEMENT_KEY = "writing_second_check"
CHECKPOINT_AGREEMENT_SCHEMA = "document-writing-second-check/1"
CHECKPOINT_AGREEMENT_EMAIL_SCHEMA = "document-writing-second-check/2"
CHECKPOINT_AGREEMENT_MAX = 6
CHECKPOINT_AGREEMENT_LABELS = {
    "corroborated": "Both checks gave an AI-like response",
    "different": "The checks gave different responses",
    "unavailable": "Second response unavailable",
}


def check_range(span: dict, size: int) -> tuple[int, int]:
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
        raise ValueError("Invalid document inspection coordinates")
    return start, end


def email_agreement_body_spans(mapping: list[dict], email: dict, size: int) -> list[dict] | None:
    """Identify the one mapped reply body before a trailing quotation or signature."""
    part = email.get("body_part")
    if email.get("body_content_type") != "text/plain" or not part or not mapping:
        return None
    body = []
    previous = 0
    excluded = False
    for block in mapping:
        start, end = check_range(block, size)
        role = block.get("quote_role")
        if (start != (previous + 2 if previous else 0) or block.get("kind") != "email"
                or block.get("mime_part") != part or not block.get("source_ranges")):
            return None
        if role == "body" and not excluded:
            body.append({"start": start, "end": end})
        elif role in {"quoted", "signature"}:
            excluded = True
        else:
            return None
        previous = end
    return body if excluded and len(body) in {1, 4} and previous == size else None


def checkpoint_window_in_scope(window: dict, body_spans: list[dict] | None) -> bool:
    return body_spans is None or all(
        any(block["start"] <= span["start"] < span["end"] <= block["end"] for block in body_spans)
        for span in [window["focus"], *window["spans"]]
    )


def validate_checkpoint_agreement(
    document: dict, text: str | None = None, *, result: dict | None = None, mapping: list[dict] | None = None,
) -> None:
    coverage = document.get("coverage") or {}
    record = coverage.get(CHECKPOINT_AGREEMENT_KEY)
    body_spans = None
    if record is not None:
        fields = {"schema", "ranges", "authorship_confidence_contribution"}
        size = document["source"]["chars"]
        if record.get("schema") == CHECKPOINT_AGREEMENT_EMAIL_SCHEMA:
            fields.add("body_spans")
            body_spans = record.get("body_spans")
            email = coverage.get("email")
            if (document["source"].get("method") != "email" or not isinstance(email, dict)
                    or email.get("body_content_type") != "text/plain"
                    or not isinstance(email.get("body_part"), str) or not email["body_part"]
                    or not isinstance(body_spans, list) or len(body_spans) not in {1, 4}):
                raise ValueError("Invalid second-check email body scope")
            boundaries = [check_range(block, size) for block in body_spans]
            if (any(set(block) != {"start", "end"} for block in body_spans) or boundaries[0][0] != 0
                    or any(left[1] + 2 != right[0] for left, right in zip(boundaries, boundaries[1:]))
                    or boundaries[-1][1] + 2 >= size
                    or mapping is not None and email_agreement_body_spans(mapping, email, size) != body_spans):
                raise ValueError("Second-check email scope differs from the mapped reply body")
        elif record.get("schema") != CHECKPOINT_AGREEMENT_SCHEMA:
            raise ValueError("Unknown second writing check schema")
        ranges = record["ranges"]
        if (set(record) != fields or type(record["authorship_confidence_contribution"]) is not int
                or record["authorship_confidence_contribution"] != 0
                or not isinstance(ranges, list) or len(ranges) > CHECKPOINT_AGREEMENT_MAX
                or text is not None and len(text) != size):
            raise ValueError("Invalid second writing check")
    if result is not None:
        for signal in result.get("signals", []):
            evidence = (signal.get("details") or {}).get("text_evidence") or {}
            if CHECKPOINT_AGREEMENT_KEY not in evidence:
                continue
            if evidence[CHECKPOINT_AGREEMENT_KEY] != record:
                raise ValueError("Second writing check differs from the frozen result")
            expected = [
                {**window["focus"], "spans": window["spans"], "input_sha256": window["input_sha256"]}
                for window in evidence["windows"] if float.fromhex(window["score_exact_hex"]) >= .5
                and checkpoint_window_in_scope(window, body_spans)
            ][:CHECKPOINT_AGREEMENT_MAX]
            if expected != [
                {key: row[key] for key in ("start", "end", "spans", "input_sha256")} for row in record["ranges"]
            ]:
                raise ValueError("Second writing check differs from the primary windows")
            break
        else:
            if record is not None:
                raise ValueError("Second writing check is missing from the frozen result")
    if record is None:
        return
    previous = 0
    for row in ranges:
        focus = check_range(row, size)
        spans = [check_range(span, size) for span in row["spans"]]
        state = row["state"]
        fields = {"start", "end", "spans", "input_sha256", "state"}
        if (focus[0] < previous or len(spans) not in {2, 3} or focus not in spans
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                or not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
                or state not in CHECKPOINT_AGREEMENT_LABELS
                or not checkpoint_window_in_scope({"focus": row, "spans": row["spans"]}, body_spans)):
            raise ValueError("Invalid second-check source coordinates")
        if state != "unavailable":
            fields |= {"response_exact_hex", "tokens"}
            try:
                response = float.fromhex(row["response_exact_hex"])
            except OverflowError as error:
                raise ValueError("Invalid second-check response") from error
            if (not 0 <= response <= 1 or row["response_exact_hex"] != response.hex()
                    or (response >= .5) != (state == "corroborated")
                    or type(row["tokens"]) is not int or not 0 < row["tokens"] <= 512):
                raise ValueError("Invalid second-check response")
        if set(row) != fields or any(set(span) != {"start", "end"} for span in row["spans"]):
            raise ValueError("Unexpected second-check evidence field")
        if text is not None:
            content = " ".join(text[a:b] for a, b in spans)
            if hashlib.sha256(content.encode()).hexdigest() != row["input_sha256"]:
                raise ValueError("Second writing check refers to a different source input")
        previous = focus[1]


def inspection_direction(window: dict) -> bool | None:
    if "matched_context_selected" in window:
        if "score_exact_hex" not in window:
            return None
        return window["matched_context_selected"] is True
    direction = window.get("direction")
    return direction == "supports_ai" if direction in {"supports_ai", "opposes_ai"} else None


def supports_inspection(window: dict) -> bool:
    return inspection_direction(window) is True


def build(focus: dict, windows: list[dict], size: int) -> dict:
    """Use supporting input spans, or an exact and separately measured ratio reduction."""
    start, end = check_range(focus, size)
    matched = [window for window in windows if window.get("matched_context_selected") is True]
    if matched:
        if len(matched) != 1:
            raise ValueError("A matched writing cue needs exactly one recorded response")
        spans = [check_range(span, size) for span in matched[0]["spans"]]
        if len(spans) != 3 or spans[1] != (start, end) or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("A matched writing cue needs its exact preceding, middle and following sentence ranges")
        return {"basis": "scored_context", "spans": [{"start": a, "end": b} for a, b in spans]}
    ranges = []
    for window in windows:
        if not supports_inspection(window):
            continue
        ratio = window.get("score_kind") == "perplexity_ratio"
        spans = [check_range(span, size) for span in window["scored_spans" if ratio else "spans"]]
        if ratio and window.get("neighbor_expansion") is False and spans == [(start, end)]:
            return {"basis": "ratio_reduction", "spans": [{"start": start, "end": end}]}
        ranges.extend(spans)
    if not ranges:
        raise ValueError("A document inspection range needs an actual supporting response")
    merged: list[dict[str, int]] = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], right)
        else:
            merged.append({"start": left, "end": right})
    return {"basis": "scored_context", "spans": merged}


def validate(packet: dict, size: int) -> None:
    scope = build(packet["focus"], packet["supporting_windows"], size)
    low, high = check_range(packet["context"], size)
    if packet.get("inspection_scope") != scope or any(
        not low <= span["start"] < span["end"] <= high for span in scope["spans"]
    ):
        raise ValueError("Document inspection scope differs from the recorded supporting inputs")


def validate_passage_coverage(document: dict) -> None:
    record = (document.get("coverage") or {}).get(PASSAGE_COVERAGE_KEY)
    if record is None:
        return
    size = document["source"]["chars"]
    ranges, counts, omitted = (record[key] for key in ("ranges", "counts", "omitted_ranges"))
    if (record["schema"] != PASSAGE_COVERAGE_SCHEMA or record["count_unit"] != "non_whitespace_unicode_codepoint"
            or not isinstance(ranges, list) or len(ranges) > PASSAGE_COVERAGE_MAX_RANGES or type(omitted) is not int
            or omitted < 0 or omitted and len(ranges) != PASSAGE_COVERAGE_MAX_RANGES or not isinstance(counts, dict) or set(counts) != set(PASSAGE_COVERAGE_STATES)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or type(record["mapped_characters"]) is not int
            or not 0 <= record["mapped_characters"] == sum(counts.values()) <= size):
        raise ValueError("Invalid document passage coverage summary")
    displayed = Counter()
    previous_end, previous_block = 0, 0
    for row in ranges:
        start, end = check_range(row, size)
        amount, block, state = (row[key] for key in ("characters", "source_block", "state"))
        if (start < previous_end or type(block) is not int or block < max(1, previous_block)
                or type(amount) is not int or not 0 < amount <= end - start or state not in PASSAGE_COVERAGE_STATES):
            raise ValueError("Invalid document passage coverage range")
        previous_end, previous_block = end, block
        displayed[state] += amount
    if any(displayed[key] > value or not omitted and displayed[key] != value for key, value in counts.items()):
        raise ValueError("Document passage coverage counts differ from the saved ranges")
