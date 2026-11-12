"""Source ranges actually measured by the checks behind a writing review card."""

from __future__ import annotations

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


def check_range(span: dict, size: int) -> tuple[int, int]:
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
        raise ValueError("Invalid document inspection coordinates")
    return start, end


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
