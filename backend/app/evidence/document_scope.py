"""Source ranges actually measured by the checks behind a writing review card."""

from __future__ import annotations

VERSION = "document-evidence-5"
MATCHED_VERSION = "document-evidence-6"
REPEAT_VERSIONS = frozenset({VERSION, MATCHED_VERSION})
SCOPE_VERSIONS = frozenset({"document-evidence-4", *REPEAT_VERSIONS})
CONTEXT_VERSIONS = frozenset({"document-evidence-3", *SCOPE_VERSIONS})
VERSIONS = frozenset({"document-evidence-1", "document-evidence-2", *CONTEXT_VERSIONS})


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
