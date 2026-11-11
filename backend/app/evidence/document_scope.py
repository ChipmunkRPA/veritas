"""Source ranges actually measured by the checks behind a writing review card."""

from __future__ import annotations

VERSION = "document-evidence-5"
SCOPE_VERSIONS = frozenset({"document-evidence-4", VERSION})
CONTEXT_VERSIONS = frozenset({"document-evidence-3", *SCOPE_VERSIONS})
VERSIONS = frozenset({"document-evidence-1", "document-evidence-2", *CONTEXT_VERSIONS})


def _range(span: dict, size: int) -> tuple[int, int]:
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
        raise ValueError("Invalid document inspection coordinates")
    return start, end


def build(focus: dict, windows: list[dict], size: int) -> dict:
    """Use supporting input spans, or an exact and separately measured ratio reduction."""
    start, end = _range(focus, size)
    ranges = []
    for window in windows:
        if window.get("direction") != "supports_ai":
            continue
        ratio = window.get("score_kind") == "perplexity_ratio"
        spans = [_range(span, size) for span in window["scored_spans" if ratio else "spans"]]
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
    low, high = _range(packet["context"], size)
    if packet.get("inspection_scope") != scope or any(
        not low <= span["start"] < span["end"] <= high for span in scope["spans"]
    ):
        raise ValueError("Document inspection scope differs from the recorded supporting inputs")
