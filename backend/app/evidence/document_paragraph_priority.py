"""Prioritize one retained passage supported by two distinct recorded inputs."""

from __future__ import annotations

from app.core.canonical import FLOAT_DP, canonical_hash
from app.evidence import document_matched_context, document_scope

KEY = "isolated_paragraph_priority"
SCHEMA = "document-isolated-paragraph-priority/1"
NOTE = (
    "This passage appears first because the AI-like response persisted when the same check read the whole paragraph. "
    "This ordering carries zero weight in the overall assessment."
)


def eligible(text: str, mapping: list[dict], method: str | None) -> bool:
    """Stay within the four-paragraph, mapped English pilot."""
    if method != "plain" or len(mapping) != 4 or not 145 <= len(text.split(None, 470)) <= 470:
        return False
    ranges = [document_scope.check_range(block, len(text)) for block in mapping]
    if (any(block.get("kind") != "text" or not block.get("source_ranges") for block in mapping)
            or ranges[0][0] != 0 or ranges[-1][1] != len(text)
            or any("\n\n" in text[a:b] for a, b in ranges)
            or any(text[a[1]:b[0]] != "\n\n" for a, b in zip(ranges, ranges[1:]))):
        return False
    from app.detectors.text.extract import _ocr_language

    language = _ocr_language(text)
    return language.get("code") == "en" and language.get("status") == "supported"


def recorded_ref(result: dict) -> dict | None:
    """Read the frozen opt-in and verify the internal detector binding."""
    from app.api.public_projection import alias_signal

    seen = set()
    ref = None
    for signal in result.get("signals", []):
        evidence = (signal.get("details") or {}).get("text_evidence") or {}
        if KEY not in evidence:
            continue
        if (signal.get("status") != "ok" or evidence[KEY] != SCHEMA or not evidence.get("isolated_paragraph_review")
                or not document_matched_context.matches(signal.get("detector_id"), signal.get("detector_version"),
                                                        signal.get("model_revisions") or {})):
            raise ValueError("Paragraph review order does not belong to the pinned check")
        digest = canonical_hash(evidence)
        if digest in seen:
            continue
        if ref is not None:
            raise ValueError("More than one check supplied a paragraph review order")
        seen.add(digest)
        ref = {"id": alias_signal(signal["id"]), "display_id": signal.get("display_id")}
    return ref


def _rank(packet: dict, ref: dict) -> tuple[float, int] | None:
    isolated = (packet.get("context_comparison") or {}).get("isolated_paragraph")
    if not isolated or isolated["signal_ref"] != ref or isolated["supports_inspection"] is not True:
        return None
    context = isolated["context"]
    paragraph = isolated["paragraph"]
    spans = context["spans"]
    focus = packet["focus"]
    if (len(spans) != 3 or spans[1] != {key: focus[key] for key in ("start", "end")}
            or context["input_sha256"] == paragraph["input_sha256"]
            or not document_matched_context.one_complete_window(context["token_coverage"])
            or not document_matched_context.one_complete_window(paragraph["token_coverage"])
            or not any(window["signal_ref"] == ref and window["input_sha256"] == context["input_sha256"]
                       and window["spans"] == spans and round(window["score"], FLOAT_DP) == round(context["score"], FLOAT_DP)
                       and window["direction"] == "supports_ai" for window in packet["supporting_windows"])):
        return None
    try:
        scores = [float.fromhex(item["score_exact_hex"]) for item in (context, paragraph)]
    except OverflowError as error:
        raise ValueError("Invalid exact response for paragraph priority") from error
    if not all(0.5 <= score <= 1 for score in scores):
        raise ValueError("Paragraph priority needs two complete AI-like responses")
    return -scores[0], focus["start"]


def preferred_packet(packets: list[dict], ref: dict) -> dict | None:
    candidates = [(rank, packet) for packet in packets if (rank := _rank(packet, ref)) is not None]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _record(packet: dict, ref: dict) -> dict:
    return {"schema": SCHEMA, "packet_id": packet["id"], "signal_ref": ref,
            "authorship_confidence_contribution": 0}


def apply(document: dict, result: dict, text: str, mapping: list[dict]) -> None:
    ref = recorded_ref(result)
    if ref is None:
        return
    if not eligible(text, mapping, document["source"].get("method")):
        raise ValueError("Frozen paragraph review order is outside its mapped scope")
    packets = document["packets"]
    if first := preferred_packet(packets, ref):
        document["version"] = document_scope.PARAGRAPH_PRIORITY_VERSION
        document["coverage"][KEY] = _record(first, ref)
        document["coverage"]["selection_policy"] += " One retained passage may appear first after a separate whole-paragraph read."
        document["packets"] = [first, *(packet for packet in packets if packet is not first)]


def validate(document: dict) -> None:
    """Recompute the first card from the publicly recorded measurements."""
    record = (document.get("coverage") or {}).get(KEY)
    if document.get("version") != document_scope.PARAGRAPH_PRIORITY_VERSION:
        if record is not None:
            raise ValueError("Historical document evidence cannot contain paragraph priority")
        return
    if not isinstance(record, dict) or document["source"].get("method") != "plain":
        raise ValueError("Missing mapped paragraph priority record")
    packets = document["packets"]
    ref = record["signal_ref"]
    preferred = preferred_packet(packets, ref)
    source_order = sorted(packets, key=lambda packet: (packet["focus"]["start"], packet["focus"]["end"]))
    expected_ids = [f"T{index:02d}" for index in range(1, len(packets) + 1)]
    if (preferred is None or record != _record(preferred, ref)
            or type(record["authorship_confidence_contribution"]) is not int
            or [packet["id"] for packet in source_order] != expected_ids
            or len(packets) > 6 or packets != [preferred, *(packet for packet in source_order if packet is not preferred)]):
        raise ValueError("Recorded paragraph priority differs from the retained passages")


def verify_source(document: dict, text: str, mapping: list[dict], *, result: dict | None = None) -> None:
    record = (document.get("coverage") or {}).get(KEY)
    ref = recorded_ref(result) if result is not None else (record or {}).get("signal_ref")
    if ref is None and record is None:
        return
    if ref is None or not eligible(text, mapping, document["source"].get("method")):
        raise ValueError("Paragraph priority differs from the frozen opt-in or mapped source")
    if result is not None:
        from app.evidence.document_context import build

        packets = [{**packet, "context_comparison": build(result, text, packet["focus"])} for packet in document["packets"]]
        preferred = preferred_packet(packets, ref)
        if record != (_record(preferred, ref) if preferred is not None else None):
            raise ValueError("Paragraph priority differs from the frozen classifier inputs")
    validate(document)
