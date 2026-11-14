"""Exact source scope and replay for the optional longer-passage writing check."""

from __future__ import annotations

from app.evidence.document_scope import check_range

LEGACY_SCHEMA = "document-passage-position-review/2"
SCHEMA = "document-passage-position-review/4"
SCHEMAS = {LEGACY_SCHEMA, SCHEMA}
POINT = 3.6821095168341915
MAX_CHARS = 5200


def geometry(text, mapping, method, sentences):
    from app.detectors.text.extract import _ocr_language

    if (method != "plain" or not 0 < len(text) <= MAX_CHARS or not 3 <= len(mapping) <= 5
            or not 9 <= len(sentences) <= 25 or not 160 <= sum(unit.n_words for unit in sentences) <= 680
            or "\ufffd" in text):
        return None
    offsets = {}
    normalized_end = source_end = 0
    for index, block in enumerate(mapping):
        ranges = block.get("source_ranges") or []
        if block.get("kind") != "text" or len(ranges) != 1:
            return None
        start, end, source_start, raw_end = ranges[0]
        if ((start, end) != (block.get("start"), block.get("end"))
                or start != normalized_end + (2 if index else 0) or not start < end <= len(text)
                or not source_end <= source_start < raw_end or end - start != raw_end - source_start):
            return None
        offsets[index] = (start, end, source_start - start)
        normalized_end, source_end = end, raw_end
    if normalized_end != len(text):
        return None
    language = _ocr_language(text)
    if language.get("code") != "en" or language.get("status") != "supported":
        return None
    spans = []
    previous = 0
    for unit in sentences:
        if unit.paragraph not in offsets:
            return None
        start, end, shift = offsets[unit.paragraph]
        if not start <= unit.start < unit.end <= end or unit.start < previous or text[unit.start:unit.end] != unit.text:
            return None
        previous = unit.end
        spans.append({"start": unit.start, "end": unit.end, "source_start": unit.start + shift,
                      "source_end": unit.end + shift})
    contexts = []
    for i in range(1, len(sentences) - 1):
        a, m, z = sentences[i - 1:i + 2]
        if (a.paragraph == m.paragraph == z.paragraph and 12 <= m.n_words <= 65
                and all(7 <= u.n_words <= 80 for u in (a, z))
                and all(u.text.endswith((".", "?", "!")) for u in (a, m, z))):
            contexts.append({"paragraph": m.paragraph, "sentences": spans[i - 1:i + 2]})
    if not contexts:
        return None
    return {"paragraphs": len(mapping), "sentence_count": len(sentences), "contexts": contexts}


def record(scope, *, readings=None, text=None, schema=SCHEMA):
    from app.evidence.document_passage_position import _amount

    common = {"schema": schema, "authorship_confidence_contribution": 0, **scope}
    if not readings or all(response is None for response in readings):
        return {**common, "status": "unassessed", "selected": None}
    contexts = [{**row, **({"readings": response} if response is not None else {})}
                for row, response in zip(scope["contexts"], readings, strict=True)]
    if any(response is None for response in readings):
        if schema != SCHEMA:
            raise ValueError("Partial writing readings require the current schema")
        return {**common, "status": "partial", "selected": None, "contexts": contexts}
    contrasts = []
    for responses in readings:
        left, center, right = (_amount(row["response_hex"]) for row in responses)
        contrasts.append(center - max(left, right))
    highest = max(contrasts)
    selected = contrasts.index(highest) if highest > POINT and contrasts.count(highest) == 1 else None
    result = {**common, "status": "flagged" if selected is not None else "no_flag", "selected": selected,
              "contexts": contexts}
    if selected is not None and text is not None:
        focus = scope["contexts"][selected]["sentences"][1]
        result["focus_text"] = text[focus["start"]:focus["end"]]
    return result


def validate(value, source):
    from app.evidence.document_passage_position import validate_readings

    required = {"schema", "status", "authorship_confidence_contribution", "paragraphs", "sentence_count",
                "contexts", "selected", "signal_ref"}
    if value.get("status") == "flagged":
        required.add("focus_text")
    schema = value.get("schema")
    if (set(value) != required or schema not in SCHEMAS or type(value["authorship_confidence_contribution"]) is not int
            or source.get("method") != "plain" or type(source.get("chars")) is not int
            or not 0 < source["chars"] <= MAX_CHARS
            or type(value["paragraphs"]) is not int or not 3 <= value["paragraphs"] <= 5
            or type(value["sentence_count"]) is not int or not 9 <= value["sentence_count"] <= 25
            or not isinstance(value["contexts"], list) or not 1 <= len(value["contexts"]) <= value["sentence_count"] - 2
            or (value["selected"] is not None and type(value["selected"]) is not int)):
        raise ValueError("Invalid longer writing position scope")
    contexts = []
    readings = []
    previous = (-1, -1)
    shifts = {}
    native_by_span = {}
    for context in value["contexts"]:
        if set(context) not in ({"paragraph", "sentences"}, {"paragraph", "sentences", "readings"}):
            raise ValueError("Invalid longer writing response context")
        paragraph, spans = context["paragraph"], context["sentences"]
        if type(paragraph) is not int or not max(0, previous[0]) <= paragraph < value["paragraphs"] or len(spans) != 3:
            raise ValueError("Invalid longer writing paragraph positions")
        end = -1
        for span in spans:
            a, b = check_range(span, source["chars"])
            if (set(span) != {"start", "end", "source_start", "source_end"} or a <= end
                    or type(span["source_start"]) is not int or type(span["source_end"]) is not int
                    or not a <= span["source_start"] < span["source_end"]
                    or span["source_end"] - span["source_start"] != b - a):
                raise ValueError("Invalid longer writing source range")
            shift = span["source_start"] - a
            if shifts.setdefault(paragraph, shift) != shift or native_by_span.setdefault((a, b), paragraph) != paragraph:
                raise ValueError("Inconsistent longer writing source mapping")
            end = b
        start = spans[1]["start"]
        if start <= previous[1]:
            raise ValueError("Longer writing centers are repeated or unordered")
        previous = (paragraph, start)
        contexts.append({"paragraph": paragraph, "sentences": spans})
        readings.append(validate_readings(context["readings"]) if "readings" in context else None)
    scope = {"paragraphs": value["paragraphs"], "sentence_count": value["sentence_count"], "contexts": contexts}
    expected = record(scope, readings=readings, schema=schema)
    if expected != {key: item for key, item in value.items() if key not in {"signal_ref", "focus_text"}}:
        raise ValueError("The longer writing position direction does not replay")
    if expected["selected"] is not None:
        focus = contexts[expected["selected"]]["sentences"][1]
        if not isinstance(value["focus_text"], str) or len(value["focus_text"]) != focus["end"] - focus["start"]:
            raise ValueError("Invalid longer writing source excerpt")
    return value


def verify_source(value, text, mapping, method, sentences):
    from app.detectors.text.extract import text_hash
    from app.evidence.document_passage_position import inputs

    scope = geometry(text, mapping, method, sentences)
    contexts = [{"paragraph": context["paragraph"], "sentences": context["sentences"]} for context in value["contexts"]]
    if (scope is None or scope["paragraphs"] != value["paragraphs"] or scope["sentence_count"] != value["sentence_count"]
            or scope["contexts"] != contexts):
        raise ValueError("Longer writing positions differ from the original source")
    if value["status"] == "unassessed":
        return False
    if (index := value["selected"]) is not None:
        focus = value["contexts"][index]["sentences"][1]
        if text[focus["start"]:focus["end"]] != value["focus_text"]:
            raise ValueError("The longer writing excerpt differs from the original source")
    for context in value["contexts"]:
        if "readings" not in context:
            continue
        for (content, _, _), reading in zip(inputs(text, context["sentences"]), context["readings"], strict=True):
            if text_hash(content) != reading["input_sha256"]:
                raise ValueError("Longer writing input differs from the original source")
    return True
