"""Project retained local responses onto mapped document text."""

from __future__ import annotations

from app.detectors.text.extract import text_hash
from app.evidence import document_long_passage_position, document_passage_position
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
    PASSAGE_POSITION_COVERAGE_SCHEMA,
    PDF_PAGE_COVERAGE_KEY,
    PDF_PAGE_COVERAGE_SCHEMA,
    check_range,
    page_gaps,
)
from app.evidence.document_scope import (
    validate_passage_coverage as validate,
)

LABELS = {
    "no_local_record": "No exact local record",
    "context_only": "Read as context only",
    "response_range": "Local response recorded",
}
PDF_EXTRACTION_LABELS = {"text_layer": "Selectable text", "ocr": "Recognized text",
                         "mixed": "Selectable and recognized text"}


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


def _marks(result: dict, text: str, *, include_position: bool) -> tuple[bytearray, bytearray, bool]:
    size = len(text)
    context, response = bytearray(size), bytearray(size)
    source_hash = text_hash(text)
    position_measured = False
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
        if not include_position:
            continue
        position = record.get(document_passage_position.KEY) or {}
        if position.get("status") not in {"flagged", "no_flag"}:
            continue
        if position.get("schema") in document_passage_position.SHORT_SCHEMAS:
            groups = [position]
        elif position.get("schema") == document_long_passage_position.SCHEMA:
            groups = position["contexts"]
        else:
            raise ValueError("Unknown writing position evidence")
        for group in groups:
            spans = group["sentences"]
            ranges = [check_range(span, size) for span in spans]
            inputs = document_passage_position.inputs(text, spans)
            readings = document_passage_position.validate_readings(group["readings"])
            for (start, end), (value, _, _), reading in zip(ranges, inputs, readings, strict=True):
                if text_hash(value) != reading["input_sha256"]:
                    raise ValueError("Writing position coverage refers to another text input")
                _fill(response, start, end)
            position_measured = True
    return context, response, position_measured


def _page_sources(meta: dict, mapping: list[dict]) -> dict[int, str]:
    total = meta["pages"]
    if meta["method"] != "pymupdf" or type(total) is not int or total < 1:
        raise ValueError("Page evidence requires a counted PDF extraction")
    modes: dict[int, set[str]] = {}
    for block in mapping:
        page, extraction = block.get("page"), block.get("extraction", "text_layer")
        if (block.get("kind") != "pdf" or type(page) is not int or not 1 <= page <= total
                or extraction not in {"text_layer", "ocr"}):
            raise ValueError("Text block does not map to a known PDF page")
        modes.setdefault(page, set()).add(extraction)
    return {page: "mixed" if len(found) > 1 else next(iter(found)) for page, found in sorted(modes.items())}


def build(result: dict, text: str, mapping: list[dict], *, pdf_meta: dict | None = None,
          include_position: bool = True) -> dict:
    context, response, position_measured = _marks(result, text, include_position=include_position)
    ranges: list[dict] = []
    counts = dict.fromkeys(STATES, 0)
    sources = _page_sources(pdf_meta, mapping) if pdf_meta is not None else {}
    page_counts = {page: dict.fromkeys(STATES, 0) for page in sources}
    total = 0
    for number, block in enumerate(mapping, 1):
        start, end = check_range(block, len(text))
        counts_on_page = page_counts.get(block.get("page"))
        current = None
        for index in range(start, end):
            if text[index].isspace():
                continue
            state = STATES[2 if response[index] else context[index]]
            counts[state] += 1
            if counts_on_page is not None:
                counts_on_page[state] += 1
            if current is not None and current["state"] == state:
                current["end"] = index + 1
                current["characters"] += 1
                continue
            current = {"start": index, "end": index + 1, "state": state, "characters": 1,
                       **_location(block, number)}
            total += 1
            if len(ranges) < MAX_RANGES:
                ranges.append(current)
    record = {"schema": PASSAGE_POSITION_COVERAGE_SCHEMA if position_measured else SCHEMA,
              "count_unit": "non_whitespace_unicode_codepoint",
              "mapped_characters": sum(counts.values()), "counts": counts, "ranges": ranges,
              "omitted_ranges": total - len(ranges)}
    if pdf_meta is not None:
        partial = {page["page"] for page in pdf_meta.get("partially_checked_pages", [])}
        record[PDF_PAGE_COVERAGE_KEY] = {"schema": PDF_PAGE_COVERAGE_SCHEMA, "page_count": pdf_meta["pages"],
            "mapped_pages": [{"page": page, "extraction": mode, "unchecked_image_areas": page in partial, "counts": page_counts[page]}
                             for page, mode in sources.items()],
            "unmapped_page_ranges": page_gaps(pdf_meta["pages"], list(sources))}
    return record


def verify_source(document: dict, text: str, mapping: list[dict], *, result: dict | None = None,
                  meta: dict | None = None) -> None:
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return
    validate(document)
    pdf = record.get(PDF_PAGE_COVERAGE_KEY)
    if pdf is not None and meta is None:
        raise ValueError("PDF page coverage has no source extraction metadata")
    if result is not None and record != build(result, text, mapping, pdf_meta=meta if pdf is not None else None,
                                             include_position=record["schema"] == PASSAGE_POSITION_COVERAGE_SCHEMA):
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
    if pdf is not None:
        _verify_pdf_pages(pdf, meta, text, mapping, blocks)


def _verify_pdf_pages(pdf: dict, meta: dict, text: str, mapping: list[dict], blocks: list[tuple[int, int]]) -> None:
    sources = _page_sources(meta, mapping)
    partial = {item["page"] for item in meta.get("partially_checked_pages", [])}
    counts = dict.fromkeys(sources, 0)
    for block, (start, end) in zip(mapping, blocks, strict=True):
        counts[block["page"]] += sum(not char.isspace() for char in text[start:end])
    observed = {row["page"]: (row["extraction"], row["unchecked_image_areas"], sum(row["counts"].values()))
                for row in pdf["mapped_pages"]}
    expected = {page: (mode, page in partial, counts[page]) for page, mode in sources.items()}
    if (pdf["page_count"] != meta["pages"] or observed != expected
            or pdf["unmapped_page_ranges"] != page_gaps(meta["pages"], list(sources))):
        raise ValueError("PDF page coverage differs from the original extraction")
