"""Map public archive changes to native Word paragraphs and text runs."""

from __future__ import annotations

from bisect import bisect_right

from app.detectors.text.extract import _join_mapped
from app.evidence import document_archive, document_gao_archive
from app.evidence.document_revision import MAX_CHARS, docx_body_paragraphs

WORD_SCOPE = ("For Word files, the comparison reads top-level body paragraphs. Tables, headers, footers, text boxes, "
              "embedded content, formatting and tracked revisions are outside this comparison.")
GAO_WORD_SCOPE = " A wording difference must fit inside one Word paragraph."


def _native_interval(ranges, start, end):
    def boundary(point):
        index = bisect_right(ranges, point, key=lambda row: row[0]) - 1
        a, b, lo, hi = ranges[index]
        if point == b:
            return hi
        if lo <= hi and a <= point < b and b - a == hi - lo:
            return lo + point - a
        if point == a:
            return lo
        raise ValueError("The Word position could not be mapped to its paragraph")

    return [boundary(start), boundary(end)]


def _location(block, excerpt):
    ranges = block["source_ranges"]
    tokens = []
    for a, b in excerpt["changed_tokens"]:
        start, end = _native_interval(ranges, a, b)
        fragments = [{"run": number, "range": [max(lo, start) - lo, min(hi, end) - lo]}
                     for number, lo, hi in block["runs"] if lo < end and start < hi and lo < hi]
        tokens.append({"extracted_range": [a, b], "paragraph_range": [start, end], "runs": fragments})
    return {"paragraph": block["paragraph"], "body_index": block["body_index"],
            "paragraph_range": _native_interval(ranges, excerpt["start"], excerpt["end"]),
            "changed_tokens": tokens, "runs_complete": block["runs_complete"]}


def _body(data: bytes):
    parts, characters = [], 0
    for number, body_index, paragraph in docx_body_paragraphs(data):
        value = paragraph.text
        characters += len(value)
        if characters > MAX_CHARS:
            raise ValueError(f"Word body text must contain at most {MAX_CHARS:,} decoded characters.")
        if not value.strip():
            continue
        cursor, runs, contents = 0, [], []
        for run_number, run in enumerate(paragraph.runs, 1):
            text = run.text
            runs.append((run_number, cursor, cursor + len(text)))
            contents.append(text)
            cursor += len(text)
        complete = "".join(contents) == value
        parts.append((value, {"kind": "docx", "paragraph": number, "body_index": body_index,
                              "runs": runs if complete else [], "runs_complete": complete}))
    return _join_mapped(parts)


def compare(data: bytes, upload_sha256: str) -> dict:
    doc = _body(data)
    result = document_archive.compare(doc, upload_sha256)
    result["scope"] += " " + WORD_SCOPE
    for row in result["aligned_paragraphs"]:
        row["docx_paragraph"] = doc.source_map[row["submitted_paragraph"] - 1]["paragraph"]
    for change in result["changes"]:
        source = doc.source_map[change["submitted_paragraph"] - 1]
        change["submitted"]["docx_location"] = _location(source, change["submitted"])
    return result


def compare_gao(data: bytes, upload_sha256: str, pack_sha256: str, reports, index) -> dict:
    doc = _body(data)
    result = document_gao_archive.compare(doc, upload_sha256, pack_sha256, reports, index)
    result["scope"] += " " + WORD_SCOPE + GAO_WORD_SCOPE
    if result["status"] != "reference_bound":
        return result

    for change in result["changes"]:
        block = doc.source_map[change["submitted_paragraph"] - 1]
        excerpt = change["submitted"]
        start, end = excerpt["start"], excerpt["end"]
        if not block["start"] <= start <= end <= block["end"] or (
            start == end and start in {block["start"], block["end"]}
        ):
            result.update(status="out_of_scope", reference=None, changes=[], submitted_sentence_ranges=[])
            result["coverage"]["matched_words"] = 0
            return result
        excerpt["docx_location"] = _location(block, excerpt)

    match_start, match_end = result["reference"]["submitted_range"]
    for number, block in enumerate(doc.source_map, 1):
        start, end = max(match_start, block["start"]), min(match_end, block["end"])
        if start < end:
            result["aligned_paragraphs"].append({
                "submitted_paragraph": number, "submitted_range": [start, end],
                "archive_paragraph": None, "alignment": "bounded_pair",
                "docx_location": _location(block, {"start": start, "end": end, "changed_tokens": []}),
            })
    return result
