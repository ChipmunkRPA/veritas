"""Map public archive changes to native Word paragraphs and text runs."""

from __future__ import annotations

from bisect import bisect_right

from docx.oxml.ns import qn

from app.detectors.text.extract import _join_mapped
from app.evidence import document_archive, document_gao_archive
from app.evidence.document_revision import MAX_CHARS, docx_body_paragraphs

WORD_SCOPE = ("For Word files, the comparison reads top-level body paragraphs. Tables, headers, footers, text boxes, "
              "embedded content, formatting and tracked revisions are outside this comparison.")
LEGACY_GAO_WORD_SCOPE = " A wording difference must fit inside one Word paragraph."
GAO_WORD_SCOPE = " Each changed passage identifies its original Word paragraph positions."
GAO_WORD_BOUNDARY_SCOPE = " Verified omissions at adjacent Word paragraph breaks identify both boundary points."
_RUN_TEXT = {qn("w:rPr"), qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")}
_REVISION = {qn("w:pPrChange"), qn("w:rPrChange"), qn("w:ins"), qn("w:del"), qn("w:moveFrom"), qn("w:moveTo")}
_UNMAPPED_RUN = _REVISION | {qn("w:vanish"), qn("w:webHidden")}


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


def ordinary_body_paragraph(paragraph):
    # Fields, tracked edits and other omitted structures can contain text at the gap.
    style = paragraph._p.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
    if style is not None and style.get(qn("w:val"), "").casefold() in {"quote", "intensequote", "blocktext"}:
        return False
    return all(
        (child.tag == qn("w:pPr") and not any(item.tag in _REVISION for item in child.iter())) or (
            child.tag == qn("w:r")
            and all(item.tag in _RUN_TEXT for item in child)
            and not any(item.tag in _UNMAPPED_RUN for item in child.iter())
        )
        for child in paragraph._p
    )


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
                              "runs": runs if complete else [], "runs_complete": complete,
                              "ordinary_boundary": complete and ordinary_body_paragraph(paragraph)}))
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


def _boundary_deletion(doc, change):
    index = change["submitted_paragraph"] - 1
    if index == 0:
        return None
    left, right = doc.source_map[index - 1:index + 1]
    point = change["submitted"]["start"]
    if (point != right["start"] or right["body_index"] != left["body_index"] + 1
            or not (left["ordinary_boundary"] and right["ordinary_boundary"])
            or doc.text[left["end"]:point] != "\n\n"):
        return None
    omitted = document_gao_archive.token_stream(change["archive"]["text"])
    previous = document_gao_archive.token_stream(doc.text[left["start"]:left["end"]])
    following = document_gao_archive.token_stream(doc.text[point:right["end"]])
    # If a repeated token can move the omission across the break, withhold the two positions.
    if not omitted or not previous or not following or omitted[-1][0] == previous[-1][0] or omitted[0][0] == following[0][0]:
        return None
    return {"docx_locations": [
        _location(block, {"start": edge, "end": edge, "changed_tokens": []})
        for block, edge in ((left, left["end"]), (right, point))
    ]}


def _change_locations(doc, change, *, allow_multi_paragraph, allow_boundary_deletion):
    excerpt = change["submitted"]
    start, end = excerpt["start"], excerpt["end"]
    block = doc.source_map[change["submitted_paragraph"] - 1]
    if block["start"] < start == end < block["end"] or block["start"] <= start < end <= block["end"]:
        return {"docx_location": _location(block, excerpt)}
    if not allow_multi_paragraph:
        return None
    if start == end:
        return _boundary_deletion(doc, change) if allow_boundary_deletion else None

    parts = [(item, max(start, item["start"]), min(end, item["end"]))
             for item in doc.source_map if start < item["end"] and item["start"] < end]
    if len(parts) < 2 or parts[0][1] != start or parts[-1][2] != end:
        return None
    # The only unmapped characters between these blocks are extraction separators.
    if any(doc.text[left[2]:right[1]] != "\n\n" for left, right in zip(parts, parts[1:])):
        return None
    locations = [
        _location(item, {"start": lo, "end": hi, "changed_tokens": [
            (max(lo, a), min(hi, b)) for a, b in excerpt["changed_tokens"] if a < hi and lo < b
        ]}) for item, lo, hi in parts
    ]
    return {
        "docx_locations": locations,
        "changed_tokens": [token["extracted_range"] for item in locations for token in item["changed_tokens"]],
    }


def compare_gao(data: bytes, upload_sha256: str, pack_sha256: str, reports, index, *,
                allow_multi_paragraph=True, allow_boundary_deletion=True) -> dict:
    doc = _body(data)
    result = document_gao_archive.compare(doc, upload_sha256, pack_sha256, reports, index)
    result["scope"] += " " + WORD_SCOPE + (GAO_WORD_SCOPE if allow_multi_paragraph else LEGACY_GAO_WORD_SCOPE)
    if allow_multi_paragraph and allow_boundary_deletion:
        result["scope"] += GAO_WORD_BOUNDARY_SCOPE
    if result["status"] != "reference_bound":
        return result

    for change in result["changes"]:
        locations = _change_locations(doc, change, allow_multi_paragraph=allow_multi_paragraph,
                                      allow_boundary_deletion=allow_boundary_deletion)
        if locations is None:
            result.update(status="out_of_scope", reference=None, changes=[], submitted_sentence_ranges=[])
            result["coverage"]["matched_words"] = 0
            return result
        change["submitted"].update(locations)

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
