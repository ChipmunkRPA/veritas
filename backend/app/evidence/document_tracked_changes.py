"""Report the text edits explicitly marked in the main Word document part."""

from __future__ import annotations

import hashlib
import re

from app.core.canonical import canonical_hash

KEY = "word_tracked_changes"
COMMITMENT = "word_tracked_changes_sha256"
JOINED_PARAGRAPHS = "word_joined_paragraphs"
SCHEMA = "document-word-tracked-changes/1"
PROJECTION_SCHEMA = "document-word-tracked-changes/2"
PARAGRAPH_SCHEMA = "document-word-tracked-changes/3"
INPUT_UNCHANGED = "no_tracked_text_changes"
INPUT_PROJECTED = "accepted_inline_changes"
INPUT_PARAGRAPHS = "accepted_paragraph_changes"
INPUT_UNSUPPORTED = "unsupported_tracked_changes"
UNSUPPORTED_HEADLINE = "Writing not assessed: unsupported Word revisions"
INPUT_NOTICES = {
    INPUT_PROJECTED: "Writing checks use the version with these inline insertions included and deletions excluded.",
    INPUT_PARAGRAPHS: "Writing checks use the accepted wording and paragraph breaks.",
    INPUT_UNSUPPORTED: "Some Word revisions could change which text is visible. Automated writing checks were skipped.",
}
MAX_ROWS = 24
MAX_EXCERPT = 240
SCOPE = "Checks inline Track Changes insertions and deletions in top-level Word body paragraphs. Tables, headers, footers, paragraph marks, moves and formatting are outside the measured scope."
PARAGRAPH_SCOPE = ("Checks simple tracked wording and paragraph breaks in the Word body. Tables, headers, footers, moves "
                   "and complex paragraph structures are outside the measured scope.")
LIMITATION = "These are markings stored in the submitted file. They do not establish who wrote the text, when it changed, or whether AI was involved."
SHA256 = re.compile(r"[0-9a-f]{64}")
PART_NAME = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.xml")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _revision_text(element, kind, tag):
    content = []
    for run in element:
        if run.tag != tag["r"]:
            return None
        for item in run:
            if item.tag == tag["rPr"]:
                continue
            if len(item):
                return None
            if item.tag == tag["t" if kind == "inserted" else "delText"]:
                content.append(item.text or "")
            elif item.tag == tag["tab"]:
                content.append("\t")
            elif item.tag == tag["br"] and item.get(tag["type"], "textWrapping") == "textWrapping":
                content.append("\n")
            else:
                return None
    return "".join(content)


def _plain_paragraph(paragraph, tag):
    for element in paragraph:
        if not isinstance(element.tag, str):
            continue
        if element.tag == tag["pPr"]:
            if any(isinstance(child.tag, str) and child.tag != tag["rPr"] for child in element):
                return False
        elif element.tag not in (tag["r"], tag["ins"], tag["del"]):
            return False
    return True


def _break_mark(parent, following, tag, kinds):
    properties = parent.findall(tag["pPr"])
    if len(properties) != 1:
        return None
    runs = properties[0].findall(tag["rPr"])
    if len(runs) != 1:
        return None
    changes = [item for item in runs[0] if item.tag in kinds]
    if len(changes) != 1 or len(changes[0]) or not _plain_paragraph(parent, tag):
        return None
    mark = changes[0]
    if (mark.tag == tag["del"] and
            (following is None or following.tag != tag["p"] or not _plain_paragraph(following, tag))):
        return None
    # Indices count XML elements, as in the inline rows.
    return mark, [_child_index(node) for node in (properties[0], runs[0], mark)]


def _child_index(node):
    return sum(isinstance(sibling.tag, str) for sibling in node.itersiblings(preceding=True))


def inspect(body, part, *, project_inline=False, joined_paragraphs=None):
    """Record native positions and populate the join plan for the accepted text when requested."""
    from docx.oxml.ns import qn
    inline_names = ("p", "pPr", "r", "rPr", "ins", "del", "t", "delText", "tab", "br", "type")
    tag = {name: qn(f"w:{name}") for name in inline_names}
    kinds = {tag["ins"]: "inserted", tag["del"]: "deleted"}
    marks = set(kinds) | {qn(f"w:{name}") for name in (
        "moveFrom", "moveTo", "moveFromRangeStart", "moveToRangeStart", "cellIns", "cellDel", "cellMerge")}
    if project_inline:
        marks.update(qn(f"w:{name}") for name in ("moveFromRangeEnd", "moveToRangeEnd", "customXmlInsRangeStart",
                     "customXmlInsRangeEnd", "customXmlDelRangeStart", "customXmlDelRangeEnd", "customXmlMoveFromRangeStart",
                     "customXmlMoveFromRangeEnd", "customXmlMoveToRangeStart", "customXmlMoveToRangeEnd"))
    visited = set()
    rows, totals = [], dict.fromkeys(kinds.values(), 0)
    breaks, joins, break_count = [], set(), 0
    unsupported, paragraph = 0, 0
    elements = [item for item in body if isinstance(item.tag, str)]
    for body_index, parent in enumerate(elements):
        if parent.tag != tag["p"]:
            continue
        paragraph += 1
        if joined_paragraphs is not None:
            following = elements[body_index + 1] if body_index + 1 < len(elements) else None
            if match := _break_mark(parent, following, tag, kinds):
                mark, xml_path = match
                visited.add(mark)
                break_count += 1
                if mark.tag == tag["del"]:
                    joins.add(parent)
                if len(breaks) < MAX_ROWS:
                    breaks.append({"kind": kinds[mark.tag], "paragraph": paragraph,
                                   "body_index": body_index, "child_path": xml_path})
        for inline_index, item in enumerate(child for child in parent if isinstance(child.tag, str)):
            if (kind := kinds.get(item.tag)) is None:
                continue
            visited.add(item)
            text = _revision_text(item, kind, tag)
            if not text or not text.strip():
                unsupported += 1
                continue
            totals[kind] += 1
            if len(rows) < MAX_ROWS:
                rows.append({"kind": kind, "paragraph": paragraph, "body_index": body_index, "inline_index": inline_index,
                             "text": text[:MAX_EXCERPT], "characters": len(text), "text_sha256": _sha(text)})
    unsupported += sum(item not in visited for item in body.iter() if item.tag in marks)
    measured = sum(totals.values())
    report = {"schema": SCHEMA, "part": part,
              "status": "partial" if unsupported else "found" if measured or break_count else "no_match",
            "scope": SCOPE, "limitation": LIMITATION,
            "paragraphs": paragraph, **totals, "unsupported_marks": unsupported, "omitted_excerpts": measured - len(rows),
            "rows": rows, "authorship_confidence_contribution": 0}
    if project_inline:
        view = (INPUT_UNSUPPORTED if unsupported else INPUT_PARAGRAPHS if break_count else
                INPUT_PROJECTED if measured else INPUT_UNCHANGED)
        report.update(schema=PROJECTION_SCHEMA, writing_input=view)
        if joined_paragraphs is not None:
            report.update(schema=PARAGRAPH_SCHEMA, scope=PARAGRAPH_SCOPE, paragraph_marks=breaks,
                          omitted_paragraph_marks=break_count - len(breaks))
        if view in (INPUT_PROJECTED, INPUT_PARAGRAPHS):
            _accepted_inline(body, rows, tag)
        if view == INPUT_PARAGRAPHS:
            joined_paragraphs.update(joins)
    return report


def _accepted_inline(body, rows, tag):
    """Project the supported run changes in memory and retain their original XML positions."""
    tracked = {(row["body_index"], row["inline_index"]): row for row in rows if row["kind"] == "inserted"}
    for body_index, paragraph in enumerate(node for node in body if isinstance(node.tag, str)):
        if paragraph.tag != tag["p"]:
            continue
        positioned = {}
        for inline_index, change in enumerate([node for node in paragraph if isinstance(node.tag, str)]):
            if change.tag == tag["ins"]:
                for run in list(change):
                    change.addprevious(run)
                    if row := tracked.get((body_index, inline_index)):
                        positioned[run] = row
            if change.tag in (tag["ins"], tag["del"]):
                paragraph.remove(change)
        if positioned:
            offset = 0
            for item in paragraph.inner_content_elements:
                end = offset + len(item.text)
                if row := positioned.get(item):
                    row.setdefault("writing_input_range", [offset, end])[1] = end
                offset = end


def unsupported(review):
    return bool(review and review.get("writing_input") == INPUT_UNSUPPORTED)


def projected(review):
    return bool(review and review.get("writing_input") in (INPUT_PROJECTED, INPUT_PARAGRAPHS))


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def validate(document, expected):
    review = (document.get("coverage") or {}).get(KEY)
    if review is None and expected is None:
        return
    if (not isinstance(review, dict) or canonical_hash(review) != expected or document["source"]["method"] != "python-docx"
            or review["schema"] not in (SCHEMA, PROJECTION_SCHEMA, PARAGRAPH_SCHEMA) or not PART_NAME.fullmatch(review["part"])
            or review["scope"] != (PARAGRAPH_SCOPE if review["schema"] == PARAGRAPH_SCHEMA else SCOPE)
            or review["limitation"] != LIMITATION
            or type(review["authorship_confidence_contribution"]) is not int
            or review["authorship_confidence_contribution"] != 0):
        raise ValueError("Tracked Word edits differ from their source commitment")
    fields = ("paragraphs", "inserted", "deleted", "unsupported_marks", "omitted_excerpts")
    if any(type(review[k]) is not int or review[k] < 0 for k in fields):
        raise ValueError("Invalid tracked edit coverage")
    measured = review["inserted"] + review["deleted"]
    rows = review["rows"]
    breaks = _validate_breaks(review)
    status = "partial" if review["unsupported_marks"] else "found" if measured or breaks else "no_match"
    if (review["status"] != status or len(rows) != min(measured, MAX_ROWS)
            or review["omitted_excerpts"] != measured - len(rows)):
        raise ValueError("Invalid tracked edit disposition")
    writing_input = (INPUT_UNSUPPORTED if status == "partial" else INPUT_PARAGRAPHS if breaks else
                     INPUT_PROJECTED if status == "found" else INPUT_UNCHANGED)
    has_projection = review["schema"] != SCHEMA
    if ((has_projection and review.get("writing_input") != writing_input)
            or (not has_projection and "writing_input" in review)):
        raise ValueError("Tracked Word edits have an invalid writing input")
    previous = (-1, -1)
    paragraph_stop = 0
    counts = {"inserted": 0, "deleted": 0}
    for row in rows:
        kind, text, length = row["kind"], row["text"], row["characters"]
        path = row["body_index"], row["inline_index"]
        if (kind not in counts or any(type(i) is not int or i < 0 for i in path) or path <= previous
                or type(row["paragraph"]) is not int or not 1 <= row["paragraph"] <= min(review["paragraphs"], path[0] + 1)
                or type(length) is not int or length < 1 or not isinstance(text, str)
                or len(text) != min(length, MAX_EXCERPT) or not SHA256.fullmatch(row["text_sha256"])
                or (length <= MAX_EXCERPT and _sha(text) != row["text_sha256"])):
            raise ValueError("A tracked Word edit has invalid text or native coordinates")
        if path[0] != previous[0]:
            paragraph_stop = 0
        if has_projection and projected(review) and kind == "inserted":
            start, end = row["writing_input_range"]
            if any(type(value) is not int for value in (start, end)) or start < paragraph_stop or end - start != length:
                raise ValueError("An inserted Word edit has invalid writing-input characters")
            paragraph_stop = end
        elif "writing_input_range" in row:
            raise ValueError("A revision outside the writing input has extracted coordinates")
        counts[kind] += 1
        previous = path
    if any(counts[k] > review[k] for k in counts):
        raise ValueError("Tracked edit counts differ from displayed excerpts")


def _validate_breaks(review):
    if review["schema"] != PARAGRAPH_SCHEMA:
        if "paragraph_marks" in review or "omitted_paragraph_marks" in review:
            raise ValueError("Historical Word evidence cannot contain paragraph mark locations")
        return 0
    marks, omitted = review["paragraph_marks"], review["omitted_paragraph_marks"]
    if (not isinstance(marks, list) or type(omitted) is not int or omitted < 0
            or len(marks) > MAX_ROWS or (omitted and len(marks) < MAX_ROWS)
            or len(marks) + omitted > review["paragraphs"]):
        raise ValueError("Invalid Word paragraph mark coverage")
    last = -1
    for row in marks:
        if (set(row) != {"kind", "paragraph", "body_index", "child_path"}
                or row["kind"] not in ("inserted", "deleted")
                or type(row["body_index"]) is not int or row["body_index"] <= last
                or type(row["paragraph"]) is not int
                or not 1 <= row["paragraph"] <= min(review["paragraphs"], row["body_index"] + 1)
                or not isinstance(row["child_path"], list) or len(row["child_path"]) != 3
                or any(type(index) is not int or index < 0 for index in row["child_path"])):
            raise ValueError("Invalid Word paragraph mark location")
        last = row["body_index"]
    return len(marks) + omitted


def verify_source(document, recorded, *, expected):
    validate(document, expected)
    review = (document.get("coverage") or {}).get(KEY)
    if review is not None and review != recorded:
        raise ValueError("Tracked Word edits differ from the original XML positions")
    return len(review["rows"]) + len(review.get("paragraph_marks", [])) if review is not None else 0
