"""Exact locations of extracted Word runs with direct hidden formatting."""

from app.core.canonical import canonical_hash
from app.evidence.document_tracked_changes import PART_NAME

KEY = "word_text_visibility"
COMMITMENT = "word_text_visibility_sha256"
SCHEMA = "document-word-text-visibility/1"
HEADLINE = "Word writing assessment limited by hidden text"
MAX_LOCATIONS = 24
SCOPE = "Checks extracted Word body and table text for direct hidden formatting."
LIMITATION = "Word can show hidden text when its display settings allow it. This check carries zero AI-authorship weight."


def inspect_paragraph(paragraph, body_path, body_paragraph):
    from docx.oxml.ns import qn
    from docx.text.run import Run

    offset = 0
    for index, element in enumerate(paragraph._p.xpath("./w:r | ./w:hyperlink/w:r")):
        text = Run(element, paragraph).text
        end = offset + len(text)
        mark = element.xpath("./w:rPr/w:vanish")
        if (text.strip() and len(mark) == 1
                and mark[0].get(qn("w:val"), "true") in {"true", "1", "on"}):
            yield {"body_path": body_path, "body_paragraph": body_paragraph,
                   "run": index, "source_start": offset, "source_end": end}
        offset = end


def build(part, paragraphs, hidden_runs, locations):
    return {"schema": SCHEMA, "part": part, "checked_paragraphs": paragraphs, "hidden_runs": hidden_runs,
            "locations": locations, "authorship_confidence_contribution": 0}


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def _native_path(path):
    if not isinstance(path, list) or not path or (len(path) - 1) % 5:
        raise ValueError("Invalid Word body path")
    order = [path[0]]
    for i in range(1, len(path), 5):
        row_label, row, cell_label, cell, element = path[i:i + 5]
        if (row_label, cell_label) != ("row", "cell"):
            raise ValueError("Invalid Word table path")
        order.extend((row, cell, element))
    if any(type(value) is not int or value < 0 for value in order):
        raise ValueError("Invalid Word body position")
    return tuple(order)


def validate(document, expected):
    record = document.get("coverage", {}).get(KEY)
    if record is None and expected is None:
        return
    if (not isinstance(record, dict) or document["source"]["method"] != "python-docx"
            or set(record) != {"schema", "part", "checked_paragraphs", "hidden_runs", "locations", "authorship_confidence_contribution"}
            or record["schema"] != SCHEMA or not PART_NAME.fullmatch(record["part"])
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or any(type(record[k]) is not int or record[k] < 0 for k in ("checked_paragraphs", "hidden_runs"))
            or not isinstance(record["locations"], list) or len(record["locations"]) != min(MAX_LOCATIONS, record["hidden_runs"])
            or canonical_hash(record) != expected):
        raise ValueError("Word visibility evidence differs from its source commitment")
    previous_path, previous_run, previous_end = (), -1, 0
    paths = set()
    for row in record["locations"]:
        if set(row) != {"body_path", "body_paragraph", "run", "source_start", "source_end"}:
            raise ValueError("Invalid hidden Word text location")
        path = _native_path(row["body_path"])
        if not ((len(path) > 1 and row["body_paragraph"] is None) or
                (len(path) == 1 and type(row["body_paragraph"]) is int and 1 <= row["body_paragraph"] <= path[0] + 1)):
            raise ValueError("Invalid hidden Word paragraph number")
        run, start, end = (row[k] for k in ("run", "source_start", "source_end"))
        if (any(type(value) is not int for value in (run, start, end)) or run < 0 or not 0 <= start < end
                or path < previous_path or (path == previous_path and (run <= previous_run or start < previous_end))):
            raise ValueError("Invalid hidden Word text coordinates")
        paths.add(path)
        previous_path, previous_run, previous_end = path, run, end
    if len(paths) > record["checked_paragraphs"]:
        raise ValueError("Invalid Word visibility coverage")


def verify_source(document, original, *, frozen=False):
    expected = document["source"].get(COMMITMENT)
    validate(document, expected)
    record = document.get("coverage", {}).get(KEY)
    if (frozen or record is not None) and record != original:
        raise ValueError("Word visibility evidence differs from the source extraction")
