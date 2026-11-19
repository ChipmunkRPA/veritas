"""Locate inspected writing in its original decoded text or Word paragraph."""

from __future__ import annotations

from app.evidence.document_scope import (
    EMAIL_HTML_COORDINATES,
    HTML_ARTICLE_KEY,
    HTML_CELL_BOUNDARY,
    HTML_CELL_BOUNDARY_KEY,
    HTML_LEGACY_ENCODING,
    HTML_LEGACY_ENCODING_KEY,
    HTML_STATIC_KEY,
    HTML_STATIC_POLICIES,
    HTML_STATIC_POLICY,
    valid_html_article_scope,
    validate_pdf_blank_pages,
)

KEY = "original_text_locations"
SCHEMA = "document-original-text-locations/1"
MAX_RANGES = 16
PARAGRAPH_MAP_KEY = "decoded_paragraph_locations"
PARAGRAPH_MAP = "decoded-body-paragraphs/1"
# The RTF reader returns the same decoded line breaks for soft breaks and paragraph breaks.
# The saved coordinate describes a decoded text block.
CELL_METHODS = {"xlsx": "spreadsheet_cell", "ods": "spreadsheet_cell", "csv": "delimited_cell", "tsv": "delimited_cell"}
PARAGRAPH_COORDINATES = {"odt": "odt_paragraph", "rtf": "rtf_paragraph", "markdown": "markdown_paragraph",
                         "pptx": "pptx_block", "odp": "odp_block", "epub": "epub_block",
                         **{method: f"{method}_cell" for method in CELL_METHODS}}
PACKAGE_METHODS = {"pptx": "slide", "odp": "slide", "epub": "epub_entry"}
PACKAGE_LABELS = {"slide": "Slide", "epub_entry": "EPUB reading entry"}
PPTX_VISIBILITY_KEY = "pptx_visibility"
PPTX_VISIBILITY = "visible-slide-groups/1"
EPUB_VISIBILITY_KEY = "epub_chapter_visibility"
EPUB_VISIBILITY = "static-epub-chapters/1"
EPUB_VISIBILITY_SCOPE = ("Text marked hidden or given a simple inline display:none style is excluded. "
                         "Stylesheets and reading-system behavior are outside this check.")
ODT_VISIBILITY_KEY = "odt_section_visibility"
ODT_VISIBILITY = "visible-odt-sections/1"
ODT_VISIBILITY_SCOPE = "Hidden and conditional sections are excluded. The reader does not evaluate display conditions."
ODT_TEXT_REFERENCES_KEY = "odt_text_reference_reader"
ODT_TEXT_REFERENCES = "matching-odt-text-references/1"
ODT_TEXT_REFERENCES_SCOPE = ("Supported body references retain saved wording that agrees with their marked text. "
                             "Stale, calculated and unsupported reference fields are outside this check.")
RTF_VISIBILITY_KEY = "rtf_run_visibility"
RTF_VISIBILITY = "visible-rtf-direct-runs/1"
RTF_VISIBILITY_SCOPE = ("The reader excludes directly hidden Rich Text runs and uses saved field display text. "
                        "Visibility from styles and viewer settings is outside this check.")
MARKDOWN_FENCES_KEY = "markdown_fenced_code"
MARKDOWN_FENCES = "exclude-commonmark-fenced-code/1"
XLSX_VISIBILITY_KEY = "xlsx_cell_visibility"
XLSX_VISIBILITY = "visible-xlsx-direct-cell-formats/1"
CSV_SEPARATOR_KEY = "csv_declared_separator"
CSV_DECLARED_SEPARATORS = (",", ";", "|", "\t")
CSV_SEPARATOR_READER_KEY = "csv_separator_reader"
CSV_SEPARATOR_READER = "first-line-declaration/1"
PDF_LINE_WRAP_KEY = "pdf_line_wrap"
PDF_LINE_WRAP = "saved-wrap-hyphens/1"
XLSX_VISIBILITY_SCOPE = ("Cells with Excel's directly applied hide-value format are excluded. Other display formatting, "
                         "including conditional and inherited styles, may differ from the saved text checked here.")
SAVED_FIELDS_KEY = "word_saved_fields"
SAVED_FIELDS_SCHEMA = "document-word-saved-fields/1"
SAVED_FIELDS_READER = "word_saved_fields_reader"
WORD_TABLE_GRID_KEY = "word_table_grid"
WORD_TABLE_GRID_V1 = "word-table-grid/1"
WORD_TABLE_GRID = "word-table-grid/2"
WORD_TABLE_GRID_VERSIONS = (WORD_TABLE_GRID_V1, WORD_TABLE_GRID)
WORD_RUN_MAP_KEY = "word_run_map"
WORD_RUN_MAP = "word-paragraph-runs/1"
WORD_LAYOUT_BREAK_KEY = "word_layout_breaks"
WORD_LAYOUT_BREAK = "word-hard-layout-breaks/1"
WORD_NOTES_KEY = "word_linked_notes"
WORD_NOTES = "word-linked-notes/1"
WORD_NOTE_LABELS = {"footnote": "Word footnote", "endnote": "Word endnote"}
WORD_NOTE_FIELDS = ("note_id", "note_paragraph_index", "reference_body_index", "reference_run_index", "reference_child_index")
WORD_TEXT_BOX_KEY = "word_text_boxes"
WORD_TEXT_BOX = "word-saved-body-text-boxes/1"
WORD_TEXT_BOX_FIELDS = ("reference_body_index", "part_index", "part_paragraph_index")
WORD_TEXT_BOX_SCOPE = ("Includes supported saved text from simple drawing text boxes anchored in the Word body. "
                       "Positions identify the stored drawing and paragraph. Page layout, linked boxes and clipping "
                       "are outside this check.")
MAX_SAVED_FIELDS = 256
MAX_SAVED_FIELD_LOCATIONS = 16
MAX_SAVED_FIELD_CHARS = 256
SAVED_FIELDS_SCOPE = ("Includes saved values from supported simple fields in ordinary Word body paragraphs. "
                      "Field formulas and links are never run. The displayed values can be out of date.")


def clip_source_ranges(ranges, start, end):
    """Clip one mapped interval and preserve any gaps in original HTML."""
    if not 0 <= start < end:
        return []
    cursor = start
    clipped = []
    for a, b, lo, hi in ranges:
        if b <= cursor:
            continue
        if a > cursor or b <= a or not 0 <= lo < hi:
            return []
        stop = min(b, end)
        if b - a == hi - lo:
            lo, hi = lo + cursor - a, lo + stop - a
        elif (cursor, stop) != (a, b):
            return []
        if clipped and clipped[-1][1] == lo:
            clipped[-1][1] = hi
        elif clipped and clipped[-1][1] > lo:
            return []
        else:
            clipped.append([lo, hi])
            if len(clipped) > MAX_RANGES:
                return []
        cursor = stop
        if cursor == end:
            return clipped
    return []


def validate_saved_fields(document):
    record = (document.get("coverage") or {}).get(SAVED_FIELDS_KEY)
    if record is None:
        return
    from app.evidence.document_tracked_changes import PART_NAME

    source = document["source"]
    if (not isinstance(record, dict) or set(record) != {"schema", "part", "found", "included", "locations"}
            or record["schema"] != SAVED_FIELDS_SCHEMA or source["method"] != "python-docx"
            or not isinstance(record["part"], str) or not PART_NAME.fullmatch(record["part"])
            or type(record["found"]) is not int or type(record["included"]) is not int
            or record["found"] <= 0 or not 0 <= record["included"] <= min(record["found"], MAX_SAVED_FIELDS)
            or not isinstance(record["locations"], list)
            or len(record["locations"]) != min(record["included"], MAX_SAVED_FIELD_LOCATIONS)):
        raise ValueError("Invalid saved Word field coverage")
    last_end = 0
    last_native = (-1, -1)
    for row in record["locations"]:
        if set(row) != {"body_index", "field_index", "source_start", "source_end", "start", "end", "text"}:
            raise ValueError("Invalid saved Word field location")
        body, field, native_start, native_end, start, end = (
            row[key] for key in ("body_index", "field_index", "source_start", "source_end", "start", "end"))
        if (any(type(n) is not int for n in (body, field, native_start, native_end, start, end))
                or min(body, field) < 0 or (body, field) <= last_native
                or not last_end <= start < end <= source["chars"]
                or not 0 <= native_start < native_end
                or not isinstance(row["text"], str) or len(row["text"]) != end - start
                or not end - start <= native_end - native_start <= MAX_SAVED_FIELD_CHARS):
            raise ValueError("Saved Word field exceeds its source text or coordinates")
        last_end, last_native = end, (body, field)


def verify_saved_fields(document, text, mapping, meta):
    validate_saved_fields(document)
    record = (document.get("coverage") or {}).get(SAVED_FIELDS_KEY)
    if record != meta.get(SAVED_FIELDS_KEY):
        raise ValueError("Saved Word fields differ from the verified extraction")
    if not record:
        return
    blocks = {block["body_path"][0]: block for block in mapping if block.get("kind") == "docx" and len(block["body_path"]) == 1}
    for row in record["locations"]:
        block = blocks.get(row["body_index"])
        start, end = row["start"], row["end"]
        if (block is None or not block["start"] <= start < end <= block["end"]
                or text[start:end] != row["text"]
                or clip_source_ranges(block["source_ranges"], start, end) != [[row["source_start"], row["source_end"]]]):
            raise ValueError("Saved Word field differs from its retained text range")


def _coordinate(source, coverage, anchor):
    method = source.get("method")
    kind = anchor.get("kind")
    if method == "plain" and kind == "text":
        return "decoded_text"
    if method == "python-docx" and kind == "docx":
        return "word_paragraph"
    if kind == "markup" and anchor.get("part_kind") in WORD_NOTE_LABELS:
        notes = coverage.get(WORD_NOTES_KEY) or {}
        if (method != "python-docx" or notes.get("schema") != WORD_NOTES or not notes.get("included")
                or any(type(anchor.get(field)) is not int or anchor[field] < 0 for field in WORD_NOTE_FIELDS)):
            raise ValueError("Missing Word footnote or endnote source position")
        return "word_note_paragraph"
    if kind == "markup" and anchor.get("part_kind") == "word_text_box":
        boxes = coverage.get(WORD_TEXT_BOX_KEY) or {}
        if (method != "python-docx" or boxes.get("schema") != WORD_TEXT_BOX or not boxes.get("boxes")
                or any(type(anchor.get(field)) is not int or anchor[field] < 0 for field in WORD_TEXT_BOX_FIELDS)):
            raise ValueError("Missing Word text box source position")
        return "word_text_box_paragraph"
    if (kind == "markup" and coverage.get(PARAGRAPH_MAP_KEY) == PARAGRAPH_MAP
            and type(anchor.get("paragraph_index")) is int and anchor["paragraph_index"] >= 0):
        if method in PACKAGE_METHODS and (anchor.get("part_kind") != PACKAGE_METHODS[method]
                or type(anchor.get("part_index")) is not int or anchor["part_index"] < 0):
            raise ValueError("Missing slide or reading-order position")
        if method in CELL_METHODS and (anchor.get("part_kind") != CELL_METHODS[method]
                or any(type(anchor.get(axis)) is not int or anchor[axis] < 1 for axis in ("row", "column"))
                or (CELL_METHODS[method] == "spreadsheet_cell"
                    and (type(anchor.get("part_index")) is not int or anchor["part_index"] < 0))):
            raise ValueError("Missing spreadsheet cell position")
        return PARAGRAPH_COORDINATES.get(method)
    if method == "html" and kind == "markup" and valid_html_article_scope(coverage.get(HTML_ARTICLE_KEY)):
        return "decoded_html"
    if method == "email" and kind == "email":
        email = coverage.get("email") or {}
        if anchor.get("mime_part") != email.get("body_part"):
            return None
        if email.get("body_content_type") == "text/plain":
            return "decoded_email"
        if email.get("body_content_type") == "text/html" and email.get("body_coordinates") == EMAIL_HTML_COORDINATES:
            return "decoded_email_html"
    return None


def package_anchor_label(anchor):
    kind = anchor.get("part_kind")
    if anchor.get("kind") == "markup" and kind == "word_text_box":
        return (f"Word text box at body block {anchor['reference_body_index'] + 1}, "
                f"drawing {anchor['part_index'] + 1}, paragraph {anchor['part_paragraph_index'] + 1}")
    if anchor.get("kind") == "markup" and kind in WORD_NOTE_LABELS:
        return (f"{WORD_NOTE_LABELS[kind]} ID {anchor['note_id']}, paragraph {anchor['note_paragraph_index'] + 1}. "
                f"Reference in Word body block {anchor['reference_body_index'] + 1}")
    if anchor.get("kind") == "markup" and kind in CELL_METHODS.values() and {"row", "column"} <= anchor.keys():
        if kind == "spreadsheet_cell" and "part_index" in anchor:
            return f"Sheet {anchor['part_index'] + 1}, row {anchor['row']}, column {anchor['column']}"
        if kind == "delimited_cell":
            return f"Row {anchor['row']}, column {anchor['column']}"
    label = PACKAGE_LABELS.get(kind)
    if label and anchor.get("kind") == "markup" and "part_index" in anchor:
        return f"{label} {anchor['part_index'] + 1}, text block {anchor['paragraph_index'] + 1}"
    return None


def build(document):
    coverage = document.get("coverage") or {}
    locations = {}
    for packet in document["packets"]:
        rows = []
        for index, anchor in enumerate(packet["anchors"]):
            coordinate = _coordinate(document["source"], coverage, anchor)
            if not coordinate:
                continue
            start, end = anchor["selected_start"], anchor["selected_end"]
            ranges = clip_source_ranges(anchor.get("source_ranges") or [], start, end)
            if ranges:
                rows.append({"anchor_index": index, "coordinate": coordinate, "ranges": ranges})
        if rows:
            locations[packet["id"]] = rows
    return {"schema": SCHEMA, "packets": locations} if locations else None


def validate(document):
    validate_saved_fields(document)
    coverage = document.get("coverage") or {}
    validate_pdf_blank_pages(document, coverage)
    if HTML_STATIC_KEY in coverage:
        method = document.get("source", {}).get("method")
        policy = coverage[HTML_STATIC_KEY]
        html_email = method == "email" and (coverage.get("email") or {}).get("body_content_type") == "text/html"
        if not isinstance(policy, str) or not (method == "html" and policy in HTML_STATIC_POLICIES
                                               or html_email and policy == HTML_STATIC_POLICY):
            raise ValueError("Invalid static HTML reading policy")
    if HTML_LEGACY_ENCODING_KEY in coverage and (coverage[HTML_LEGACY_ENCODING_KEY] != HTML_LEGACY_ENCODING
            or document.get("source", {}).get("method") != "html"
            or coverage.get(HTML_STATIC_KEY) != HTML_STATIC_POLICY):
        raise ValueError("Invalid legacy HTML encoding policy")
    if HTML_CELL_BOUNDARY_KEY in coverage and (coverage[HTML_CELL_BOUNDARY_KEY] != HTML_CELL_BOUNDARY
            or HTML_STATIC_KEY not in coverage):
        raise ValueError("Invalid HTML table cell boundary policy")
    if WORD_TABLE_GRID_KEY in coverage and (coverage[WORD_TABLE_GRID_KEY] not in WORD_TABLE_GRID_VERSIONS
            or document.get("source", {}).get("method") != "python-docx"):
        raise ValueError("Invalid Word table grid mapping")
    if WORD_RUN_MAP_KEY in coverage and (coverage[WORD_RUN_MAP_KEY] != WORD_RUN_MAP
            or document.get("source", {}).get("method") != "python-docx"):
        raise ValueError("Invalid Word paragraph run mapping")
    if WORD_LAYOUT_BREAK_KEY in coverage and (coverage[WORD_LAYOUT_BREAK_KEY] != WORD_LAYOUT_BREAK
            or document.get("source", {}).get("method") != "python-docx"):
        raise ValueError("Invalid Word page or column break mapping")
    if WORD_NOTES_KEY in coverage:
        notes = coverage[WORD_NOTES_KEY]
        if (not isinstance(notes, dict) or set(notes) != {"schema", "references", "included", "paragraphs"}
                or notes["schema"] != WORD_NOTES or document.get("source", {}).get("method") != "python-docx"
                or any(type(notes[key]) is not int or not 0 <= notes[key] <= 4096
                       for key in ("references", "included", "paragraphs"))
                or notes["included"] > min(notes["references"], notes["paragraphs"])
                or bool(notes["included"]) != bool(notes["paragraphs"])):
            raise ValueError("Invalid linked Word note coverage")
    if WORD_TEXT_BOX_KEY in coverage:
        boxes = coverage[WORD_TEXT_BOX_KEY]
        if (not isinstance(boxes, dict) or set(boxes) != {"schema", "boxes", "paragraphs"}
                or boxes["schema"] != WORD_TEXT_BOX or document.get("source", {}).get("method") != "python-docx"
                or any(type(boxes[key]) is not int or not 0 <= boxes[key] <= 4096 for key in ("boxes", "paragraphs"))
                or boxes["boxes"] > boxes["paragraphs"] or bool(boxes["boxes"]) != bool(boxes["paragraphs"])):
            raise ValueError("Invalid Word text box coverage")
    if PARAGRAPH_MAP_KEY in coverage and (coverage[PARAGRAPH_MAP_KEY] != PARAGRAPH_MAP
            or document.get("source", {}).get("method") not in PARAGRAPH_COORDINATES):
        raise ValueError("Invalid decoded body paragraph mapping")
    if PPTX_VISIBILITY_KEY in coverage and (coverage[PPTX_VISIBILITY_KEY] != PPTX_VISIBILITY
            or document.get("source", {}).get("method") != "pptx"):
        raise ValueError("Invalid PowerPoint visibility policy")
    if EPUB_VISIBILITY_KEY in coverage and (coverage[EPUB_VISIBILITY_KEY] != EPUB_VISIBILITY
            or document.get("source", {}).get("method") != "epub"):
        raise ValueError("Invalid EPUB visibility policy")
    if ODT_VISIBILITY_KEY in coverage and (coverage[ODT_VISIBILITY_KEY] != ODT_VISIBILITY
            or document.get("source", {}).get("method") != "odt"):
        raise ValueError("Invalid OpenDocument section visibility policy")
    if ODT_TEXT_REFERENCES_KEY in coverage and (coverage[ODT_TEXT_REFERENCES_KEY] != ODT_TEXT_REFERENCES
            or document.get("source", {}).get("method") != "odt"):
        raise ValueError("Invalid OpenDocument text reference policy")
    if RTF_VISIBILITY_KEY in coverage and (coverage[RTF_VISIBILITY_KEY] != RTF_VISIBILITY
            or document.get("source", {}).get("method") != "rtf"):
        raise ValueError("Invalid Rich Text run visibility policy")
    if MARKDOWN_FENCES_KEY in coverage and (coverage[MARKDOWN_FENCES_KEY] != MARKDOWN_FENCES
            or document.get("source", {}).get("method") != "markdown"):
        raise ValueError("Invalid Markdown fenced code policy")
    if XLSX_VISIBILITY_KEY in coverage and (coverage[XLSX_VISIBILITY_KEY] != XLSX_VISIBILITY
            or document.get("source", {}).get("method") != "xlsx"):
        raise ValueError("Invalid Excel cell visibility policy")
    if CSV_SEPARATOR_KEY in coverage and (coverage[CSV_SEPARATOR_KEY] not in CSV_DECLARED_SEPARATORS
            or document.get("source", {}).get("method") != "csv"):
        raise ValueError("Invalid declared CSV separator")
    if PDF_LINE_WRAP_KEY in coverage and (coverage[PDF_LINE_WRAP_KEY] != PDF_LINE_WRAP
            or document.get("source", {}).get("method") != "pymupdf"):
        raise ValueError("Invalid PDF line-wrap policy")
    article = coverage.get(HTML_ARTICLE_KEY)
    if article is not None and (document.get("source", {}).get("method") != "html" or not valid_html_article_scope(article)):
        raise ValueError("Invalid saved HTML article boundaries")
    record = coverage.get(KEY)
    if (record is not None or article is not None
            or any(key in coverage for key in (PARAGRAPH_MAP_KEY, WORD_NOTES_KEY, WORD_TEXT_BOX_KEY))) and record != build(document):
        raise ValueError("Original text locations differ from their mapped evidence spans")
    if article is not None and record is not None:
        start, end = article["scope"]
        if any(not start <= a < b <= end for rows in record["packets"].values() for row in rows for a, b in row["ranges"]):
            raise ValueError("An exact HTML location extends outside the saved article")
