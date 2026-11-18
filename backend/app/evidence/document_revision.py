"""Compare two reviewer-supplied text versions without a model or persistence."""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from collections import Counter
from difflib import SequenceMatcher
from email import policy
from email.parser import BytesParser
from html import unescape
from io import BytesIO
from itertools import takewhile
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from app.core.deadline import StepTimeout
from app.detectors.text import html_article
from app.detectors.text.extract import _join_mapped, _normalise, extract, text_hash
from app.evidence import document_native_locations, document_revision_pdf, document_revision_values
from app.evidence.document_scope import (
    email_agreement_body_spans,
    mapped_email_html,
    valid_html_article_scope,
)
from app.storage.files import check_docx_safe

SCHEMA = "chipmunk.document-revision-comparison/1"
HTML_SCHEMA = "chipmunk.document-revision-comparison/2"
MIXED_SCHEMA = "chipmunk.document-revision-comparison/3"
RELOCATION_SCHEMA = "chipmunk.document-revision-comparison/6"
EDITED_RELOCATION_SCHEMA = "chipmunk.document-revision-comparison/8"
ARTICLE_SCHEMA = "chipmunk.document-revision-comparison/9"
HTML_COORDINATES = "email_html_extracted_text_unicode_codepoint"
ARTICLE_COORDINATES = "html_article_extracted_text_unicode_codepoint"
MIXED_COORDINATES = "per_source_unicode_codepoint"
TOKENS = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")
MAX_BYTES = 1024 * 1024
MAX_CHARS = 30_000
MAX_TOKENS = 2_000
MAX_DOCX_XML_BYTES = 2 * 1024 * 1024
ROLES = ("body", "quoted", "signature")
FORMATS = {".eml": "email", ".txt": "plain_text", ".docx": "docx", ".odt": "odt", ".pdf": "pdf", ".rtf": "rtf", ".html": "html"}
MIXED_DRAFTS = (".txt", ".docx", ".pdf")
EXTRACTED_PARAGRAPH_DRAFTS = {".odt": "OpenDocument", ".rtf": "Rich Text"}
COMPARABLE_DRAFTS = (*MIXED_DRAFTS, *EXTRACTED_PARAGRAPH_DRAFTS)
TEXT_COORDINATES = "decoded_source_unicode_codepoint"
WORD_COORDINATES = "docx_body_unicode_codepoint"
ODT_COORDINATES = "odt_body_unicode_codepoint"
RTF_COORDINATES = "rtf_extracted_text_unicode_codepoint"
SUPPORTED_DRAFTS = "matching .eml, .txt, .docx, .odt, .rtf, .pdf or HTML files"
SCOPE = "Compares words and punctuation in the decoded text. Email body, quoted text and signatures are mapped separately."
HTML_SCOPE = ("Compares visible words and punctuation in the reply body. Quoted text, attachments, headers and "
              "formatting are outside this comparison.")
ARTICLE_SCOPE = ("Compares words and punctuation in supported paragraphs, headings, lists and descriptions within one "
                 "marked HTML article or main area in each saved file. Navigation, quotations, figures, tables, "
                 "and explicitly hidden content are outside the comparison. Scripts and linked resources stay unloaded.")
DOCX_SCOPE = ("Compares words and punctuation in top-level Word body paragraphs. Tables, headers, footers, text boxes, "
              "embedded content, formatting and tracked revision markup are outside this comparison.")
ODT_SCOPE = ("OpenDocument coverage includes extracted body paragraphs, headings, lists and table text in reading order. "
             "The comparison preserves uniquely matched unchanged sentences when their positions move. "
             "Headers, footers, notes, tracked changes and embedded objects are outside the comparison.")
RTF_SCOPE = ("Rich Text positions refer to extracted text and its paragraphs. The comparison preserves uniquely matched "
             "unchanged sentences when their positions move. Formatting and embedded objects are outside the comparison.")
MIXED_SCOPE = "Compares words and punctuation across file formats. Formatting is outside the comparison."
MIXED_WORD_SCOPE = "Word coverage includes top-level body paragraphs. Word tables, headers, footers and text boxes are excluded."
MIXED_PDF_SCOPE = "PDF coverage includes the selectable text on one upright page with a clear reading order. Scanned text is excluded."
MIXED_MULTIPAGE_PDF_SCOPE = "PDF coverage includes selectable text on all {pages} upright, single-column pages. Scanned text is excluded."
DOCX_PART = "word/document.xml"
INTERPRETATION = "This comparison locates changes between the supplied versions. The revision sequence and authorship require independent evidence."
RELOCATION_SCOPE = "The reordered passage has the same wording and punctuation in both versions. Its position changed relative to other matching paragraphs."
EDITED_RELOCATION_SCOPE = ("One body passage changed position relative to other matching paragraphs. Its unchanged opening and "
                           "closing words link it across versions. Highlights show wording and punctuation changes within that passage.")
WordParagraphs = list[tuple[int, int, int, int]]
WordLayouts = tuple[WordParagraphs, WordParagraphs]
PdfLayouts = tuple[list[dict], list[dict]]
MixedLayouts = tuple[WordParagraphs | list[dict] | None, WordParagraphs | list[dict] | None]


def draft_extension(filename: str | Path) -> str:
    from app.storage.files import HTML_EXTS

    suffix = Path(filename).suffix.lower()
    return ".html" if suffix in HTML_EXTS else suffix


def tokens(text, offsets):
    return [(item.group(), item.start() + start, item.end() + start)
            for start, end in offsets for item in TOKENS.finditer(text[start:end])]


def differences(earlier, later):
    """Build half-open source ranges for paired changes and their exact token extents."""
    edits = []
    for action, a, b, c, d in SequenceMatcher(None, [row[0] for row in earlier], [row[0] for row in later],
                                               autojunk=False).get_opcodes():
        if action == "equal":
            continue
        if edits and a - edits[-1][1] <= 2 and c - edits[-1][3] <= 2:
            old_a, _, old_c, _, changed_before, changed_after = edits[-1]
            edits[-1] = (old_a, b, old_c, d, [*changed_before, *earlier[a:b]], [*changed_after, *later[c:d]])
        else:
            edits.append((a, b, c, d, earlier[a:b], later[c:d]))

    def bounds(rows, start, stop):
        if start < stop:
            return [rows[start][1], rows[stop - 1][2]]
        point = rows[start][1] if start < len(rows) else (rows[-1][2] if rows else None)
        return [point, point] if point is not None else None

    return [{"earlier": bounds(earlier, a, b), "later": bounds(later, c, d),
             "earlier_changed_tokens": [[u, v] for _, u, v in old], "later_changed_tokens": [[u, v] for _, u, v in new]}
            for a, b, c, d, old, new in edits]


def _sentence_anchors(text, layout, extension, body):
    from app.detectors.text.segment import spans

    if extension == ".docx" or extension in EXTRACTED_PARAGRAPH_DRAFTS:
        blocks = [(start, end) for _, _, start, end in layout]
    elif extension == ".html":
        blocks = layout[2]
    elif extension == ".pdf":
        blocks = [(row["start"], row["end"]) for row in layout]
    else:
        blocks = layout
    starts = [row[1] for row in body]
    result = {}
    repeated = set()
    for start, end, _ in spans(text, blocks):
        a, b = (bisect_right(starts, point - 1) for point in (start, end))
        if b - a < 8 or body[b - 1][2] > end or a and body[a - 1][2] > start:
            continue
        key = tuple(word for word, _, _ in body[a:b])
        if key in result:
            repeated.add(key)
        result[key] = a, b
    return {key: value for key, value in result.items() if key not in repeated}


def _localized_body_differences(old_text, new_text, old_layout, new_layout, old_extension, new_extension, old, new):
    """Use shared sentences to constrain changes and preserve their unchanged text when moved."""
    left = _sentence_anchors(old_text, old_layout, old_extension, old)
    right = _sentence_anchors(new_text, new_layout, new_extension, new)
    old_keys = [key for key in left if key in right]
    new_keys = [key for key in right if key in left]
    anchors = [(left[key], right[key]) for a, _, count in SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_matching_blocks()
               for key in old_keys[a:a + count]]
    old_unchanged, new_unchanged = (
        {i for key in old_keys for i in range(*positions[key])} for positions in (left, right)
    )
    ranges = []
    old_cursor = new_cursor = 0
    for (a, b), (c, d) in [*anchors, ((len(old), len(old)), (len(new), len(new)))]:
        # Keep matching edges even when an edit altered how the sentence split.
        before = [old[index] for index in range(old_cursor, a) if index not in old_unchanged]
        after = [new[index] for index in range(new_cursor, c) if index not in new_unchanged]
        prefix = sum(1 for _ in takewhile(_equal_words, zip(before, after)))
        suffix = sum(1 for _ in takewhile(_equal_words, zip(reversed(before[prefix:]), reversed(after[prefix:]))))
        for row in differences(before[prefix:len(before) - suffix], after[prefix:len(after) - suffix]):
            for side, values, index, window in (("earlier", old, a, before), ("later", new, c, after)):
                if row[side] is None and values:
                    point = (window[prefix][1] if prefix < len(window) else
                             values[index][1] if index < len(values) else values[-1][2])
                    row[side] = [point, point]
            ranges.append(row)
        old_cursor, new_cursor = b, d
    return ranges


def _equal_words(pair):
    return pair[0][0] == pair[1][0]


def _part_id(message, target, parent="1"):
    if message is target:
        return parent
    for i, child in enumerate(message.iter_parts(), 1):
        if found := _part_id(child, target, f"{parent}.{i}"):
            return found
    return None


def _source_record(data, decoded, mime_part=None):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "decoded_text_sha256": text_hash(decoded), "decoded_characters": len(decoded), "mime_part": mime_part}


def _docx_part(archive, name):
    matches = [info for info in archive.infolist() if info.filename == name]
    if len(matches) != 1 or not 0 < matches[0].file_size <= MAX_DOCX_XML_BYTES:
        raise ValueError("The Word document must contain one readable body within the XML size limit.")
    with archive.open(matches[0]) as stream:
        data = stream.read(MAX_DOCX_XML_BYTES + 1)
    if len(data) != matches[0].file_size:
        raise ValueError("The Word document contains a damaged XML part.")
    return data


def _standard_docx_body(archive):
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn

    declarations = parse_xml(_docx_part(archive, "[Content_Types].xml"))
    namespace = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    if not any(node.get("PartName") == f"/{DOCX_PART}" and node.get("ContentType") == content_type
               for node in declarations.findall(namespace + "Override")):
        raise ValueError("Choose a Word document saved in the standard .docx format.")
    document = parse_xml(_docx_part(archive, DOCX_PART))
    if (document.tag != qn("w:document") or (body := document.find(qn("w:body"))) is None
            or document.getroottree().docinfo.doctype or declarations.getroottree().docinfo.doctype):
        raise ValueError("The Word document body could not be read.")
    return body


def docx_body(data):
    """Read the bounded main part used by the earlier-draft comparison."""
    from lxml.etree import XMLSyntaxError

    from app.detectors.text.strict_word import read_document

    check_docx_safe(BytesIO(data))
    try:
        with ZipFile(BytesIO(data)) as archive:
            try:
                return _standard_docx_body(archive)
            except (ValueError, XMLSyntaxError):
                strict = read_document(archive, include_styles=False, max_xml_bytes=MAX_DOCX_XML_BYTES)
                if strict is None:
                    raise
                return strict.element.body
    except (BadZipFile, RuntimeError, NotImplementedError, XMLSyntaxError) as error:
        raise ValueError("The Word document body could not be read.") from error


def _word_paragraphs(body):
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    paragraph_number = 0
    for body_index, element in enumerate(body):
        if element.tag != qn("w:p"):
            continue
        paragraph_number += 1
        yield paragraph_number, body_index, Paragraph(element, None)


def docx_body_paragraphs(data):
    """Yield numbered top-level paragraphs from a bounded native Word package."""
    yield from _word_paragraphs(docx_body(data))


def _docx_source(data, *, body=None, allow_empty=False):
    paragraphs, anchors, cursor = [], [], 0
    for paragraph_number, body_index, paragraph in _word_paragraphs(docx_body(data) if body is None else body):
        value = paragraph.text
        if not value.strip():
            continue
        start = cursor + (2 if paragraphs else 0)
        cursor = start + len(value)
        if cursor > MAX_CHARS:
            raise ValueError(f"Each version can contain up to {MAX_CHARS:,} decoded body characters.")
        paragraphs.append(value)
        anchors.append((paragraph_number, body_index, start, cursor))
    if not paragraphs and not allow_empty:
        raise ValueError("Each Word version needs readable top-level body paragraphs.")
    decoded = "\n\n".join(paragraphs)
    body_tokens = tokens(decoded, [(start, end) for _, _, start, end in anchors])
    if len(body_tokens) > MAX_TOKENS:
        raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
    return decoded, {"body": body_tokens, "quoted": [], "signature": []}, _source_record(data, decoded), anchors


def _html_source(data, decoded, doc, mime_part):
    email = doc.meta["email"]
    spans = email_agreement_body_spans(doc.source_map, email, len(doc.text)) if mapped_email_html(email) else None
    if spans is None:
        raise ValueError("Use HTML-only replies with 2, 3, 5 or 6 simple body paragraphs followed by a blockquote.")
    body = tokens(doc.text, [(span["start"], span["end"]) for span in spans])
    if len(body) > MAX_TOKENS:
        raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} body words and punctuation tokens.")
    mapping = [row for block in doc.source_map[:len(spans)] for row in block["source_ranges"]]
    record = _source_record(data, doc.text, mime_part) | {
        "decoded_html_sha256": text_hash(decoded), "decoded_html_characters": len(decoded),
    }
    return doc.text, {"body": body, "quoted": [], "signature": []}, record, (decoded, mapping)


def _article_source(data):
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    decoded = data.decode(encoding)
    article = html_article.read(decoded)
    if article is None or not valid_html_article_scope(article[1]):
        raise ValueError("Use saved HTML files with readable paragraphs in one marked article or main area.")
    doc = _join_mapped(article[0])
    if not 0 < len(doc.text) <= MAX_CHARS:
        raise ValueError(f"Each article needs readable text and at most {MAX_CHARS:,} extracted characters.")
    body = tokens(doc.text, doc.paragraphs)
    if len(body) > MAX_TOKENS:
        raise ValueError(f"Each article can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
    mapping = [row for block in doc.source_map for row in block["source_ranges"]]
    record = _source_record(data, doc.text) | {
        "decoded_html_sha256": text_hash(decoded), "decoded_html_characters": len(decoded),
        "html_article_scope": {**article[1], "included_blocks": len(doc.paragraphs), "unmapped_text": article[2]},
    }
    return doc.text, {"body": body, "quoted": [], "signature": []}, record, (decoded, mapping, doc.paragraphs)


def _html_token_sources(text, layout, changed_tokens):
    decoded, mapping = layout[:2]
    located = []
    for start, end in changed_tokens:
        spans = []
        cursor = start
        first = bisect_right(mapping, start, key=lambda row: row[1])
        for index in range(first, len(mapping)):
            a, b, lo, hi = mapping[index]
            if a >= end:
                break
            begin, stop = max(a, start), min(b, end)
            if b - a == hi - lo:
                lo, hi = lo + begin - a, lo + stop - a
            piece = decoded[lo:hi]
            if (cursor != begin or not 0 <= lo < hi <= len(decoded)
                    or (piece != text[begin:stop] and unescape(piece) != text[begin:stop])):
                break
            if spans and spans[-1][1] == lo:
                spans[-1][1] = hi
            else:
                spans.append([lo, hi])
            cursor = stop
        if cursor != end:
            raise ValueError("A changed token could not be mapped to the decoded HTML source.")
        located.append({"start": start, "end": end, "source_ranges": spans})
    return located


def _source(path: Path, data: bytes, extension: str, *, mixed: bool = False, word_body=None, allow_empty_word_body=False,
            skip_hidden_odt_sections: bool = True, skip_hidden_rtf_runs: bool = True):
    if extension == ".html":
        return _article_source(data)
    if extension == ".docx":
        return _docx_source(data, body=word_body, allow_empty=allow_empty_word_body)
    if extension in EXTRACTED_PARAGRAPH_DRAFTS:
        path.write_bytes(data)
        label = EXTRACTED_PARAGRAPH_DRAFTS[extension]
        if extension == ".rtf":
            from app.detectors.text.pipeline import extract_within

            try:
                doc = extract_within(path, extension, skip_hidden_rtf_runs=skip_hidden_rtf_runs)
            except (RuntimeError, StepTimeout):
                raise ValueError("The Rich Text draft could not be read. Save it as DOCX or TXT and try again.") from None
        else:
            doc = extract(path, extension, skip_hidden_odt_sections=skip_hidden_odt_sections)
        if not 0 < len(doc.text) <= MAX_CHARS or "\ufffd" in doc.text or "\x00" in doc.text or not doc.source_map:
            raise ValueError(f"Each {label} version needs readable body text and at most {MAX_CHARS:,} extracted characters.")
        anchors = []
        for index, row in enumerate(doc.source_map):
            # The draft format has always numbered the nonempty decoded RTF blocks consecutively.
            paragraph_index = index if extension == ".rtf" else row.get("paragraph_index", index)
            anchors.append((paragraph_index + 1, paragraph_index, row["start"], row["end"]))
        body = tokens(doc.text, [(start, end) for _, _, start, end in anchors])
        if len(body) > MAX_TOKENS:
            raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
        return doc.text, {"body": body, "quoted": [], "signature": []}, _source_record(data, doc.text), anchors
    if extension == ".pdf":
        doc = document_revision_pdf.extract_native(path, data, mixed_draft=mixed)
        if len(doc.text) > MAX_CHARS:
            raise ValueError(f"Each version can contain up to {MAX_CHARS:,} decoded characters.")
        body = tokens(doc.text, doc.paragraphs)
        if len(body) > MAX_TOKENS:
            raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
        hidden = set(doc.meta.get("hidden_text_pages") or ())
        layout = [row | {"hidden_text_on_page": row["page"] in hidden} for row in doc.source_map]
        return doc.text, {"body": body, "quoted": [], "signature": []}, _source_record(data, doc.text), layout
    if extension == ".eml":
        message = BytesParser(policy=policy.default).parsebytes(data)
        if not any(field in message for field in ("From", "To", "Subject", "Date", "Message-ID", "Content-Type", "MIME-Version")):
            raise ValueError("Choose an .eml file with email headers and a readable message body.")
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None:
            raise ValueError("Choose an email with a readable plain-text or HTML body.")
        if body.get_filename() is not None or body.get_content_disposition() == "attachment":
            raise ValueError("The selected text part is an attachment. A message body is required.")
        mime_part = _part_id(message, body)
        decoded = body.get_content()
    else:
        mime_part = None
    path.write_bytes(data)
    doc = extract(path, extension)
    if extension == ".eml":
        if not mime_part or doc.meta["email"]["body_part"] != mime_part:
            raise ValueError("The selected email body did not match the source map.")
    else:
        decoded = data.decode(doc.meta["encoding"], errors="strict")
    if not isinstance(decoded, str) or "\ufffd" in decoded or "\x00" in decoded:
        raise ValueError("The source text could not be decoded without replacement characters.")
    if len(decoded) > MAX_CHARS or not doc.source_map:
        raise ValueError(f"Each version needs readable text and at most {MAX_CHARS:,} decoded characters.")
    if extension == ".eml" and body.get_content_type() == "text/html":
        return _html_source(data, decoded, doc, mime_part)
    offsets = {role: [] for role in ROLES}
    previous = None
    for block in doc.source_map:
        role = block.get("quote_role", "body")
        ranges = block.get("source_ranges")
        if role not in offsets or not ranges or (role != previous and offsets[role]):
            raise ValueError("Each text, quote and signature section needs a contiguous source role.")
        a, b = ranges[0][2], ranges[-1][3]
        if (not 0 <= a < b <= len(decoded) or block.get("mime_part") != mime_part
                or _normalise(decoded[a:b])[0] != doc.text[block["start"]:block["end"]]):
            raise ValueError("Decoded text positions could not be mapped to the original.")
        offsets[role].append((a, b))
        previous = role
    role_tokens = {role: tokens(decoded, ranges) for role, ranges in offsets.items()}
    if sum(len(rows) for rows in role_tokens.values()) > MAX_TOKENS:
        raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
    return decoded, role_tokens, _source_record(data, decoded, mime_part), offsets["body"] if extension == ".txt" else None


def _excerpt(text: str, extent: list[int] | None, changed_tokens: list[list[int]], layout, extension: str) -> dict | None:
    if extent is None:
        return None
    start, end = extent
    result = {"start": start, "end": end, "text": text[start:end], "changed_tokens": changed_tokens}
    if extension == ".pdf":
        result["pdf_pages"] = document_revision_pdf.page_locations(extent, changed_tokens, layout)
    elif extension == ".docx" or extension in EXTRACTED_PARAGRAPH_DRAFTS:
        result["paragraphs"] = [
            {"paragraph": paragraph, "body_index": index, "start": max(start, a) - a, "end": min(end, b) - a}
            for paragraph, index, a, b in layout
            if max(start, a) < min(end, b) or start == end and a <= start <= b
        ]
    elif extension in {".eml", ".html"} and layout is not None:
        result["html_tokens"] = _html_token_sources(text, layout, changed_tokens)
    return result


def _ordinary_body_indices(text, layout, extension, body):
    if extension == ".txt":
        return {index for index, (start, end) in enumerate(layout)
                if all(not line.lstrip().startswith(">") for line in text[start:end].splitlines())}
    from app.evidence.document_archive_word import ordinary_body_paragraph

    positions = {index for _, index, paragraph in _word_paragraphs(body) if ordinary_body_paragraph(paragraph)}
    return {index for index, (_, body_index, _, _) in enumerate(layout) if body_index in positions}


def _figure_blocks(text, layout, by_role, extension, word_body):
    if extension == ".txt":
        return layout
    if extension == ".docx":
        visible = _ordinary_body_indices(text, layout, extension, word_body)
        return [(a, b) for index, (_, _, a, b) in enumerate(layout) if index in visible]
    if extension in EXTRACTED_PARAGRAPH_DRAFTS:
        return [(a, b) for _, _, a, b in layout]
    if extension == ".html":
        return layout[2]
    if extension == ".pdf":
        if (any(row["hidden_text_on_page"] for row in layout)
                or not document_revision_pdf.ordered_text(
                    text, layout, page_limit=document_revision_pdf.NATIVE_COMPARISON_MAX_PAGES, single_column=True)):
            return []
        return [(row["start"], row["end"]) for row in layout]
    body = _email_body_prefix(text, by_role)
    boundaries = [0, *(match.end() for match in re.finditer(r"\r?\n[ \t]*\r?\n", body)), len(body)]
    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if a < b]


def _figure_comparison(earlier, later):
    versions = (earlier, later)
    tables = [document_revision_values.readings(source[0], _figure_blocks(*source)) for source in versions]
    paired = document_revision_values.compare(*tables)
    if not paired:
        return None
    changed = [pair for pair in paired if pair[0].value != pair[1].value]
    summary = {"schema": document_revision_values.SCHEMA, "scope": document_revision_values.SCOPE,
               "affects_overall_confidence": False, "compared": len(paired), "differences": len(changed), "changes": []}
    for pair in changed:
        if len(summary["changes"]) >= document_revision_values.MAX_RESULTS:
            break
        excerpts = {}
        for side, figure, (text, layout, _, extension, _) in zip(("earlier", "later"), pair, versions, strict=True):
            a, b = figure.span
            # The existing PDF and HTML locator accepts source word extents.
            # A date or scaled figure can cover more than one of those words.
            parts = [[a + part.start(), a + part.end()] for part in re.finditer(r"\S+", text[a:b])]
            try:
                excerpts[side] = _excerpt(text, list(figure.context), parts, layout, extension)
            except ValueError:
                break
            excerpts[f"{side}_value"] = text[a:b]
            excerpts[f"{side}_range"] = [a, b]
        if len(excerpts) == 6:
            summary["changes"].append({"kind": pair[0].kind, **excerpts})
    return summary


def _relocated_paragraph(earlier, later, old_layout, new_layout, extension, word_bodies):
    """Locate one unambiguous paragraph reorder among uniquely matched body paragraphs."""
    def paragraph_index(text, layout):
        rows = [(tuple(text[start:end].split()), [start, end], index)
                for index, (*_, start, end) in enumerate(layout)]
        counts = Counter(key for key, _, _ in rows)
        return {key: (position, index, words) for key, position, index in rows if counts[key] == 1
                and (words := sum(any(char.isalnum() for char in token) for token in key)) >= 4}

    old, new = paragraph_index(earlier, old_layout), paragraph_index(later, new_layout)
    common = old.keys() & new.keys()
    if len(common) < 3 or [key for key in old if key in common] == [key for key in new if key in common]:
        return None

    old_visible = _ordinary_body_indices(earlier, old_layout, extension, word_bodies[0])
    new_visible = _ordinary_body_indices(later, new_layout, extension, word_bodies[1])
    common = {key for key in common if old[key][1] in old_visible and new[key][1] in new_visible}
    if len(common) < 3:
        return None

    old_order = [key for key in old if key in common]
    new_order = [key for key in new if key in common]
    if old_order == new_order:
        return None
    # A one-paragraph reorder must involve one of the first two differing entries.
    index = next(i for i, (a, b) in enumerate(zip(old_order, new_order, strict=True)) if a != b)
    candidates = [key for key in (old_order[index], new_order[index])
                  if [value for value in old_order if value != key] == [value for value in new_order if value != key]]
    if len(candidates) != 1 or old[candidates[0]][2] < 8:
        return None
    key = candidates[0]
    return {"earlier": _excerpt(earlier, old[key][0], [], old_layout, extension),
            "later": _excerpt(later, new[key][0], [], new_layout, extension)}


def _edited_relocation(earlier, later, old_layout, new_layout, extension, word_bodies):
    """Pair one edited paragraph across an otherwise unchanged, ordered body."""
    if len(old_layout) != len(new_layout) or len(old_layout) < 4:
        return None
    bounds = [[row[-2:] for row in layout] for layout in (old_layout, new_layout)]
    old, new = [[tokens(text, [span]) for span in positions]
                for text, positions in zip((earlier, later), bounds, strict=True)]
    left, right = ([tuple(value for value, _, _ in paragraph) for paragraph in side] for side in (old, new))
    common = set(left) & set(right)
    if (len(set(left)) != len(left) or len(set(right)) != len(right) or len(common) != len(left) - 1
            or [row for row in left if row in common] != [row for row in right if row in common]):
        return None
    a, b = (next(index for index, row in enumerate(side) if row not in common) for side in (left, right))
    if abs(a - b) < 2:
        return None
    if (len(_ordinary_body_indices(earlier, old_layout, extension, word_bodies[0])) != len(left)
            or len(_ordinary_body_indices(later, new_layout, extension, word_bodies[1])) != len(right)):
        return None

    before, after = left[a], right[b]
    opening = tuple(pair[0] for pair in takewhile(lambda pair: pair[0] == pair[1], zip(before, after)))
    closing = tuple(pair[0] for pair in takewhile(lambda pair: pair[0] == pair[1], zip(reversed(before), reversed(after))))[::-1]
    shared = len(opening) + len(closing)
    if shared > min(len(before), len(after)) or 5 * shared < 3 * max(len(before), len(after)):
        return None
    for anchor in (opening, closing):
        if sum(any(character.isalnum() for character in token) for token in anchor) < 5:
            return None
        if any(sum(paragraph[index:index + len(anchor)] == anchor for paragraph in side
                   for index in range(len(paragraph) - len(anchor) + 1)) != 1 for side in (left, right)):
            return None
    edits = differences(old[a][len(opening):-len(closing)], new[b][len(opening):-len(closing)])
    return {
        "earlier": _excerpt(earlier, list(bounds[0][a]), [span for edit in edits for span in edit["earlier_changed_tokens"]],
                            old_layout, extension),
        "later": _excerpt(later, list(bounds[1][b]), [span for edit in edits for span in edit["later_changed_tokens"]],
                          new_layout, extension),
    }


def compare(earlier: bytes, later: bytes, extension: str) -> dict:
    result, _, _ = compare_with_text(earlier, later, extension)
    return result


def _email_body_prefix(text: str, by_role: dict) -> str:
    body = by_role["body"]
    if not body:
        return ""
    end = body[-1][2]
    if any(by_role[role] and by_role[role][0][1] < end for role in ("quoted", "signature")):
        return ""
    return text[:end]


def compare_with_text(earlier: bytes, later: bytes, extension: str, *,
                      email_body_only: bool = False, pdf_paired_text_only: bool = False) -> tuple[dict, str, str]:
    """Return the unchanged comparison and ephemeral text under the requested writing scope."""
    result, old_text, new_text, _ = compare_with_sources(earlier, later, extension, email_body_only=email_body_only,
                                                        pdf_paired_text_only=pdf_paired_text_only)
    return result, old_text, new_text


def compare_with_sources(earlier: bytes, later: bytes, extension: str, *,
                         email_body_only: bool = False, pdf_paired_text_only: bool = False, earlier_extension: str | None = None,
                         pdf_independent: bool = False, mixed_paired: bool = False,
                         include_word_tables: bool = True,
                         strict_word_tables: bool = False,
                         include_relocations: bool = True,
                         include_edited_relocations: bool = True,
                         include_values: bool = False,
                         skip_hidden_odt_sections: bool = True,
                         skip_hidden_rtf_runs: bool = True,
                         ) -> tuple[dict, str, str, WordLayouts | PdfLayouts | MixedLayouts | None]:
    """Return ephemeral native layouts for the requested paired sentence scope."""
    if extension not in FORMATS:
        raise ValueError(f"Choose two {SUPPORTED_DRAFTS}.")
    old_extension = extension if earlier_extension is None else earlier_extension
    mixed = old_extension != extension
    if mixed and any(value not in COMPARABLE_DRAFTS for value in (old_extension, extension)):
        raise ValueError("Use TXT, DOCX, ODT, RTF or PDF when comparing different formats. Compare email with email and HTML with HTML.")
    mixed_paired = mixed_paired and all(value in MIXED_DRAFTS for value in (old_extension, extension))
    if any(not 0 < len(item) <= MAX_BYTES for item in (earlier, later)):
        raise ValueError("Each version must be nonempty and at most 1 MiB.")
    changes = []
    relocation = None
    edited_relocation = False
    schema = SCHEMA
    scope, coordinates = SCOPE, TEXT_COORDINATES
    mixed_pdf_scope = MIXED_PDF_SCOPE
    compare_word = extension == ".docx" and not mixed
    check_relocation = include_relocations and not strict_word_tables and not mixed and extension in {".txt", ".docx"}
    word_bodies = (None, None)
    if compare_word and (include_word_tables or check_relocation) or mixed_paired or include_values:
        word_bodies = tuple(docx_body(data) if suffix == ".docx" else None
                            for data, suffix in ((earlier, old_extension), (later, extension)))
    word_tables = None
    if compare_word and include_word_tables and word_bodies[0] is not None:
        if strict_word_tables:
            from app.evidence import document_revision_word_tables as word_table_comparator
        else:
            from app.evidence import document_revision_word_alignment as word_table_comparator
        word_tables = word_table_comparator.compare(*word_bodies)
    has_compared_table = bool(word_tables and word_tables["compared"])
    with TemporaryDirectory(prefix="document-revision-") as staging:
        directory = Path(staging)
        earlier_path, later_path = directory / f"earlier{old_extension}", directory / f"later{extension}"
        old_text, old_tokens, old_source, old_layout = _source(earlier_path, earlier, old_extension, mixed=mixed,
            word_body=word_bodies[0], allow_empty_word_body=has_compared_table, skip_hidden_odt_sections=skip_hidden_odt_sections,
            skip_hidden_rtf_runs=skip_hidden_rtf_runs)
        new_text, new_tokens, new_source, new_layout = _source(later_path, later, extension, mixed=mixed,
            word_body=word_bodies[1], allow_empty_word_body=has_compared_table, skip_hidden_odt_sections=skip_hidden_odt_sections,
            skip_hidden_rtf_runs=skip_hidden_rtf_runs)
        if mixed:
            spaces = {".txt": TEXT_COORDINATES, ".docx": WORD_COORDINATES, ".odt": ODT_COORDINATES,
                      ".rtf": RTF_COORDINATES, ".pdf": document_revision_pdf.COORDINATES}
            for suffix, text, layout, source in ((old_extension, old_text, old_layout, old_source),
                                                 (extension, new_text, new_layout, new_source)):
                if suffix == ".pdf":
                    limit = document_revision_pdf.NATIVE_COMPARISON_MAX_PAGES
                    if not document_revision_pdf.ordered_text(text, layout, page_limit=limit):
                        raise ValueError(f"For drafts in different formats, use an upright PDF with selectable text in a clear reading order. "
                                         f"Multi-page PDFs need a single column per page and up to {limit} pages.")
                    if layout[-1]["page"] > 1:
                        mixed_pdf_scope = MIXED_MULTIPAGE_PDF_SCOPE.format(pages=layout[-1]["page"])
                source.update(format=FORMATS[suffix], coordinate_space=spaces[suffix])
        if extension == ".eml":
            if (old_layout is None) != (new_layout is None):
                raise ValueError("Use two email drafts with the same body type. Both must use plain text or supported HTML.")
            if old_layout is not None:
                schema, scope, coordinates = HTML_SCHEMA, HTML_SCOPE, HTML_COORDINATES
        if extension == ".html":
            schema, scope, coordinates = ARTICLE_SCHEMA, ARTICLE_SCOPE, ARTICLE_COORDINATES
        values = None
        if include_values:
            values = _figure_comparison(
                (old_text, old_layout, old_tokens, old_extension, word_bodies[0]),
                (new_text, new_layout, new_tokens, extension, word_bodies[1]),
            )
        if check_relocation:
            relocation = _relocated_paragraph(old_text, new_text, old_layout, new_layout, extension, word_bodies)
            if relocation is None and include_edited_relocations:
                relocation = _edited_relocation(old_text, new_text, old_layout, new_layout, extension, word_bodies)
                edited_relocation = relocation is not None
            if relocation:
                for side, by_role in (("earlier", old_tokens), ("later", new_tokens)):
                    excerpt = relocation[side]
                    by_role["body"] = [token for token in by_role["body"]
                                       if token[2] <= excerpt["start"] or token[1] >= excerpt["end"]]
        for role in ROLES:
            if role == "body" and (extension == ".html" or old_extension in EXTRACTED_PARAGRAPH_DRAFTS
                                   or extension in EXTRACTED_PARAGRAPH_DRAFTS):
                edit_rows = _localized_body_differences(old_text, new_text, old_layout, new_layout, old_extension,
                                                        extension, old_tokens[role], new_tokens[role])
            else:
                edit_rows = differences(old_tokens[role], new_tokens[role])
            for change in edit_rows:
                before = _excerpt(old_text, change["earlier"], change["earlier_changed_tokens"], old_layout, old_extension)
                after = _excerpt(new_text, change["later"], change["later_changed_tokens"], new_layout, extension)
                changes.append({"id": f"C{len(changes) + 1:02d}", "role": role, "earlier": before, "later": after})
        if ".pdf" in (extension, old_extension):
            value_rows = values["changes"] if values else []
            rendered = document_revision_pdf.render_previews([*changes, *value_rows], earlier_path, later_path)
            if values:
                values["changes"] = rendered[len(changes):]
            changes = rendered[:len(changes)]
            scope, coordinates = document_revision_pdf.SCOPE, document_revision_pdf.COORDINATES
    if extension == ".docx":
        scope, coordinates = DOCX_SCOPE, WORD_COORDINATES
    if extension == ".odt":
        scope, coordinates = ODT_SCOPE, ODT_COORDINATES
    if extension == ".rtf":
        scope, coordinates = RTF_SCOPE, RTF_COORDINATES
    if mixed:
        schema, coordinates = MIXED_SCHEMA, MIXED_COORDINATES
        scopes = {".docx": MIXED_WORD_SCOPE, ".odt": ODT_SCOPE, ".rtf": RTF_SCOPE, ".pdf": mixed_pdf_scope}
        scope = " ".join([MIXED_SCOPE, *(scopes[suffix] for suffix in (old_extension, extension) if suffix in scopes)])
    result = {"schema": schema, "format": "mixed" if mixed else FORMATS[extension],
            "coordinate_space": coordinates,
            "earlier": old_source, "later": new_source, "changes": changes,
            "scope": scope, "interpretation": INTERPRETATION}
    if skip_hidden_odt_sections and ".odt" in (old_extension, extension):
        result[document_native_locations.ODT_VISIBILITY_KEY] = document_native_locations.ODT_VISIBILITY
        result["scope"] += " " + document_native_locations.ODT_VISIBILITY_SCOPE
    if skip_hidden_rtf_runs and ".rtf" in (old_extension, extension):
        result[document_native_locations.RTF_VISIBILITY_KEY] = document_native_locations.RTF_VISIBILITY
        result["scope"] += " " + document_native_locations.RTF_VISIBILITY_SCOPE
    if values is not None:
        result["numeric_comparison"] = values
    if mixed:
        layouts = (old_layout, new_layout)
        verified = mixed_paired and all(
            suffix != ".pdf" or (
                not any(row["hidden_text_on_page"] for row in layout)
                and document_revision_pdf.ordered_text(text, layout, page_limit=document_revision_pdf.NATIVE_COMPARISON_MAX_PAGES,
                                                       single_column=True))
            for suffix, text, layout in zip((old_extension, extension), (old_text, new_text), layouts, strict=True)
        )
        if not verified:
            return result, old_text, new_text, None
        paired_layouts = []
        for suffix, text, layout, body in zip((old_extension, extension), (old_text, new_text), layouts, word_bodies, strict=True):
            if suffix == ".docx":
                eligible = _ordinary_body_indices(text, layout, suffix, body)
                layout = [location for index, location in enumerate(layout) if index in eligible]
            paired_layouts.append(layout)
        return result, old_text, new_text, tuple(paired_layouts)
    if word_tables is not None:
        result.update(schema=word_table_comparator.SCHEMA, scope=word_table_comparator.SCOPE, word_tables=word_tables)
    if relocation:
        result.update(schema=EDITED_RELOCATION_SCHEMA if edited_relocation else RELOCATION_SCHEMA,
                      scope=scope + " " + (EDITED_RELOCATION_SCOPE if edited_relocation else RELOCATION_SCOPE),
                      relocated_passages=[relocation])
    if extension == ".eml" and email_body_only:
        return result, _email_body_prefix(old_text, old_tokens), _email_body_prefix(new_text, new_tokens), None
    if extension == ".pdf" and pdf_paired_text_only:
        if pdf_independent and not mixed:
            layouts = old_layout, new_layout
            limit = document_revision_pdf.NATIVE_COMPARISON_MAX_PAGES
            texts = [document_revision_pdf.ordered_text(text, layout, page_limit=limit, single_column=True)
                     if not any(row["hidden_text_on_page"] for row in layout) else ""
                     for text, layout in zip((old_text, new_text), layouts, strict=True)]
            if all(texts):
                return result, texts[0], texts[1], layouts
        return (result, document_revision_pdf.ordered_text(old_text, old_layout),
                document_revision_pdf.ordered_text(new_text, new_layout), None)
    return result, old_text, new_text, (old_layout, new_layout) if extension == ".docx" else None
