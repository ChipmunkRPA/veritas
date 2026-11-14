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
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from app.detectors.text.extract import _normalise, extract, text_hash
from app.evidence import document_revision_pdf
from app.evidence.document_scope import email_agreement_body_spans, mapped_email_html
from app.storage.files import check_docx_safe

SCHEMA = "chipmunk.document-revision-comparison/1"
HTML_SCHEMA = "chipmunk.document-revision-comparison/2"
MIXED_SCHEMA = "chipmunk.document-revision-comparison/3"
RELOCATION_SCHEMA = "chipmunk.document-revision-comparison/6"
HTML_COORDINATES = "email_html_extracted_text_unicode_codepoint"
MIXED_COORDINATES = "per_source_unicode_codepoint"
TOKENS = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")
MAX_BYTES = 1024 * 1024
MAX_CHARS = 30_000
MAX_TOKENS = 2_000
MAX_DOCX_XML_BYTES = 2 * 1024 * 1024
ROLES = ("body", "quoted", "signature")
FORMATS = {".eml": "email", ".txt": "plain_text", ".docx": "docx", ".pdf": "pdf"}
MIXED_DRAFTS = (".txt", ".docx", ".pdf")
TEXT_COORDINATES = "decoded_source_unicode_codepoint"
WORD_COORDINATES = "docx_body_unicode_codepoint"
SUPPORTED_DRAFTS = "matching .eml, .txt, .docx or .pdf files"
SCOPE = "Compares words and punctuation in the decoded text. Email body, quoted text and signatures are mapped separately."
HTML_SCOPE = ("Compares visible words and punctuation in the reply body. Quoted text, attachments, headers and "
              "formatting are outside this comparison.")
DOCX_SCOPE = ("Compares words and punctuation in top-level Word body paragraphs. Tables, headers, footers, text boxes, "
              "embedded content, formatting and tracked revision markup are outside this comparison.")
MIXED_SCOPE = "Compares words and punctuation across file formats. Formatting is outside the comparison."
MIXED_WORD_SCOPE = "Word coverage includes top-level body paragraphs. Word tables, headers, footers and text boxes are excluded."
MIXED_PDF_SCOPE = "PDF coverage includes the selectable text on one upright page with a clear reading order. Scanned text is excluded."
MIXED_MULTIPAGE_PDF_SCOPE = "PDF coverage includes selectable text on all {pages} upright, single-column pages. Scanned text is excluded."
DOCX_PART = "word/document.xml"
INTERPRETATION = "This comparison locates changes between the supplied versions. The revision sequence and authorship require independent evidence."
RELOCATION_SCOPE = "The reordered passage has the same wording and punctuation in both versions. Its position changed relative to other matching paragraphs."
WordParagraphs = list[tuple[int, int, int, int]]
WordLayouts = tuple[WordParagraphs, WordParagraphs]


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


def docx_body(data):
    """Read the bounded main part used by the earlier-draft comparison."""
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn
    from lxml.etree import XMLSyntaxError

    check_docx_safe(BytesIO(data))
    try:
        with ZipFile(BytesIO(data)) as archive:
            declarations = parse_xml(_docx_part(archive, "[Content_Types].xml"))
            namespace = "{http://schemas.openxmlformats.org/package/2006/content-types}"
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
            if not any(node.get("PartName") == f"/{DOCX_PART}" and node.get("ContentType") == content_type
                       for node in declarations.findall(namespace + "Override")):
                raise ValueError("Choose a Word document saved in the standard .docx format.")
            document = parse_xml(_docx_part(archive, DOCX_PART))
    except (BadZipFile, RuntimeError, NotImplementedError, XMLSyntaxError) as error:
        raise ValueError("The Word document body could not be read.") from error
    if (document.tag != qn("w:document") or (body := document.find(qn("w:body"))) is None
            or document.getroottree().docinfo.doctype or declarations.getroottree().docinfo.doctype):
        raise ValueError("The Word document body could not be read.")
    return body


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


def _html_token_sources(text, layout, changed_tokens):
    decoded, mapping = layout
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
            raise ValueError("A changed token could not be mapped to the decoded HTML reply.")
        located.append({"start": start, "end": end, "source_ranges": spans})
    return located


def _source(path: Path, data: bytes, extension: str, *, mixed: bool = False, word_body=None, allow_empty_word_body=False):
    if extension == ".docx":
        return _docx_source(data, body=word_body, allow_empty=allow_empty_word_body)
    if extension == ".pdf":
        doc = document_revision_pdf.extract_native(path, data, mixed_draft=mixed)
        if len(doc.text) > MAX_CHARS:
            raise ValueError(f"Each version can contain up to {MAX_CHARS:,} decoded characters.")
        body = tokens(doc.text, doc.paragraphs)
        if len(body) > MAX_TOKENS:
            raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
        return doc.text, {"body": body, "quoted": [], "signature": []}, _source_record(data, doc.text), doc.source_map
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
    elif extension == ".docx":
        result["paragraphs"] = [
            {"paragraph": paragraph, "body_index": index, "start": max(start, a) - a, "end": min(end, b) - a}
            for paragraph, index, a, b in layout
            if max(start, a) < min(end, b) or start == end and a <= start <= b
        ]
    elif extension == ".eml" and layout is not None:
        result["html_tokens"] = _html_token_sources(text, layout, changed_tokens)
    return result


def _relocated_paragraph(earlier, later, old_layout, new_layout, extension, word_bodies):
    """Locate one unambiguous paragraph reorder among uniquely matched body paragraphs."""
    word = extension == ".docx"

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

    if word:
        from docx.oxml.ns import qn
        from docx.text.paragraph import Paragraph

        from app.evidence.document_archive_word import ordinary_body_paragraph

        visibility = [{i for i, paragraph in enumerate(body)
                       if paragraph.tag == qn("w:p") and ordinary_body_paragraph(Paragraph(paragraph, None))}
                      for body in word_bodies]
        common = {key for key in common if old_layout[old[key][1]][1] in visibility[0]
                  and new_layout[new[key][1]][1] in visibility[1]}
    else:
        def unquoted(text, position):
            return all(not line.lstrip().startswith(">") for line in text[slice(*position)].splitlines())

        common = {key for key in common if unquoted(earlier, old[key][0]) and unquoted(later, new[key][0])}
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
                         include_word_tables: bool = True,
                         strict_word_tables: bool = False,
                         include_relocations: bool = True,
                         ) -> tuple[dict, str, str, WordLayouts | None]:
    """Keep the native Word paragraph locations available for exact paired sentence mapping."""
    if extension not in FORMATS:
        raise ValueError(f"Choose two {SUPPORTED_DRAFTS}.")
    old_extension = extension if earlier_extension is None else earlier_extension
    mixed = old_extension != extension
    if mixed and any(value not in MIXED_DRAFTS for value in (old_extension, extension)):
        raise ValueError("Use TXT, DOCX or PDF when comparing drafts across formats. Email drafts require matching EML files.")
    if any(not 0 < len(item) <= MAX_BYTES for item in (earlier, later)):
        raise ValueError("Each version must be nonempty and at most 1 MiB.")
    changes = []
    relocation = None
    schema = SCHEMA
    scope, coordinates = SCOPE, TEXT_COORDINATES
    mixed_pdf_scope = MIXED_PDF_SCOPE
    compare_word = extension == ".docx" and not mixed
    check_relocation = include_relocations and not strict_word_tables and not mixed and extension in {".txt", ".docx"}
    word_bodies = (docx_body(earlier), docx_body(later)) if compare_word and (include_word_tables or check_relocation) else (None, None)
    word_tables = None
    if include_word_tables and word_bodies[0] is not None:
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
            word_body=word_bodies[0], allow_empty_word_body=has_compared_table)
        new_text, new_tokens, new_source, new_layout = _source(later_path, later, extension, mixed=mixed,
            word_body=word_bodies[1], allow_empty_word_body=has_compared_table)
        if mixed:
            spaces = {".txt": TEXT_COORDINATES, ".docx": WORD_COORDINATES, ".pdf": document_revision_pdf.COORDINATES}
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
        if check_relocation:
            relocation = _relocated_paragraph(old_text, new_text, old_layout, new_layout, extension, word_bodies)
            if relocation:
                for side, by_role in (("earlier", old_tokens), ("later", new_tokens)):
                    excerpt = relocation[side]
                    by_role["body"] = [token for token in by_role["body"]
                                       if token[2] <= excerpt["start"] or token[1] >= excerpt["end"]]
        for role in ROLES:
            for change in differences(old_tokens[role], new_tokens[role]):
                before = _excerpt(old_text, change["earlier"], change["earlier_changed_tokens"], old_layout, old_extension)
                after = _excerpt(new_text, change["later"], change["later_changed_tokens"], new_layout, extension)
                changes.append({"id": f"C{len(changes) + 1:02d}", "role": role, "earlier": before, "later": after})
        if ".pdf" in (extension, old_extension):
            changes = document_revision_pdf.render_previews(changes, earlier_path, later_path)
            scope, coordinates = document_revision_pdf.SCOPE, document_revision_pdf.COORDINATES
    if extension == ".docx":
        scope, coordinates = DOCX_SCOPE, WORD_COORDINATES
    if mixed:
        schema, coordinates = MIXED_SCHEMA, MIXED_COORDINATES
        scopes = {".docx": MIXED_WORD_SCOPE, ".pdf": mixed_pdf_scope}
        scope = " ".join([MIXED_SCOPE, *(scopes[suffix] for suffix in (old_extension, extension) if suffix in scopes)])
    result = {"schema": schema, "format": "mixed" if mixed else FORMATS[extension],
            "coordinate_space": coordinates,
            "earlier": old_source, "later": new_source, "changes": changes,
            "scope": scope, "interpretation": INTERPRETATION}
    if mixed:
        return result, old_text, new_text, None
    if word_tables is not None:
        result.update(schema=word_table_comparator.SCHEMA, scope=word_table_comparator.SCOPE, word_tables=word_tables)
    if relocation:
        result.update(schema=RELOCATION_SCHEMA, scope=scope + " " + RELOCATION_SCOPE,
                      relocated_passages=[relocation])
    if extension == ".eml" and email_body_only:
        return result, _email_body_prefix(old_text, old_tokens), _email_body_prefix(new_text, new_tokens), None
    if extension == ".pdf" and pdf_paired_text_only:
        return (result, document_revision_pdf.ordered_text(old_text, old_layout),
                document_revision_pdf.ordered_text(new_text, new_layout), None)
    return result, old_text, new_text, (old_layout, new_layout) if extension == ".docx" else None
