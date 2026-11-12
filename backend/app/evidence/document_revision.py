"""Compare two reviewer-supplied text versions without a model or persistence."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from app.detectors.text.extract import _normalise, extract, text_hash
from app.evidence import document_revision_pdf
from app.storage.files import check_docx_safe

SCHEMA = "chipmunk.document-revision-comparison/1"
TOKENS = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")
MAX_BYTES = 1024 * 1024
MAX_CHARS = 30_000
MAX_TOKENS = 2_000
MAX_DOCX_XML_BYTES = 2 * 1024 * 1024
ROLES = ("body", "quoted", "signature")
FORMATS = {".eml": "email", ".txt": "plain_text", ".docx": "docx", ".pdf": "pdf"}
SUPPORTED_DRAFTS = "matching .eml, .txt, .docx or .pdf files"
SCOPE = "Compares words and punctuation in the decoded text. Email body, quoted text and signatures are mapped separately."
DOCX_SCOPE = ("Compares words and punctuation in top-level Word body paragraphs. Tables, headers, footers, text boxes, "
              "embedded content, formatting and tracked revision markup are outside this comparison.")
DOCX_PART = "word/document.xml"
INTERPRETATION = "This comparison locates changes between the supplied versions. The revision sequence and authorship require independent evidence."


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


def _docx_source(data):
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
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
    paragraphs, anchors, paragraph_number, cursor = [], [], 0, 0
    for body_index, element in enumerate(body):
        if element.tag != qn("w:p"):
            continue
        paragraph_number += 1
        value = Paragraph(element, None).text
        if not value.strip():
            continue
        start = cursor + (2 if paragraphs else 0)
        cursor = start + len(value)
        if cursor > MAX_CHARS:
            raise ValueError(f"Each version can contain up to {MAX_CHARS:,} decoded body characters.")
        paragraphs.append(value)
        anchors.append((paragraph_number, body_index, start, cursor))
    if not paragraphs:
        raise ValueError("Each Word version needs readable top-level body paragraphs.")
    decoded = "\n\n".join(paragraphs)
    body_tokens = tokens(decoded, [(start, end) for _, _, start, end in anchors])
    if len(body_tokens) > MAX_TOKENS:
        raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
    return decoded, {"body": body_tokens, "quoted": [], "signature": []}, _source_record(data, decoded), anchors


def _source(path: Path, data: bytes, extension: str):
    if extension == ".docx":
        return _docx_source(data)
    if extension == ".pdf":
        doc = document_revision_pdf.extract_native(path, data)
        if len(doc.text) > MAX_CHARS:
            raise ValueError(f"Each version can contain up to {MAX_CHARS:,} decoded characters.")
        body = tokens(doc.text, doc.paragraphs)
        if len(body) > MAX_TOKENS:
            raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
        return doc.text, {"body": body, "quoted": [], "signature": []}, _source_record(data, doc.text), doc.source_map
    if extension == ".eml":
        message = BytesParser(policy=policy.default).parsebytes(data)
        if not any(field in message for field in ("From", "To", "Subject", "Date", "Message-ID", "Content-Type", "MIME-Version")):
            raise ValueError("Choose an .eml file with email headers and a plain-text body.")
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None or body.get_content_type() != "text/plain":
            raise ValueError("Choose an email with a readable plain-text body.")
        if body.get_filename() is not None or body.get_content_disposition() == "attachment":
            raise ValueError("The selected plain-text part is an attachment. A plain-text message body is required.")
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
    return decoded, role_tokens, _source_record(data, decoded, mime_part), None


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
    return result


def compare(earlier: bytes, later: bytes, extension: str) -> dict:
    result, _, _ = compare_with_text(earlier, later, extension)
    return result


def compare_with_text(earlier: bytes, later: bytes, extension: str) -> tuple[dict, str, str]:
    """Return the comparison and the same ephemeral decoded text used for its offsets."""
    if extension not in FORMATS:
        raise ValueError(f"Choose two {SUPPORTED_DRAFTS}.")
    if any(not 0 < len(item) <= MAX_BYTES for item in (earlier, later)):
        raise ValueError("Each version must be nonempty and at most 1 MiB.")
    changes = []
    scope, coordinates = SCOPE, "decoded_source_unicode_codepoint"
    with TemporaryDirectory(prefix="document-revision-") as staging:
        directory = Path(staging)
        earlier_path, later_path = directory / f"earlier{extension}", directory / f"later{extension}"
        old_text, old_tokens, old_source, old_layout = _source(earlier_path, earlier, extension)
        new_text, new_tokens, new_source, new_layout = _source(later_path, later, extension)
        for role in ROLES:
            for change in differences(old_tokens[role], new_tokens[role]):
                before = _excerpt(old_text, change["earlier"], change["earlier_changed_tokens"], old_layout, extension)
                after = _excerpt(new_text, change["later"], change["later_changed_tokens"], new_layout, extension)
                changes.append({"id": f"C{len(changes) + 1:02d}", "role": role, "earlier": before, "later": after})
        if extension == ".pdf":
            changes = document_revision_pdf.render_previews(changes, earlier_path, later_path)
            scope, coordinates = document_revision_pdf.SCOPE, document_revision_pdf.COORDINATES
    if extension == ".docx":
        scope, coordinates = DOCX_SCOPE, "docx_body_unicode_codepoint"
    result = {"schema": SCHEMA, "format": FORMATS[extension],
            "coordinate_space": coordinates,
            "earlier": old_source, "later": new_source, "changes": changes,
            "scope": scope, "interpretation": INTERPRETATION}
    return result, old_text, new_text
