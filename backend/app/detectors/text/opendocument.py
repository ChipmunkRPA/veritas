"""Read body paragraphs from an OpenDocument Text archive without rendering it."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from app.storage.files import ODT_MIME, check_document_archive_safe

OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
STYLE = "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
BODY_CONTAINERS = frozenset({TEXT + name for name in ("section", "list", "list-item", "list-header")}
                            | {TABLE + name for name in ("table", "table-row", "table-cell", "table-header-rows", "table-rows", "table-row-group")})
PARAGRAPHS = frozenset({TEXT + "p", TEXT + "h"})
INLINE_TEXT = frozenset({TEXT + "span", TEXT + "a"})
INLINE_SPACE = frozenset({TEXT + "s", TEXT + "tab", TEXT + "line-break"})
MAX_BODY_XML_BYTES = 20 * 1024 * 1024
_TEXT_REFERENCE_FIELDS = {
    TEXT + "reference-ref": (TEXT + "reference-mark", TEXT + "reference-mark-start", TEXT + "reference-mark-end"),
    TEXT + "bookmark-ref": (TEXT + "bookmark", TEXT + "bookmark-start", TEXT + "bookmark-end"),
}
_MARKER_TAGS = frozenset(tag for tags in _TEXT_REFERENCE_FIELDS.values() for tag in tags)
_SIMPLE_PARAGRAPH = INLINE_SPACE | _MARKER_TAGS | _TEXT_REFERENCE_FIELDS.keys()


def _matching_reference_fields(body) -> set[etree._Element]:
    """Find saved text that agrees with one ordinary same-body marked range."""
    paragraphs = {node for node in body if node.tag in PARAGRAPHS
                  and all(child.tag in _SIMPLE_PARAGRAPH and (
                      child.tag != TEXT + "s" or child.get(TEXT + "c", "1") == "1") for child in node)}
    fields = [node for paragraph in paragraphs for node in paragraph if node.tag in _TEXT_REFERENCE_FIELDS
              and node.get(TEXT + "reference-format") == "text" and not len(node)
              and node.text and len(node.text) <= 1024 and node.text.isprintable() and node.text.strip()]
    if not fields:
        return set()

    markers = defaultdict(list)
    for node in body.iter(*_MARKER_TAGS):
        markers[node.tag, node.get(TEXT + "name")].append(node)
    matches = set()
    for field in fields:
        name = field.get(TEXT + "ref-name")
        point_tag, start_tag, end_tag = _TEXT_REFERENCE_FIELDS[field.tag]
        if not name or markers.get((point_tag, name)):
            continue
        starts, ends = markers.get((start_tag, name), ()), markers.get((end_tag, name), ())
        if len(starts) != 1 or len(ends) != 1:
            continue
        start, end = starts[0], ends[0]
        if (start.getparent() in paragraphs and start.getnext() is end and not len(start) and not len(end)
                and not start.text and not end.text and start.tail == field.text):
            matches.add(field)
    # A skipped field could move later characters when a viewer renders the paragraph.
    incomplete = {paragraph for paragraph in paragraphs if any(
        child.tag in _TEXT_REFERENCE_FIELDS and child not in matches for child in paragraph)}
    return {field for field in matches if field.getparent() not in incomplete}


def _visible_text(element, decode_space=None, matching_fields=frozenset()):
    yield element.text or ""
    for child in element:
        if child.tag in INLINE_TEXT:
            yield from _visible_text(child, decode_space, matching_fields)
        elif child.tag in INLINE_SPACE:
            yield decode_space(child) if decode_space is not None else " "
        elif child in matching_fields:
            yield child.text
        yield child.tail or ""


def _paragraphs(element, *, skip_hidden_sections: bool, matching_fields):
    for child in element:
        if child.tag in PARAGRAPHS:
            yield "".join(_visible_text(child, matching_fields=matching_fields))
        elif child.tag in BODY_CONTAINERS:
            if (skip_hidden_sections and child.tag == TEXT + "section"
                    and child.get(TEXT + "display") in {"none", "condition"}):
                continue
            yield from _paragraphs(child, skip_hidden_sections=skip_hidden_sections, matching_fields=matching_fields)


def body_paragraphs(path: Path, *, check_crc: bool = False, skip_hidden_sections: bool = True,
                    include_matching_text_references: bool = False) -> list[str]:
    check_document_archive_safe(path, ".odt", check_crc=check_crc)
    try:
        with ZipFile(path) as archive:
            with archive.open("mimetype") as kind:
                if kind.read(len(ODT_MIME) + 1) != ODT_MIME.encode("ascii"):
                    raise ValueError("This archive is not an OpenDocument Text file")
            with archive.open("content.xml") as stream:
                source = stream.read(MAX_BODY_XML_BYTES + 1)
        if len(source) > MAX_BODY_XML_BYTES:
            raise ValueError("The ODT body XML exceeds the document reader limit")
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        root = etree.fromstring(source, parser)
        body = root.find(f"{OFFICE}body/{OFFICE}text")
        if root.tag != OFFICE + "document-content" or root.getroottree().docinfo.doctype or body is None:
            raise ValueError("The ODT file has no supported body text")
    except (KeyError, etree.XMLSyntaxError) as error:
        raise ValueError("The ODT file has no readable body text") from error
    matching_fields = _matching_reference_fields(body) if include_matching_text_references else frozenset()
    paragraphs = [value for value in _paragraphs(body, skip_hidden_sections=skip_hidden_sections,
                                                  matching_fields=matching_fields) if value.strip()]
    if not paragraphs:
        raise ValueError("The ODT file has no readable body paragraphs")
    return paragraphs
