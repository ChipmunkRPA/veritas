"""Read body paragraphs from an OpenDocument Text archive without rendering it."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from app.storage.files import ODT_MIME, check_document_archive_safe

OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
BODY_CONTAINERS = frozenset({TEXT + name for name in ("section", "list", "list-item", "list-header")}
                            | {TABLE + name for name in ("table", "table-row", "table-cell", "table-header-rows", "table-rows", "table-row-group")})
PARAGRAPHS = frozenset({TEXT + "p", TEXT + "h"})
INLINE_TEXT = frozenset({TEXT + "span", TEXT + "a"})
INLINE_SPACE = frozenset({TEXT + "s", TEXT + "tab", TEXT + "line-break"})
MAX_BODY_XML_BYTES = 20 * 1024 * 1024


def _visible_text(element):
    yield element.text or ""
    for child in element:
        if child.tag in INLINE_TEXT:
            yield from _visible_text(child)
        elif child.tag in INLINE_SPACE:
            yield " "
        yield child.tail or ""


def _paragraphs(element):
    for child in element:
        if child.tag in PARAGRAPHS:
            yield "".join(_visible_text(child))
        elif child.tag in BODY_CONTAINERS:
            yield from _paragraphs(child)


def body_paragraphs(path: Path, *, check_crc: bool = False) -> list[str]:
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
    paragraphs = [value for value in _paragraphs(body) if value.strip()]
    if not paragraphs:
        raise ValueError("The ODT file has no readable body paragraphs")
    return paragraphs
