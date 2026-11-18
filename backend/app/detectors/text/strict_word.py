"""Read ISO/IEC 29500 Strict Word body markup using the existing DOCX text reader."""

from __future__ import annotations

import posixpath
from zipfile import ZipFile

from lxml import etree

from app.evidence.document_tracked_changes import PART_NAME
from app.storage.files import check_docx_safe

STRICT = "http://purl.oclc.org/ooxml/"
WORD = STRICT + "wordprocessingml/main"
RELATIONSHIP = STRICT + "officeDocument/relationships/"
TRANSITIONAL_RELATIONSHIP = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
RELATIONS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
CONTENT_TYPES = "{http://schemas.openxmlformats.org/package/2006/content-types}"
MAIN_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
MAX_XML_BYTES = 20 * 1024 * 1024
NAMESPACES = {
    WORD: "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    RELATIONSHIP.rstrip("/"): "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _read_xml(archive: ZipFile, name: str, limit: int):
    member = archive.getinfo(name)
    if not 0 < member.file_size <= limit:
        raise ValueError("A Word XML part exceeds the reader limit or is empty.")
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
    with archive.open(member) as stream:
        content = stream.read(limit + 1)
    if len(content) != member.file_size:
        raise ValueError("The Word file contains a damaged XML part.")
    root = etree.fromstring(content, parser)
    if root.getroottree().docinfo.doctype:
        raise ValueError("Word files with XML document type declarations are unsupported.")
    return root


def _target(relationships, name: str, directory: str = "", *, namespace: str = RELATIONSHIP) -> str | None:
    if relationships.tag != RELATIONS + "Relationships":
        return None
    matches = [node for node in relationships.findall(RELATIONS + "Relationship")
               if node.get("Type") == namespace + name]
    if not matches:
        return None
    target = matches[0].get("Target", "")
    if len(matches) != 1 or matches[0].get("TargetMode", "Internal") != "Internal":
        raise ValueError("The Strict Word document has an invalid package relationship.")
    path = posixpath.normpath(target[1:] if target.startswith("/") else posixpath.join(directory, target))
    if not target or not PART_NAME.fullmatch(path):
        raise ValueError("The Strict Word document has an unsupported XML part path.")
    return path


def _mapped_name(name: str) -> str:
    namespace, _, local = name.removeprefix("{").partition("}")
    replacement = NAMESPACES.get(namespace)
    return f"{{{replacement}}}{local}" if replacement else name


def _word_xml(archive: ZipFile, path: str, tag: str, limit: int, *, strict: bool = True):
    from docx.oxml import parse_xml

    root = _read_xml(archive, path, limit)
    namespace = WORD if strict else NAMESPACES[WORD]
    if root.tag != f"{{{namespace}}}{tag}":
        raise ValueError("The Word document contains an unexpected XML part.")
    if not strict:
        return parse_xml(etree.tostring(root))
    for element in root.iter(etree.Element):
        element.tag = _mapped_name(element.tag)
        for name in list(element.attrib):
            mapped = _mapped_name(name)
            if mapped != name:
                if mapped in element.attrib:
                    raise ValueError("The Strict Word document mixes conflicting XML attributes.")
                element.set(mapped, element.attrib.pop(name))
    return parse_xml(etree.tostring(root))


def read_document(archive: ZipFile, *, include_styles: bool = True, max_xml_bytes: int = MAX_XML_BYTES):
    """Use the internal Strict relationship and leave the original package untouched."""
    from docx.opc.constants import CONTENT_TYPE, RELATIONSHIP_TYPE
    from docx.opc.packuri import PackURI
    from docx.oxml.ns import qn
    from docx.parts.document import DocumentPart
    from docx.parts.styles import StylesPart

    if "_rels/.rels" not in archive.namelist():
        return None
    try:
        if not (path := _target(_read_xml(archive, "_rels/.rels", max_xml_bytes), "officeDocument")):
            return None
        declarations = _read_xml(archive, "[Content_Types].xml", max_xml_bytes)
        content = [row.get("ContentType") for row in declarations.findall(CONTENT_TYPES + "Override")
                   if row.get("PartName") == "/" + path]
        if declarations.tag != CONTENT_TYPES + "Types" or content != [MAIN_TYPE]:
            raise ValueError("Choose a Strict Word document saved in .docx format.")
        root = _word_xml(archive, path, "document", max_xml_bytes)
        if len(root.findall(qn("w:body"))) != 1:
            raise ValueError("The Strict Word document has no readable body.")
        part = DocumentPart(PackURI("/" + path), MAIN_TYPE, root, None)
        if include_styles and part.partname.rels_uri.membername in archive.namelist():
            relationships = _read_xml(archive, part.partname.rels_uri.membername, max_xml_bytes)
            if styles_path := _target(relationships, "styles", posixpath.dirname(path)):
                style_root = _word_xml(archive, styles_path, "styles", max_xml_bytes)
                style_part = StylesPart(PackURI("/" + styles_path), CONTENT_TYPE.WML_STYLES, style_root, None)
                part.relate_to(style_part, RELATIONSHIP_TYPE.STYLES)
    except (KeyError, etree.XMLSyntaxError) as error:
        raise ValueError("The Strict Word document could not be read.") from error
    return part.document


def open_document(path):
    """Keep the usual parser for other DOCX files and fall back on Strict packages."""
    import docx

    try:
        return docx.Document(str(path))
    except KeyError:
        check_docx_safe(path)
        with ZipFile(path) as archive:
            document = read_document(archive)
        if document is None:
            raise
        return document
