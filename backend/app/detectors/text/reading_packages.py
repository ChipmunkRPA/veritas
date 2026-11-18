"""Bounded, offline readers for slide text and EPUB reading-order text."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile, ZipFile

from lxml import etree

from app.core.config import settings
from app.storage.files import EPUB_MIME, check_document_archive_safe

PRESENTATION_NAMESPACES = {
    "{http://schemas.openxmlformats.org/presentationml/2006/main}": (
        "{http://schemas.openxmlformats.org/drawingml/2006/main}",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    ),
    "{http://purl.oclc.org/ooxml/presentationml/main}": (
        "{http://purl.oclc.org/ooxml/drawingml/main}",
        "http://purl.oclc.org/ooxml/officeDocument/relationships",
    ),
}
RELATIONSHIPS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
CONTENT_TYPES = "{http://schemas.openxmlformats.org/package/2006/content-types}"
PRESENTATION_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
PACKAGE = "{http://www.idpf.org/2007/opf}"
XHTML = "{http://www.w3.org/1999/xhtml}"
ENCRYPTION = "{http://www.w3.org/2001/04/xmlenc#}"
MAX_PART_BYTES = 6 * 1024 * 1024
MAX_TOTAL_XML_BYTES = 24 * 1024 * 1024
MAX_TEXT_PARTS = 600
MAX_BLOCKS = 8000
HTML_BLOCKS = {
    "address", "article", "aside", "blockquote", "caption", "dd", "div", "dl", "dt", "figcaption", "figure",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "ol", "p", "pre", "section",
    "table", "td", "th", "tr", "ul",
}
HTML_OMITTED = {"audio", "canvas", "form", "iframe", "math", "nav", "script", "style", "svg", "template", "video"}


def _part_name(base: str, target: str) -> str:
    uri = urlsplit(target)
    path = unquote(uri.path, errors="strict")
    if (not path or uri.scheme or uri.netloc or uri.query or uri.fragment or "\\" in path
            or any(ord(char) < 32 for char in path)):
        raise ValueError("A document part does not point to a local package entry")
    parts = [] if path.startswith("/") else list(PurePosixPath(base).parts[:-1])
    for part in path.lstrip("/").split("/"):
        if part == "..":
            if not parts:
                raise ValueError("A document part points outside the package")
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    if not parts:
        raise ValueError("A document part has no package entry")
    return "/".join(parts)


class _Package:
    def __init__(self, archive: ZipFile) -> None:
        self.archive = archive
        self.remaining = MAX_TOTAL_XML_BYTES

    def xml(self, name: str, *, xhtml: bool = False) -> etree._Element:
        entry = self.archive.getinfo(name)
        if not 0 < entry.file_size <= min(MAX_PART_BYTES, self.remaining):
            raise ValueError("The document's text parts exceed the local reader limit")
        self.remaining -= entry.file_size
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        root = etree.fromstring(self.archive.read(entry), parser=parser)
        info = root.getroottree().docinfo
        if info.doctype and (not xhtml or info.internalDTD.entities()):
            raise ValueError("The document's text parts use unsupported XML declarations")
        return root

    def relationships(self, name: str) -> dict:
        root = self.xml(name)
        rows = {}
        if root.tag != RELATIONSHIPS + "Relationships":
            raise ValueError("The presentation has an invalid relationship table")
        for row in root.findall(RELATIONSHIPS + "Relationship"):
            key = row.get("Id")
            if not key or key in rows:
                raise ValueError("The presentation has ambiguous relationships")
            rows[key] = row
        return rows


def _slide_paragraphs(shapes, presentation, drawing):
    for shape in shapes:
        if shape.tag == presentation + "grpSp":
            yield from _slide_paragraphs(shape, presentation, drawing)
        elif shape.tag == presentation + "sp":
            properties = shape.find(f"{presentation}nvSpPr/{presentation}cNvPr")
            if properties is None or properties.get("hidden") not in ("1", "true"):
                yield from shape.findall(f"{presentation}txBody/{drawing}p")
        elif shape.tag == presentation + "graphicFrame":
            properties = shape.find(f"{presentation}nvGraphicFramePr/{presentation}cNvPr")
            if properties is not None and properties.get("hidden") in ("1", "true"):
                continue
            table = shape.find(f"{drawing}graphic/{drawing}graphicData/{drawing}tbl")
            if table is None:
                continue
            for cell in table.findall(f"{drawing}tr/{drawing}tc"):
                if all(cell.get(key) not in ("1", "true") for key in ("hMerge", "vMerge")):
                    yield from cell.findall(f"{drawing}txBody/{drawing}p")


def _powerpoint(package: _Package):
    main = [
        row for row in package.relationships("_rels/.rels").values()
        if row.get("Type") in {f"{value[1]}/officeDocument" for value in PRESENTATION_NAMESPACES.values()}
        and row.get("TargetMode", "Internal") == "Internal"
    ]
    if len(main) != 1:
        raise ValueError("The PPTX has no unique presentation part")
    main_name = _part_name("", main[0].get("Target", ""))
    types = package.xml("[Content_Types].xml")
    if types.tag != CONTENT_TYPES + "Types" or [
        row.get("ContentType") for row in types.findall(CONTENT_TYPES + "Override")
        if row.get("PartName") == "/" + main_name
    ] != [PRESENTATION_TYPE]:
        raise ValueError("Save the presentation as a standard, macro-free PowerPoint .pptx file")
    root = package.xml(main_name)
    presentation = next((ns for ns in PRESENTATION_NAMESPACES if root.tag == ns + "presentation"), None)
    if presentation is None:
        raise ValueError("The PPTX presentation part is unreadable")
    drawing, relationship = PRESENTATION_NAMESPACES[presentation]
    part = PurePosixPath(main_name)
    links = package.relationships(str(part.parent / "_rels" / (part.name + ".rels")))
    slides = root.findall(f"{presentation}sldIdLst/{presentation}sldId")
    if not 0 < len(slides) <= MAX_TEXT_PARTS:
        raise ValueError(f"Upload a presentation with between 1 and {MAX_TEXT_PARTS} slides")

    seen = set()
    for index, row in enumerate(slides):
        link = links.get(row.get(f"{{{relationship}}}id"))
        if link is None or link.get("Type") != f"{relationship}/slide" or link.get("TargetMode", "Internal") != "Internal":
            raise ValueError("The PPTX is missing an internal slide relationship")
        name = _part_name(main_name, link.get("Target", ""))
        if name in seen:
            raise ValueError("The PPTX repeats a slide relationship")
        seen.add(name)
        slide = package.xml(name)
        tree = slide.find(f"{presentation}cSld/{presentation}spTree")
        if slide.tag != presentation + "sld" or tree is None:
            raise ValueError("The PPTX has an unreadable slide")
        if slide.get("show") in ("0", "false"):
            continue
        for paragraph_index, paragraph in enumerate(_slide_paragraphs(tree, presentation, drawing)):
            value = "".join(
                "\n" if child.tag == drawing + "br" else child.findtext(drawing + "t", default="")
                for child in paragraph if child.tag in {drawing + "r", drawing + "fld", drawing + "br"}
            )
            yield value, index, paragraph_index


def _chapter_blocks(body):
    text = []

    def flush():
        if text:
            value = "".join(text)
            text.clear()
            if value.strip():
                yield value

    def walk(element):
        if not isinstance(element.tag, str) or not element.tag.startswith(XHTML):
            return
        tag = element.tag.removeprefix(XHTML)
        if tag in HTML_OMITTED or element.get("hidden") is not None or element.get("aria-hidden") == "true":
            return
        if tag in HTML_BLOCKS:
            yield from flush()
        if tag == "br":
            text.append("\n")
        text.append(element.text or "")
        for child in element:
            yield from walk(child)
            text.append(child.tail or "")
        if tag in HTML_BLOCKS:
            yield from flush()

    yield from walk(body)
    yield from flush()


def _ebook(package: _Package):
    with package.archive.open("mimetype") as media_type:
        if media_type.read(len(EPUB_MIME) + 1) != EPUB_MIME.encode("ascii"):
            raise ValueError("This archive is not an EPUB book")
    container = package.xml("META-INF/container.xml")
    rootfile = container.find(f"{CONTAINER}rootfiles/{CONTAINER}rootfile")
    if container.tag != CONTAINER + "container" or rootfile is None:
        raise ValueError("The EPUB is missing its main book file")
    book_name = _part_name("", rootfile.get("full-path", ""))
    book = package.xml(book_name)
    spine = book.findall(f"{PACKAGE}spine/{PACKAGE}itemref")
    if book.tag != PACKAGE + "package" or not 0 < len(spine) <= MAX_TEXT_PARTS:
        raise ValueError(f"Upload an EPUB with between 1 and {MAX_TEXT_PARTS} reading-order entries")
    items = {}
    for item in book.findall(f"{PACKAGE}manifest/{PACKAGE}item"):
        key = item.get("id")
        if not key or key in items:
            raise ValueError("The EPUB has ambiguous manifest entries")
        items[key] = item
    encrypted = set()
    if "META-INF/encryption.xml" in package.archive.namelist():
        encrypted = {
            _part_name("", row.get("URI", ""))
            for row in package.xml("META-INF/encryption.xml").iter(ENCRYPTION + "CipherReference")
        }
    for index, row in enumerate(spine):
        if row.get("linear") == "no":
            continue
        item = items.get(row.get("idref"))
        if item is None:
            raise ValueError("The EPUB reading order refers to a missing chapter")
        if item.get("media-type") != "application/xhtml+xml" or "nav" in item.get("properties", "").split():
            continue
        name = _part_name(book_name, item.get("href", ""))
        if name in encrypted:
            raise ValueError("The EPUB contains protected reading text. Upload an unlocked EPUB or PDF")
        chapter = package.xml(name, xhtml=True)
        body = chapter.find(XHTML + "body")
        if chapter.tag != XHTML + "html" or body is None:
            raise ValueError("The EPUB contains an unreadable XHTML chapter")
        for paragraph_index, text in enumerate(_chapter_blocks(body)):
            yield text, index, paragraph_index


def body_parts(path: Path, extension: str, *, check_crc: bool = False) -> list[tuple[str, dict]]:
    check_document_archive_safe(path, extension, check_crc=check_crc)
    reader = {".pptx": _powerpoint, ".epub": _ebook}[extension]
    part_kind = "slide" if extension == ".pptx" else "epub_entry"
    parts = []
    characters = raw_characters = 0
    limit = settings.max_document_chars
    try:
        with ZipFile(path) as archive:
            for text, index, paragraph_index in reader(_Package(archive)):
                if not text.strip():
                    continue
                raw_characters += len(text)
                characters += len(" ".join(text.split())) + (2 if parts else 0)
                if characters > limit or raw_characters > 4 * limit:
                    raise ValueError(f"The document exceeds the {limit:,}-character writing limit. Upload a shorter excerpt")
                if len(parts) == MAX_BLOCKS:
                    raise ValueError("The document has too many text blocks. Upload a shorter excerpt")
                parts.append((text, {"kind": "markup", "part_kind": part_kind, "part_index": index,
                                     "paragraph_index": paragraph_index}))
    except (KeyError, BadZipFile, NotImplementedError, etree.XMLSyntaxError) as error:
        raise ValueError(f"The {extension[1:].upper()} archive is damaged or uses unsupported text parts") from error
    return parts
