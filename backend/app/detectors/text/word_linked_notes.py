# SPDX-License-Identifier: Apache-2.0
"""Read saved Word footnotes and endnotes linked from the main story."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from zipfile import ZipFile

from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from lxml.etree import XMLSyntaxError

from app.detectors.text import strict_word, word_saved_fields
from app.evidence.document_word_visibility import paragraph_run_elements

PART_KINDS = ("footnote", "endnote")
REF_TAGS = {qn(f"w:{kind}Reference"): kind for kind in PART_KINDS}
ID = qn("w:id")
MAX_ID = 2**31 - 1
MAX_NOTE_BYTES = 4 * 1024 * 1024
MAX_REFERENCES = 4096
MAX_NOTE_PARAGRAPHS = 4096
OFFICE_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml."
_PARAGRAPH_CHILDREN = {qn("w:" + name) for name in ("pPr", "r", "hyperlink", "proofErr", "bookmarkStart", "bookmarkEnd")}
_HYPERLINK_CHILDREN = {qn("w:" + name) for name in ("r", "proofErr", "bookmarkStart", "bookmarkEnd")}
_RUN_CHILDREN = {qn("w:" + name) for name in (
    "rPr", "t", "tab", "br", "cr", "noBreakHyphen", "softHyphen", "footnoteRef", "endnoteRef",
)}


def _id(element) -> int | None:
    value = element.get(ID)
    if value is None or not re.fullmatch(r"[0-9]{1,10}", value):
        return None
    note = int(value)
    return note if note <= MAX_ID else None


def _visible_run(element, paragraph) -> bool:
    visibility = element.xpath("./w:rPr/w:vanish | ./w:rPr/w:specVanish | ./w:rPr/w:webHidden")
    if any(node.get(qn("w:val")) not in {"0", "false", "off"} for node in visibility):
        return False
    return word_saved_fields.style_allows_visible_runs(Run(element, paragraph).style)


def references(paragraph) -> Iterator[tuple[str, int, int, int]]:
    if not word_saved_fields.paragraph_allows_visible_runs(paragraph):
        return
    # Cached text and references inside inline fields may depend on recalculation.
    if paragraph._p.xpath(".//w:fldChar | .//w:fldSimple | .//w:instrText"):
        return
    for run_index, run in enumerate(paragraph_run_elements(paragraph)):
        refs = [(index, item) for index, item in enumerate(run) if item.tag in REF_TAGS]
        if refs and _visible_run(run, paragraph):
            for index, node in refs:
                if (number := _id(node)) is not None:
                    yield REF_TAGS[node.tag], number, run_index, index


def _entries(archive, relationships, declarations, directory, kind):
    tag = kind + "s"
    targets = [
        (value, namespace == strict_word.RELATIONSHIP)
        for namespace in (strict_word.RELATIONSHIP, strict_word.TRANSITIONAL_RELATIONSHIP)
        if (value := strict_word._target(relationships, tag, directory, namespace=namespace))
    ]
    if len(targets) != 1 or declarations.tag != strict_word.CONTENT_TYPES + "Types":
        return {}
    path, strict = targets[0]
    types = [node.get("ContentType") for node in declarations.findall(strict_word.CONTENT_TYPES + "Override")
             if node.get("PartName") == "/" + path]
    if types != [f"{OFFICE_TYPE}{tag}+xml"]:
        return {}
    root = strict_word._word_xml(archive, path, tag, MAX_NOTE_BYTES, strict=strict)
    entries = {}
    duplicates = set()
    for child in root.findall(qn(f"w:{kind}")):
        number = _id(child)
        if number is None:
            continue
        if number in entries:
            duplicates.add(number)
        entries[number] = child
    return {number: element for number, element in entries.items()
            if number not in duplicates and element.get(qn("w:type"), "normal") == "normal"}


def _read_note(element, document):
    for index, child in enumerate(element):
        if child.tag != qn("w:p"):
            continue
        paragraph = Paragraph(child, document)
        if (any(node.tag not in _PARAGRAPH_CHILDREN for node in child)
                or any(node.tag not in _HYPERLINK_CHILDREN for link in child.findall(qn("w:hyperlink")) for node in link)
                or not word_saved_fields.paragraph_allows_visible_runs(paragraph)):
            continue
        runs = paragraph_run_elements(paragraph)
        if any(any(node.tag not in _RUN_CHILDREN for node in run) or not _visible_run(run, paragraph) for run in runs):
            continue
        if (text := paragraph.text).strip():
            yield index, text


def paragraphs(path, document, refs):
    """Return supported saved paragraphs in body-reference order, once per native note."""
    if not refs:
        return []
    part = document.part.partname
    notes = {}
    with ZipFile(path) as archive:
        try:
            relationships = strict_word._read_xml(archive, part.rels_uri.membername, MAX_NOTE_BYTES)
            declarations = strict_word._read_xml(archive, "[Content_Types].xml", MAX_NOTE_BYTES)
        except (ValueError, KeyError, XMLSyntaxError):
            return []
        for kind in {item[0] for item in refs}:
            try:
                notes[kind] = _entries(archive, relationships, declarations, posixpath.dirname(part.membername), kind)
            except (ValueError, KeyError, XMLSyntaxError):
                notes[kind] = {}
    result = []
    for (kind, note_id), (body_index, run_index, child_index) in refs.items():
        if (element := notes[kind].get(note_id)) is None:
            continue
        for native_index, text in _read_note(element, document):
            if len(result) == MAX_NOTE_PARAGRAPHS:
                return result
            result.append((text, {"kind": "markup", "part_kind": kind, "note_id": note_id,
                                  "note_paragraph_index": native_index, "reference_body_index": body_index,
                                  "reference_run_index": run_index, "reference_child_index": child_index}))
    return result
