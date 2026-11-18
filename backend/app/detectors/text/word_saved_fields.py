# SPDX-License-Identifier: Apache-2.0
"""Project saved display text from a bounded subset of Word simple fields."""

from __future__ import annotations

from docx.oxml.ns import qn

from app.evidence import document_native_locations as locations

_SIMPLE = qn("w:fldSimple")
_COMPLEX = qn("w:fldChar")
_MARKER_TYPE = qn("w:fldCharType")
_RUN = qn("w:r")
_PROPERTIES = qn("w:rPr")
_TEXT = qn("w:t")
_HYPERLINK = qn("w:hyperlink")
_VISIBILITY_NODES = "*[self::w:vanish or self::w:specVanish or self::w:webHidden]"
_SAVED_COMMANDS = frozenset({
    "DATE", "TIME", "CREATEDATE", "SAVEDATE", "PRINTDATE", "SEQ", "REF", "PAGEREF",
    "PAGE", "NUMPAGES", "SECTION", "SECTIONPAGES", "DOCPROPERTY", "DOCVARIABLE", "MERGEFIELD",
})


def defaults_allow_visible_runs(document) -> bool:
    return not document.styles.element.xpath(f"./w:docDefaults/w:rPrDefault/w:rPr/{_VISIBILITY_NODES}")


def complex_field_blocks(body) -> set:
    """Identify body blocks inside a complex field, including across paragraphs."""
    depth = 0
    affected = set()
    for block in body:
        if depth:
            affected.add(block)
        for marker in block.iter(_COMPLEX):
            affected.add(block)
            if marker.get(_MARKER_TYPE) == "begin":
                depth += 1
            elif marker.get(_MARKER_TYPE) == "end":
                depth = max(0, depth - 1)
    return affected


def _body_style_allows_visible_runs(paragraph) -> bool:
    if paragraph._p.xpath(f"./w:pPr/w:rPr/{_VISIBILITY_NODES}"):
        return False
    style = paragraph.style
    visited = set()
    while style is not None:
        if style.style_id in visited or len(visited) >= 32:
            return False
        if style.element.xpath(f"./w:rPr/{_VISIBILITY_NODES}"):
            return False
        visited.add(style.style_id)
        style = style.base_style
    return True


def _saved_value(field) -> str | None:
    instruction = field.get(qn("w:instr"), "")
    if not 0 < len(instruction) <= 1024 or not (parts := instruction.split(maxsplit=1)) or parts[0].upper() not in _SAVED_COMMANDS:
        return None
    if not 0 < len(field) <= 32:
        return None
    for run in field:
        if run.tag != _RUN or any(item.tag not in {_PROPERTIES, _TEXT} for item in run):
            return None
        if run.xpath("./w:rPr/w:vanish | ./w:rPr/w:specVanish | ./w:rPr/w:webHidden | ./w:rPr/w:rStyle"):
            return None
    value = "".join(run.text for run in field)
    if 0 < len(value) <= locations.MAX_SAVED_FIELD_CHARS and value.strip() and value.isprintable():
        return value
    return None


def project(paragraph, slots: int) -> list[tuple[int, int, int]]:
    """Move supported saved runs into the in-memory paragraph, in source order."""
    element = paragraph._p
    if (element.find(_SIMPLE) is None or element.xpath(".//w:fldChar | .//w:instrText | .//w:sdt")
            or not _body_style_allows_visible_runs(paragraph)):
        return []
    selected = []
    offset = 0
    for index, child in enumerate(list(element)):
        if len(selected) == slots:
            break
        if child.tag in {_RUN, _HYPERLINK}:
            offset += len(child.text)
        elif child.tag == _SIMPLE and (value := _saved_value(child)) is not None:
            for run in list(child):
                child.addprevious(run)
            element.remove(child)
            start = offset + len(value) - len(value.lstrip())
            selected.append((index, start, offset + len(value.rstrip())))
            offset += len(value)
    return selected
