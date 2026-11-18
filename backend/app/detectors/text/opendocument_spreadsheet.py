"""Read saved ODS cell text and its sheet, row and column without rendering."""

from __future__ import annotations

import re
from bisect import bisect_left

from app.core.config import settings
from app.detectors.text.opendocument import OFFICE, PARAGRAPHS, TABLE, TEXT, _visible_text
from app.detectors.text.reading_packages import MAX_TEXT_PARTS
from app.detectors.text.spreadsheet_cells import MAX_COLUMNS, MAX_ROWS
from app.storage.files import ODS_MIME

STYLE = "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}"
HIDDEN = frozenset({"false", "0"})


def _end(element, axis, first, ceiling):
    count = element.get(TABLE + f"number-{axis}-repeated", "1").strip()
    if not re.fullmatch(r"\+?[0-9]{1,7}", count) or not 1 <= int(count) <= ceiling - first + 1:
        raise ValueError(f"The ODS workbook has an invalid repeated {axis} range")
    return first + int(count)


def _items(element, axis, visible=True):
    tag = TABLE + f"table-{axis}"
    group = tag + "-group"
    containers = {group, tag + "s", TABLE + f"table-header-{axis}s"}
    for child in element:
        if child.tag == tag:
            yield child, visible and child.get(TABLE + "visibility", "visible") == "visible"
        elif child.tag in containers:
            yield from _items(child, axis, visible and child.get(TABLE + "display") not in HIDDEN)


def _columns(sheet):
    columns = []
    first = 1
    for element, visible in _items(sheet, "column"):
        stop = _end(element, "columns", first, MAX_COLUMNS)
        if visible:
            columns.extend(range(first, stop))
        first = stop
    columns.extend(range(first, MAX_COLUMNS + 1))
    return columns


def _space(element):
    if element.tag == TEXT + "tab":
        return "\t"
    if element.tag == TEXT + "line-break":
        return "\n"
    count = element.get(TEXT + "c", "1").strip()
    if not re.fullmatch(r"\+?[0-9]{1,7}", count) or not 1 <= int(count) <= 4 * settings.max_document_chars:
        raise ValueError("The ODS workbook exceeds the writing limit for whitespace in a cell")
    return " " * int(count)


def _text(cell):
    if cell.get(OFFICE + "value-type") != "string" or TABLE + "formula" in cell.attrib:
        return ""
    paragraphs = []
    size = 0
    for child in cell:
        if child.tag not in PARAGRAPHS:
            continue
        fragments = []
        size += bool(paragraphs)
        for fragment in _visible_text(child, _space):
            size += len(fragment)
            if size > 4 * settings.max_document_chars:
                raise ValueError("The ODS workbook exceeds the writing limit for a cell")
            fragments.append(fragment)
        paragraphs.append("".join(fragments))
    return "\n".join(paragraphs) if paragraphs else cell.get(OFFICE + "string-value", "")


def _entries(element, columns):
    first = 1
    for cell in element:
        if cell.tag not in {TABLE + "table-cell", TABLE + "covered-table-cell"}:
            continue
        stop = _end(cell, "columns", first, MAX_COLUMNS)
        if cell.tag == TABLE + "table-cell":
            begin, end = bisect_left(columns, first), bisect_left(columns, stop)
            if begin < end:
                value = _text(cell)
                if any(character.isalpha() for character in value):
                    yield value, begin, end
        first = stop


def _sheet_cells(sheet):
    columns = _columns(sheet)
    row = 1
    for element, visible in _items(sheet, "row"):
        stop = _end(element, "rows", row, MAX_ROWS)
        if visible:
            entries = tuple(_entries(element, columns))
            if entries:
                for number in range(row, stop):
                    for value, first, end in entries:
                        for column_index in range(first, end):
                            yield value, number, columns[column_index]
        row = stop


def _styles(package, content):
    styles = {}
    roots = [content]
    if "styles.xml" in package.archive.NameToInfo:
        roots.insert(0, package.xml("styles.xml"))
    for root in roots:
        for name in ("styles", "automatic-styles"):
            for style in root.iterfind(f"{OFFICE}{name}/{STYLE}style"):
                if style.get(STYLE + "family") == "table" and (key := style.get(STYLE + "name")):
                    styles[key] = style
    return styles


def _sheet_visible(sheet, styles):
    if sheet.get(TABLE + "display") in HIDDEN or sheet.find(TABLE + "table-source") is not None:
        return False
    name = sheet.get(TABLE + "style-name")
    seen = set()
    while name in styles:
        if name in seen or len(seen) == 64:
            raise ValueError("The ODS workbook has unsupported nested sheet styles")
        seen.add(name)
        style = styles[name]
        properties = style.find(STYLE + "table-properties")
        if properties is not None and (value := properties.get(TABLE + "display")) is not None:
            return value not in HIDDEN
        name = style.get(STYLE + "parent-style-name")
    return True


def workbook_cells(package):
    with package.archive.open("mimetype") as mime:
        if mime.read(len(ODS_MIME) + 1) != ODS_MIME.encode("ascii"):
            raise ValueError("This archive is not an OpenDocument Spreadsheet file")
    content = package.xml("content.xml")
    body = content.find(f"{OFFICE}body/{OFFICE}spreadsheet")
    if content.tag != OFFICE + "document-content" or body is None:
        raise ValueError("The ODS workbook has no supported spreadsheet body")
    sheets = body.findall(TABLE + "table")
    if not 0 < len(sheets) <= MAX_TEXT_PARTS:
        raise ValueError(f"Upload an ODS workbook with between 1 and {MAX_TEXT_PARTS} sheets")
    styles = _styles(package, content)
    for index, sheet in enumerate(sheets):
        if _sheet_visible(sheet, styles):
            for value, row, column in _sheet_cells(sheet):
                yield value, index, row, column
