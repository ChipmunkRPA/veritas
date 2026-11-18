"""Read literal prose cells from macro-free SpreadsheetML packages offline."""

from __future__ import annotations

import re
from itertools import accumulate
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from lxml import etree

from app.detectors.text.reading_packages import (
    CONTENT_TYPES,
    MAX_BLOCKS,
    MAX_TEXT_PARTS,
    _Package,
    _part_name,
)
from app.storage.files import check_document_archive_safe

NAMESPACES = {
    "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "{http://purl.oclc.org/ooxml/spreadsheetml/main}": "http://purl.oclc.org/ooxml/officeDocument/relationships",
}
WORKBOOK_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
MAX_ROWS, MAX_COLUMNS = 1048576, 16384
CELL_ADDRESS = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})\Z")
ESCAPE = re.compile(r"_x([0-9a-fA-F]{4})_")
TRUE = {"1", "true"}


def _target(base: str, link) -> str:
    if link is None or link.get("TargetMode", "Internal") != "Internal":
        raise ValueError("The XLSX workbook is missing an internal relationship")
    return _part_name(base, link.get("Target", ""))


def _string(element, ns: str) -> str:
    if element is None:
        return ""
    texts = []
    for child in element:
        text = child if child.tag == ns + "t" else child.find(ns + "t") if child.tag == ns + "r" else None
        if text is None or not text.text:
            continue
        value = ESCAPE.sub(lambda match: chr(int(match[1], 16)), text.text)
        texts.append(value)
    return "".join(texts).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _index(value: str, ceiling: int) -> int:
    if not re.fullmatch(r"[0-9]{1,9}", value):
        raise ValueError("The XLSX workbook has an invalid cell coordinate or string index")
    index = int(value)
    if index > ceiling:
        raise ValueError("The XLSX workbook has a cell coordinate or string index outside its range")
    return index


def _hidden_columns(sheet, ns: str) -> list[int]:
    changes = [0] * (MAX_COLUMNS + 2)
    for column in sheet.iterfind(f"{ns}cols/{ns}col"):
        first = _index(column.get("min", ""), MAX_COLUMNS)
        last = _index(column.get("max", ""), MAX_COLUMNS)
        if not 1 <= first <= last:
            raise ValueError("The XLSX workbook has an invalid column range")
        if column.get("hidden") in TRUE or column.get("width") in {"0", "0.0"}:
            changes[first] += 1
            changes[last + 1] -= 1
    return list(accumulate(changes))


def _worksheet_cells(sheet, ns: str, shared: list[str]):
    rows = sheet.find(ns + "sheetData")
    if sheet.tag != ns + "worksheet" or rows is None:
        raise ValueError("The XLSX workbook contains an unreadable worksheet")
    hidden_columns = _hidden_columns(sheet, ns)
    defaults = sheet.find(ns + "sheetFormatPr")
    rows_hidden_by_default = defaults is not None and defaults.get("zeroHeight") in TRUE
    previous_row = 0
    for row in rows.iterfind(ns + "row"):
        row_number = _index(row.get("r", str(previous_row + 1)), MAX_ROWS)
        # Rows with no explicit reference may derive their coordinate from the first cell.
        first_cell = row.find(ns + "c")
        if row.get("r") is None and first_cell is not None and first_cell.get("r"):
            address = CELL_ADDRESS.fullmatch(first_cell.get("r"))
            if address:
                row_number = _index(address[2], MAX_ROWS)
        if row_number <= previous_row:
            raise ValueError("The XLSX workbook has repeated or unordered worksheet rows")
        previous_row = row_number
        if (row.get("hidden") in TRUE or row.get("ht") in {"0", "0.0"}
                or (rows_hidden_by_default and row.get("hidden") is None and row.get("ht") is None)):
            continue
        previous_column = 0
        for cell in row.iterfind(ns + "c"):
            column = previous_column + 1
            if address := cell.get("r"):
                match = CELL_ADDRESS.fullmatch(address)
                if match is None or int(match[2]) != row_number:
                    raise ValueError("The XLSX workbook has an invalid cell reference")
                column = 0
                for letter in match[1]:
                    column = 26 * column + ord(letter) - ord("A") + 1
            if not previous_column < column <= MAX_COLUMNS:
                raise ValueError("The XLSX workbook has repeated, unordered or out-of-range cells")
            previous_column = column
            if hidden_columns[column] or cell.find(ns + "f") is not None:
                continue
            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                value = _string(cell.find(ns + "is"), ns)
            elif cell_type == "s":
                value = shared[_index(cell.findtext(ns + "v", default=""), len(shared) - 1)]
            else:
                continue
            if any(character.isalpha() for character in value):
                yield value, row_number, column


def _workbook_cells(package: _Package):
    offices = {relationship + "/officeDocument" for relationship in NAMESPACES.values()}
    workbook_links = [link for link in package.relationships("_rels/.rels").values() if link.get("Type") in offices]
    if len(workbook_links) != 1:
        raise ValueError("The XLSX file has no unique workbook part")
    name = _target("", workbook_links[0])
    types = package.xml("[Content_Types].xml")
    if types.tag != CONTENT_TYPES + "Types" or [
        entry.get("ContentType") for entry in types.iterfind(CONTENT_TYPES + "Override")
        if entry.get("PartName") == "/" + name
    ] != [WORKBOOK_TYPE] or any("vbaproject" in entry.get("ContentType", "").lower() for entry in types):
        raise ValueError("Save the workbook as a standard, macro-free Excel XLSX file")
    workbook = package.xml(name)
    ns = next((candidate for candidate in NAMESPACES if workbook.tag == candidate + "workbook"), None)
    if ns is None:
        raise ValueError("The XLSX workbook part is unreadable")
    relationship = NAMESPACES[ns]
    part = PurePosixPath(name)
    links = package.relationships(str(part.parent / "_rels" / (part.name + ".rels")))
    sheets = workbook.findall(f"{ns}sheets/{ns}sheet")
    if not 0 < len(sheets) <= MAX_TEXT_PARTS:
        raise ValueError(f"Upload an XLSX workbook with between 1 and {MAX_TEXT_PARTS} sheets")
    string_links = [link for link in links.values() if link.get("Type") == relationship + "/sharedStrings"]
    if len(string_links) > 1:
        raise ValueError("The XLSX workbook has ambiguous shared text")
    shared = []
    if string_links:
        strings = package.xml(_target(name, string_links[0]))
        if strings.tag != ns + "sst":
            raise ValueError("The XLSX workbook has unreadable shared text")
        shared = [_string(item, ns) for item in strings.iterfind(ns + "si")]
    seen = set()
    for index, sheet in enumerate(sheets):
        state = sheet.get("state", "visible")
        if state in {"hidden", "veryHidden"}:
            continue
        link = links.get(sheet.get(f"{{{relationship}}}id"))
        if state != "visible" or link is None:
            raise ValueError("The XLSX workbook has an unreadable sheet entry")
        link_type = link.get("Type")
        if link_type in {relationship + "/chartsheet", relationship + "/dialogsheet"}:
            continue
        if link_type != relationship + "/worksheet":
            raise ValueError("The XLSX workbook contains an unsupported sheet type")
        target = _target(name, link)
        if target in seen:
            raise ValueError("The XLSX workbook repeats a worksheet relationship")
        seen.add(target)
        for value, row, column in _worksheet_cells(package.xml(target), ns, shared):
            yield value, index, row, column


def read(path: Path, *, max_chars: int) -> list[tuple[str, dict]]:
    check_document_archive_safe(path, ".xlsx")
    parts = []
    characters = raw_characters = 0
    try:
        with ZipFile(path) as archive:
            for value, sheet, row, column in _workbook_cells(_Package(archive)):
                raw_characters += len(value)
                characters += len(" ".join(value.split())) + (2 if parts else 0)
                if characters > max_chars or raw_characters > 4 * max_chars or len(parts) == MAX_BLOCKS:
                    raise ValueError("The XLSX workbook exceeds the writing limit. Upload fewer cells or a shorter excerpt.")
                parts.append((value, {"kind": "markup", "part_kind": "spreadsheet_cell", "part_index": sheet,
                                      "paragraph_index": len(parts), "row": row, "column": column}))
    except (KeyError, BadZipFile, NotImplementedError, etree.XMLSyntaxError) as error:
        raise ValueError("The XLSX workbook is damaged or uses unsupported text parts") from error
    return parts
