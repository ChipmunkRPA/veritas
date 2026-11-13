"""Source-bound wording changes in matching, simple top-level Word tables."""
from __future__ import annotations

from dataclasses import dataclass

from app.evidence.document_revision import MAX_CHARS, MAX_TOKENS, differences, tokens

SCHEMA = "chipmunk.document-revision-comparison/4"
SCOPE = ("Compares text in top-level Word paragraphs and matching simple tables. Each compared table keeps the same "
         "position, headings, row labels and cell layout in both drafts. Merged cells, nested tables and tracked markup "
         "are outside this table check. Headers, footers and text boxes are outside this comparison. Table ranges do not assign authorship.")
MAX_CELLS = 96
MAX_TABLES_RETAINED = 96
UNSUPPORTED = {"ins", "del", "moveFrom", "moveTo", "cellIns", "cellDel", "cellMerge", "pPrChange", "rPrChange",
               "tblPrChange", "tblGridChange", "trPrChange", "tcPrChange"}


@dataclass
class Cell:
    text: str
    paragraphs: list[tuple[int, int, int, int]]
    tokens: list[tuple[str, int, int]]

    def words(self):
        return tuple(word for word, _, _ in self.tokens)

    def excerpt(self, extent, changed):
        start, end = extent if extent is not None else (0, 0)
        return {"start": start, "end": end, "text": self.text[start:end], "changed_tokens": changed,
                "paragraphs": [{"paragraph": number, "cell_index": index, "start": max(start, a) - a,
                                "end": min(end, b) - a} for number, index, a, b in self.paragraphs
                               if max(start, a) < min(end, b) or start == end and a <= start <= b]}


def _tag(name):
    return f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{name}"


def _elements(parent):
    return (child for child in parent if isinstance(child.tag, str))


def _children(parent, *names):
    allowed = {_tag(name) for name in names}
    return all(child.tag in allowed for child in _elements(parent))


def _cell(element, max_chars, max_tokens):
    from docx.text.paragraph import Paragraph

    if not _children(element, "tcPr", "p"):
        return None
    if (properties := element.find(_tag("tcPr"))) is not None and any(
            properties.find(_tag(name)) is not None for name in ("gridSpan", "hMerge", "vMerge", "cellIns", "cellDel", "cellMerge")):
        return None
    paragraphs, values, cursor, number = [], [], 0, 0
    for index, item in enumerate(_elements(element)):
        if item.tag != _tag("p"):
            continue
        number += 1
        if not _children(item, "pPr", "r", "hyperlink", "bookmarkStart", "bookmarkEnd"):
            return None
        if any(not _children(link, "r") for link in item.findall(_tag("hyperlink"))):
            return None
        if any(not _children(run, "rPr", "t", "tab", "br", "cr") for run in item.iter(_tag("r"))):
            return None
        value = Paragraph(item, None).text
        if not value.strip():
            continue
        start = cursor + (2 if values else 0)
        cursor = start + len(value)
        if cursor > max_chars:
            raise ComparisonLimit()
        values.append(value)
        paragraphs.append((number, index, start, cursor))
    text = "\n\n".join(values)
    body = tokens(text, [(a, b) for _, _, a, b in paragraphs])
    if len(body) > max_tokens:
        raise ComparisonLimit()
    return Cell(text, paragraphs, body)


class ComparisonLimit(Exception):
    pass


def _table(table, budget):
    if not _children(table, "tblPr", "tblGrid", "tr"):
        return None
    grid = table.find(_tag("tblGrid"))
    columns = len(grid.findall(_tag("gridCol"))) if grid is not None else 0
    native_rows = table.findall(_tag("tr"))
    if columns < 2 or len(native_rows) < 2:
        return None
    cells = len(native_rows) * columns
    if cells > budget["cells"]:
        raise ComparisonLimit()
    caps = {**budget, "cells": budget["cells"] - cells}
    tags = {_tag(name) for name in UNSUPPORTED}
    if any(item.tag in tags for item in table.iter()):
        return None
    rows = []
    for row in native_rows:
        cells = row.findall(_tag("tc"))
        if len(cells) != columns or not _children(row, "trPr", "tc"):
            return None
        if (properties := row.find(_tag("trPr"))) is not None and any(
                properties.find(_tag(name)) is not None for name in ("gridBefore", "gridAfter", "ins", "del")):
            return None
        row_cells = []
        for cell in cells:
            value = _cell(cell, caps["chars"], caps["tokens"])
            if value is None:
                return None
            caps["chars"] -= len(value.text)
            caps["tokens"] -= len(value.tokens)
            row_cells.append(value)
        rows.append(row_cells)
    return rows, caps


def _tables(body):
    return {index: (number, table) for number, (index, table) in enumerate(
        ((index, node) for index, node in enumerate(_elements(body)) if node.tag == _tag("tbl")), start=1)}


def _same_rows(earlier, later):
    if earlier is None or later is None or (len(earlier), len(earlier[0])) != (len(later), len(later[0])):
        return False
    if [cell.words() for cell in earlier[0]] != [cell.words() for cell in later[0]] or any(not cell.tokens for cell in earlier[0]):
        return False
    labels = [row[0].words() for row in earlier[1:]]
    return all(labels) and len(set(labels)) == len(labels) and labels == [row[0].words() for row in later[1:]]


def _compare(index, before_num, after_num, earlier, later, start_id):
    changes = []
    for row, (old_cells, new_cells) in enumerate(zip(earlier, later, strict=True), start=1):
        for column, (old, new) in enumerate(zip(old_cells, new_cells, strict=True), start=1):
            for edit in differences(old.tokens, new.tokens):
                changes.append({"id": f"T{start_id + len(changes):02d}", "row": row, "column": column,
                                "earlier": old.excerpt(edit["earlier"], edit["earlier_changed_tokens"]),
                                "later": new.excerpt(edit["later"], edit["later_changed_tokens"])})
    return {"body_index": index, "earlier_table": before_num, "later_table": after_num,
            "rows": len(earlier), "columns": len(earlier[0]), "changes": changes}


def compare(earlier, later):
    old_tables, new_tables = _tables(earlier), _tables(later)
    if not old_tables and not new_tables:
        return None
    tables, skipped = [], []
    old_budget = {"cells": MAX_CELLS, "tokens": MAX_TOKENS, "chars": MAX_CHARS}
    new_budget = dict(old_budget)
    next_id = 1
    positions = sorted(old_tables.keys() | new_tables.keys())
    for index in positions[:MAX_TABLES_RETAINED]:
        before_num, before_xml = old_tables.get(index, (None, None))
        after_num, after_xml = new_tables.get(index, (None, None))
        if before_xml is None or after_xml is None:
            reason = "no_matching_position"
        else:
            try:
                first, second = _table(before_xml, old_budget), _table(after_xml, new_budget)
            except ComparisonLimit:
                reason = "comparison_limit"
            else:
                if first is None or second is None:
                    reason = "unsupported_layout"
                elif not _same_rows(first[0], second[0]):
                    reason = "changed_structure"
                else:
                    old, old_budget = first
                    new, new_budget = second
                    entry = _compare(index, before_num, after_num, old, new, next_id)
                    tables.append(entry)
                    next_id += len(entry["changes"])
                    continue
        skipped.append({"body_index": index, "earlier_table": before_num, "later_table": after_num, "reason": reason})
    return {"coordinate_space": "docx_table_cell_unicode_codepoint", "compared": tables, "skipped": skipped,
            "omitted_positions": max(0, len(positions) - MAX_TABLES_RETAINED)}
