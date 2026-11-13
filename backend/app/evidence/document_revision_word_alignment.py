"""Match reorganized Word tables by exact, source-local labels."""
from __future__ import annotations

from collections import Counter

from app.evidence.document_revision import MAX_CHARS, MAX_TOKENS, differences
from app.evidence.document_revision_word_tables import (
    MAX_CELLS,
    MAX_TABLES_RETAINED,
    ComparisonLimit,
    _table,
    _tables,
    _tag,
)

SCHEMA = "chipmunk.document-revision-comparison/5"
SCOPE = ("Compares top-level Word paragraphs and matched table cells. Reordered table rows and columns use exact, unique labels. "
         "A moved table needs a shared, unique Word caption. Unmatched or repeated labels are listed as unchecked. "
         "Merged cells, nested tables, tracked markup, headers, footers and text boxes are outside this comparison. "
         "Table differences do not assign authorship.")


def _caption(table):
    props = table.find(_tag("tblPr"))
    items = props.findall(_tag("tblCaption")) if props is not None else []
    if not items:
        return None
    if len(items) != 1:
        return ""
    caption = " ".join(items[0].get(_tag("val"), "").split())
    return caption if 0 < len(caption) <= 240 else ""


def _unique(entries):
    counts = Counter(key for key, _ in entries)
    return {key: index for key, index in entries if key and counts[key] == 1}


def _label_matches(old, new):
    left, right = _unique(old), _unique(new)
    keys = left.keys() & right.keys()
    paired = sorted((left[key], right[key]) for key in keys)
    excluded = {"earlier": len(old) - len(paired), "later": len(new) - len(paired)}
    return paired, excluded


def _alignment(old, new):
    headers = [[(cell.words(), i) for i, cell in enumerate(rows[0][1:], start=1)] for rows in (old, new)]
    if not old[0][0].tokens or old[0][0].words() != new[0][0].words():
        return None
    columns, unchecked_columns = _label_matches(*headers)
    labels = [[(row[0].words(), i) for i, row in enumerate(rows[1:], start=1)] for rows in (old, new)]
    rows, unchecked_rows = _label_matches(*labels)
    if not rows or not columns:
        return None
    return rows, columns, unchecked_rows, unchecked_columns


def _positions(old, new, all_captions):
    # Captions are evaluated across both full bodies before output truncation, so a repeated caption cannot bind a moved table.
    captions = [_unique([(caption, index) for index, caption in items.items()]) for items in all_captions]
    shared = captions[0].keys() & captions[1].keys()
    paired = [(captions[0][key], captions[1][key], "caption") for key in shared]
    old_matched, new_matched = ({pair[side] for pair in paired} for side in (0, 1))
    for index in (old.keys() & new.keys()) - old_matched - new_matched:
        if all(items[index] is None for items in all_captions):
            paired.append((index, index, "body_position"))
            old_matched.add(index)
            new_matched.add(index)
    paired.extend((index, None, None) for index in old.keys() - old_matched)
    paired.extend((None, index, None) for index in new.keys() - new_matched)
    return sorted(paired, key=lambda item: (item[0] if item[0] is not None else float("inf"),
                                            item[1] if item[1] is not None else float("inf")))


def _changes(old, new, rows, columns, next_id):
    changes = []
    for a, b in rows:
        for x, y in columns:
            first, second = old[a][x], new[b][y]
            for edit in differences(first.tokens, second.tokens):
                changes.append({"id": f"T{next_id + len(changes):02d}", "row": b + 1, "column": y + 1,
                                "earlier_row": a + 1, "earlier_column": x + 1,
                                "earlier": first.excerpt(edit["earlier"], edit["earlier_changed_tokens"]),
                                "later": second.excerpt(edit["later"], edit["later_changed_tokens"])})
    return changes


def compare(earlier, later):
    old_tables, new_tables = _tables(earlier), _tables(later)
    if not old_tables and not new_tables:
        return None
    captions = [{index: _caption(table) for index, (_, table) in items.items()} for items in (old_tables, new_tables)]
    positions = _positions(old_tables, new_tables, captions)
    old_budget = {"cells": MAX_CELLS, "tokens": MAX_TOKENS, "chars": MAX_CHARS}
    new_budget = dict(old_budget)
    compared, skipped, next_id = [], [], 1
    for old_index, new_index, match in positions[:MAX_TABLES_RETAINED]:
        old_number, old_xml = old_tables.get(old_index, (None, None))
        new_number, new_xml = new_tables.get(new_index, (None, None))
        identity = {"earlier_body_index": old_index, "later_body_index": new_index,
                    "earlier_table": old_number, "later_table": new_number}
        if match is None:
            reason = ("caption_unmatched" if any(value is not None for value in (captions[0].get(old_index), captions[1].get(new_index)))
                      else "no_matching_position")
        else:
            try:
                first, second = _table(old_xml, old_budget), _table(new_xml, new_budget)
            except ComparisonLimit:
                reason = "comparison_limit"
            else:
                if first is None or second is None:
                    reason = "unsupported_layout"
                elif (aligned := _alignment(first[0], second[0])) is None:
                    reason = "changed_structure"
                else:
                    old, old_budget = first
                    new, new_budget = second
                    rows, columns, unchecked_rows, unchecked_columns = aligned
                    changes = _changes(old, new, rows, columns, next_id)
                    compared.append({**identity, "match": match, "rows": len(new), "columns": len(new[0]),
                                     "earlier_rows": len(old), "earlier_columns": len(old[0]),
                                     "unchecked_rows": unchecked_rows, "unchecked_columns": unchecked_columns,
                                     "changes": changes})
                    next_id += len(changes)
                    continue
        skipped.append({**identity, "reason": reason})
    return {"coordinate_space": "docx_table_cell_unicode_codepoint", "compared": compared, "skipped": skipped,
            "omitted_positions": max(0, len(positions) - MAX_TABLES_RETAINED)}
