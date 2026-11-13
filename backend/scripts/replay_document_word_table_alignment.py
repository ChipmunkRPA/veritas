"""Rebuild the Word table alignment comparison from pinned public CBO CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts import writing_uswds_revision_transfer as source

SCHEMA = "chipmunk.word-table-alignment-cbo/1"


def _observations(root, entry, keys):
    data = (root / entry["path"]).read_bytes()
    source.require(source.sha(data) == entry["sha256"], "CBO publisher bytes differ")
    result = {}
    for record in csv.DictReader(StringIO(data.decode("utf-8-sig"))):
        key = record["date"], record["variable"], record.get("estimate_type", "")
        if record["value"] and source.sha("|".join(key)) in keys:
            label = " | ".join(part for part in key if part)
            source.require(label not in result, "Source label is not unique")
            result[label] = record["value"]
    source.require(len(result) == len(keys), "Source rows were not recovered")
    return result


def _common_keys(root, sources):
    groups = []
    for entry in sources:
        data = (root / entry["path"]).read_bytes()
        values = {}
        for row in csv.DictReader(StringIO(data.decode("utf-8-sig"))):
            if row["value"]:
                key = source.sha("|".join((row["date"], row["variable"], row.get("estimate_type", ""))))
                values.setdefault(key, set()).add(row["value"])
        groups.append({key for key, readings in values.items() if len(readings) == 1})
    return sorted(groups[0] & groups[1])[:8]


def _word(rows, caption, *, moved=False, reorder=False, duplicate=False):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    doc.add_paragraph("Congressional Budget Office archived data release.")
    if moved:
        doc.add_paragraph("An explanatory note added before this release.")
    data = [["Period and measure", "Published value", "Unit label"],
            *[[key, value, "Publisher estimate"] for key, value in rows]]
    if duplicate:
        data[-1][0] = data[-2][0]
        data[-1][1] = "-999999"
    table = doc.add_table(rows=len(data), cols=3)
    node = OxmlElement("w:tblCaption")
    node.set(qn("w:val"), caption)
    table._tbl.tblPr.append(node)
    for i, row in enumerate(data):
        for j in range(3):
            table.cell(i, j).text = row[(0, 2, 1)[j] if reorder else j]
    encoded = BytesIO()
    doc.save(encoded)
    frozen = BytesIO()
    with ZipFile(encoded) as archive, ZipFile(frozen, "w") as output:
        for name in sorted(archive.namelist()):
            output.writestr(ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0)), archive.read(name), compress_type=ZIP_DEFLATED)
    return frozen.getvalue()


def _numbers(earlier, later, rows):
    return {(earlier.index(key) + 2, 2, later.index(key) + 2, 3): (a, b) for key, a, b in rows}


def _actual(coverage, before_values, after_values):
    [entry] = coverage["compared"]
    source.require(not coverage["skipped"] and entry["match"] == "caption"
                   and (entry["earlier_body_index"], entry["later_body_index"]) == (1, 2), "Native table alignment differs")
    observed = {}
    for change in entry["changes"]:
        old_row, new_row = change["earlier_row"], change["row"]
        coordinate = old_row, change["earlier_column"], new_row, change["column"]
        old_value, new_value = before_values[old_row - 2], after_values[new_row - 2]
        for side, value in (("earlier", old_value), ("later", new_value)):
            excerpt = change[side]
            source.require(excerpt["text"] == value[excerpt["start"]:excerpt["end"]]
                           and all(excerpt["start"] <= a < b <= excerpt["end"] for a, b in excerpt["changed_tokens"]), "Cell source range differs")
        observed[coordinate] = old_value, new_value
    return observed, entry


def score(roster, root, output):
    import pymupdf

    from app.evidence import document_revision
    from app.reports.document_revision_pdf import render
    from scripts.review_document_revision_export import review

    intake = source.read(roster)
    source.require(source.digest(root / "LICENSE.md") == intake["license_sha256"], "Pinned license differs")
    output.mkdir()
    result = []
    for i, group in enumerate(intake["pairs"], start=1):
        keys = set(group["key_sha256"])
        left, right = [_observations(root, item, keys) for item in group["sources"]]
        source.require(_common_keys(root, group["sources"]) == group["key_sha256"], "The source key freeze differs")
        base = sorted(left)
        revised = [*reversed(base[1:]), "A newly added row"]
        original = _word([(key, left[key]) for key in base], group["name"])
        target = _word([(key, right.get(key, "0")) for key in revised], group["name"], moved=True, reorder=True)
        old, *_ = document_revision.compare_with_sources(original, target, ".docx", strict_word_tables=True)
        source.require(not old["word_tables"]["compared"], "The old comparison unexpectedly checked the table")
        match = document_revision.compare(original, target, ".docx")
        observed, entry = _actual(match["word_tables"], [left[key] for key in base], [right.get(key, "0") for key in revised])
        expected = _numbers(base, revised, [(key, left[key], right[key]) for key in base[1:] if left[key] != right[key]])
        source.require(observed == expected and entry["unchecked_rows"] == {"earlier": 1, "later": 1}, "Revised cells differ from publisher values")
        controls = []
        for duplicate in (False, True):
            control = _word([(key, left[key]) for key in reversed(base)], group["name"], moved=True, reorder=True, duplicate=duplicate)
            [coverage] = document_revision.compare(original, control, ".docx")["word_tables"]["compared"]
            source.require(not coverage["changes"] and coverage["unchecked_rows"] ==
                           ({"earlier": 2, "later": 2} if duplicate else {"earlier": 0, "later": 0}), "A control row was misattributed")
            controls.append(source.sha(control))
        case = output / f"group-{i}"
        case.mkdir()
        a, b, export = (case / name for name in ("earlier.docx", "analyzed.docx", "comparison.json"))
        a.write_bytes(original)
        b.write_bytes(target)
        source.write(export, match)
        verified = review(export, a, b)
        pdf_verified = bool(observed) and not any(item["pdf_verified"] for item in result)
        if pdf_verified:
            pdf = render(match)
            with pymupdf.open(stream=pdf, filetype="pdf") as report:
                shown = " ".join(word for page in report for word in page.get_text().split())
            source.require("Earlier table 1 · Row" in shown and "Analyzed table 1 · Row" in shown
                           and "Unchecked rows: 1 earlier, 1 analyzed." in shown, "The PDF lost its matched coordinates")
            (case / "comparison.pdf").write_bytes(pdf)
        result.append({"name": group["name"], "changed_cells": len(observed), "strict_changed_cells": 0,
                       "false_reordered": 0, "false_duplicate": 0, "coordinates": [list(key) for key in sorted(observed)],
                       "source_sha256": [source.sha(original), source.sha(target)], "control_sha256": controls,
                       "export_sha256": source.digest(export), "token_ranges": verified["word_cell_token_ranges_verified"],
                       "pdf_verified": pdf_verified})
    return {"schema": SCHEMA, "roster_sha256": source.digest(roster), "rows": result, "summary": _summarize(result)}


def _summarize(rows):
    return dict(Counter(pairs=len(rows), pairs_with_changes=sum(row["changed_cells"] > 0 for row in rows),
                        **{key: sum(row[key] for row in rows) for key in ("changed_cells", "strict_changed_cells", "false_reordered",
                                                                        "false_duplicate", "token_ranges", "pdf_verified")}))


def review_receipt(receipt, roster):
    panel, plan = source.read(receipt), source.read(roster)
    source.require(panel["schema"] == SCHEMA and panel["roster_sha256"] == source.digest(roster), "Wrong source roster")
    rows = panel["rows"]
    source.require([row["name"] for row in rows] == [row["name"] for row in plan["pairs"]] and len(rows) == 6, "Missing source pair")
    source.require(all(len(row["coordinates"]) == len({tuple(p) for p in row["coordinates"]}) == row["changed_cells"]
                   and all(2 <= a <= 9 and x == 2 and 2 <= b <= 8 and y == 3 for a, x, b, y in row["coordinates"])
                   for row in rows), "Incorrect source coordinates")
    summary = _summarize(rows)
    source.require(summary == panel["summary"], "Recorded arithmetic differs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "review"))
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        source.write(args.receipt, score(args.roster, args.source, args.output))
    print(json.dumps(review_receipt(args.receipt, args.roster), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
