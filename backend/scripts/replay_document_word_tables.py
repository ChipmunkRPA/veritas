"""Verify Word table change localization against retained, frozen writing pairs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts import writing_uswds_revision_transfer as study

SCHEMA = "chipmunk.document-word-table-comparison/1"


def document(text, *, bold=False):
    from docx import Document

    doc = Document()
    doc.core_properties.created = doc.core_properties.modified = datetime(2000, 1, 1, tzinfo=UTC)
    doc.add_paragraph("Public guidance revision excerpt.")
    table = doc.add_table(rows=3, cols=2)
    for row, values in enumerate((("Guidance item", "Wording"), ("Review passage", text),
                                  ("Unchanged example", "Keep the documented wording."))):
        for column, value in enumerate(values):
            table.cell(row, column).paragraphs[0].add_run(value).bold = bold
    saved = BytesIO()
    doc.save(saved)
    canonical = BytesIO()
    with ZipFile(saved) as original, ZipFile(canonical, "w", compression=ZIP_DEFLATED) as target:
        for name in sorted(original.namelist()):
            target.writestr(ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0)), original.read(name), compress_type=ZIP_DEFLATED)
    return canonical.getvalue()


def verify_rows(run, panel):
    import pymupdf

    from app.evidence import document_revision
    from app.reports.document_revision_pdf import render
    from scripts.review_document_revision_export import review

    path = Path(run)
    cases = study.cases(path)
    reference = {case["group"]: case for case in cases if case["family"] == "before"}
    upstream = study.read(panel)
    expected = {(row["group"], row["family"]): row for row in upstream["rows"]}
    study.require(len(cases) == len(expected) == 21, "The matched upstream roster changed")
    output = []
    for row in cases:
        group, family = row["group"], row["family"]
        before = reference[group]["text"]
        after = row["text"]
        study.require(study.sha(after) == expected[group, family]["text_sha256"], "The frozen generated or original text changed")
        earlier = document(before)
        later = document(after, bold=True)
        legacy, *_ = document_revision.compare_with_sources(earlier, later, ".docx", include_word_tables=False)
        found, *_ = document_revision.compare_with_sources(earlier, later, ".docx", strict_word_tables=True)
        plain = document_revision.compare(before.encode(), after.encode(), ".txt")
        study.require(not legacy["changes"] and not found["changes"] and not found["word_tables"]["skipped"],
                      "Source text escaped its matched table")
        [table] = found["word_tables"]["compared"]
        study.require(len(table["changes"]) == len(plain["changes"]), "A table wording change was missed")
        for observed, control in zip(table["changes"], plain["changes"], strict=True):
            study.require((observed["row"], observed["column"]) == (2, 2), "Wrong native cell")
            for side, original in (("earlier", before), ("later", after)):
                excerpt = observed[side]
                study.require(excerpt["changed_tokens"] == control[side]["changed_tokens"]
                              and excerpt["text"] == original[excerpt["start"]:excerpt["end"]], "Exact word positions differ")
        files = path / f"word-table-{group}-{family}"
        files.mkdir(exist_ok=False)
        old_path, new_path, export = (files / name for name in ("earlier.docx", "analyzed.docx", "comparison.json"))
        old_path.write_bytes(earlier)
        new_path.write_bytes(later)
        study.write(export, found)
        verified = review(export, old_path, new_path)
        pdf_verified = None
        if table["changes"] and not any(item["pdf_coordinates_verified"] for item in output):
            pdf = render(found)
            with pymupdf.open(stream=pdf, filetype="pdf") as report:
                pdf_text = " ".join(word for page in report for word in page.get_text().split())
            study.require("Word table wording changes" in pdf_text and "Row 2, column 2" in pdf_text
                          and all(change["id"] in pdf_text for change in table["changes"]), "The PDF lost native cell coordinates")
            (files / "comparison.pdf").write_bytes(pdf)
            pdf_verified = True
        output.append({"group": group, "family": family, "source_sha256": [study.sha(earlier), study.sha(later)],
                       "comparison_sha256": study.digest(export), "legacy_changes": 0,
                       "table_changes": len(table["changes"]), "exact_token_ranges": verified["word_cell_token_ranges_verified"],
                       "pdf_coordinates_verified": pdf_verified})
    return {"schema": SCHEMA, "parent_sha256": study.digest(panel), "rows": output, "summary": summarize(output)}


def summarize(rows):
    totals = {}
    for row in rows:
        stats = totals.setdefault(row["family"], Counter())
        stats.update(pairs=1, legacy_located=int(row["legacy_changes"] > 0), table_located=int(row["table_changes"] > 0),
                     exact_token_ranges=row["exact_token_ranges"], pdf_checked=int(row["pdf_coordinates_verified"] is True))
    return totals


def replay(receipt, panel):
    ledger, parent = study.read(receipt), study.read(panel)
    study.require(ledger["schema"] == SCHEMA and ledger["parent_sha256"] == study.digest(panel), "Wrong upstream writing ledger")
    rows = ledger["rows"]
    study.require(len(rows) == 21 and {(row["group"], row["family"]) for row in rows}
                  == {(row["group"], row["family"]) for row in parent["rows"]}, "Wrong matched source groups")
    study.require(all(len(row["source_sha256"]) == 2 and row["legacy_changes"] == 0
                      and (row["table_changes"] > 0) == (row["exact_token_ranges"] > 0) == (row["family"] != "before")
                      for row in rows), "A retained cell observation differs")
    result = summarize(rows)
    study.require(result == ledger["summary"], "Recorded aggregates differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path, help="Scratch directory from the USWDS writing study for native scoring")
    args = parser.parse_args()
    if args.phase == "score":
        if args.run is None:
            parser.error("--run is required for native scoring")
        study.write(args.receipt, verify_rows(args.run, args.panel))
    print(json.dumps(replay(args.receipt, args.panel), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
