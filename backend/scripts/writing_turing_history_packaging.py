# SPDX-License-Identifier: Apache-2.0
"""Repack historical writing inputs as Word and selectable PDF and check native locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence.document_interval_locations import checked_end, locate
from scripts.writing_transition_deviation import file_hash, read, text_hash

FONT = Path(__file__).resolve().parents[1] / "app/reports/fonts/Inter-400-latin.ttf"


def replay(plan, output):
    import pymupdf
    from docx import Document

    works = read(plan)["works"][:2]
    if len(works) != 2:
        raise ValueError("Use the historical source plan with at least two works")
    output.mkdir(mode=0o700)
    result = []
    for number, work in enumerate(works, 1):
        for row, witness in zip(work["variants"], work["witnesses"], strict=True):
            title = f"work-{number}-{row['version']}"
            source = row["text"]
            paragraphs = source.split("\n\n")
            word = Document()
            for paragraph in paragraphs:
                word.add_paragraph(paragraph)
            word.save(output / f"{title}.docx")
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=612, height=1008)
                y, bottom = 45, 963
                for paragraph in paragraphs:
                    spare = page.insert_textbox(pymupdf.Rect(42, y, 570, bottom), paragraph, fontname="FixtureInter",
                                                fontfile=str(FONT), fontsize=10.5, lineheight=1.25)
                    if spare < 0:
                        raise ValueError("The selectable PDF source text does not fit the page")
                    y = bottom - spare + 14
                pdf.save(output / f"{title}.pdf", no_new_id=True)
            for suffix in (".docx", ".pdf"):
                path = output / (title + suffix)
                doc = extract(path, suffix, allow_ocr=False)
                actual = segment(doc)
                if (doc.text != source or [[unit.start, unit.end] for unit in actual] != [r[:2] for r in row["sentences"]]
                        or checked_end(doc.meta["method"], doc.text, doc.source_map, doc.meta, doc.word_review) != len(doc.text)):
                    raise ValueError("A delivery changed exact text, sentence geometry or extractable coverage")
                start, end = witness["focus"]
                places = locate(doc.meta["method"], doc.source_map, start, end)
                if not places or min(row["start"] for row in places) != start or max(row["end"] for row in places) != end:
                    raise ValueError("The historical edit failed to map to its original source positions")
                result.append({"case": title, "format": suffix[1:], "source_file_sha256": file_hash(path),
                               "text_sha256": text_hash(doc.text), "exact_extracted_text": True,
                               "matched_all_non_whitespace": True, "target_codepoints": [start, end],
                               "source_locations": places})
    receipt = {"source_plan_sha256": file_hash(plan), "cases": result}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Verified native source coordinates for all {len(result)} Word and PDF deliveries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    replay(**vars(parser.parse_args()))
