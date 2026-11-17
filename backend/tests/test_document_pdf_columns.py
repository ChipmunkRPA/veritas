import hashlib
from copy import deepcopy
from textwrap import wrap

import pymupdf
import pytest

from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.evidence import document_gao_archive as archive
from app.evidence import document_published_wording as wording

PARAGRAPHS = (
    "Auditors collected planning records from the regional office and documented how the committee established its financial priorities. "
    "The original worksheet identified each program and assigned its review to a separate department.",
    "During the spring review, staff interviewed the officers responsible for the water and transportation programs. The board "
    "received the completed reconciliation on Monday and discussed the remaining questions with the budget administrator.",
    "The working group prepared an inventory of current practices. It compared operating procedures against written policies and "
    "asked administrators to explain how they tracked exceptions throughout the year.",
    "Several offices supplied evidence that confirmed the recommendations had been implemented on schedule. The auditors retained "
    "the supporting calculations and documented their agreement with the published guidance.",
    "A later assessment examined the procurement register. It reviewed compliance with contract requirements and compared findings "
    "with the administrative notices previously delivered to participants.",
    "The project manager summarized the decisions in an approval memo and shared them with the inspection committee. This memo "
    "listed the affected locations and described the follow-up procedure for each department.",
    "The final evaluation distinguishes completed actions from recommendations still open at the end of the reporting period. "
    "It also identifies which outstanding matters need a new review during the winter.",
    "The agency subsequently published its observations and invited the working group to check technical details. Their responses "
    "appear in the public summary along with instructions for requesting additional records.",
)


def _render_pdf(path, *, cover=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=612, height=792)
        for index, words in enumerate(PARAGRAPHS):
            x = 42 if index < 4 else 327
            y = 56 + (index % 4) * 112 + (5 if x > 42 else 0)
            pages = "\n".join(wrap(words.replace("Monday", "Thursday"), 43, break_long_words=False))
            page.insert_text((x, y), pages, fontname="cour", fontsize=9, lineheight=1.29)
        if cover:
            page.draw_rect(pymupdf.Rect(40, 153, 306, 238), fill=(1, 1, 1), color=None, overlay=True)
        pdf.save(path)


def test_column_comparison_binds_native_positions_and_replays_without_flags(tmp_path):
    path = tmp_path / "original.pdf"
    _render_pdf(path)
    doc = extract(path, ".pdf", allow_ocr=False)
    meta = doc.meta | {"original_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not wording.eligible(".pdf", doc.text, doc.source_map, meta, None)
    assert wording.eligible(".pdf", doc.text, doc.source_map, meta, None, pdf_columns=True)
    reports = {"GAO-16-442": {"text": "\n\n".join(PARAGRAPHS), "published_date": "2016-04-18", "split": "train",
                              "source_record_sha256": "1" * 64}}
    reading = wording.capture(doc.text, doc.paragraphs, doc.source_map, meta, None, "2" * 64, reports,
                              archive.build_index(reports), pdf_columns=True)
    [section] = reading["sections"]
    assert section["column"] == "both" and section["page"] == 1
    assert reading["authorship_confidence_contribution"] == 0
    [change] = section["changes"]
    [place] = change["locations"]
    assert change["submitted"] == doc.text[place["start"]:place["end"]] == "Thursday"
    assert place["page"] == 1 and len(place["boxes"]) == 1 and place["boxes"][0][0] < 327
    assert place["start"] != change["start"]  # The raw PDF extraction interleaves the columns.

    def evidence(record):
        return {"coverage": {wording.KEY: record}, "source": {"method": "pymupdf", "chars": len(doc.text),
                                                               wording.COMMITMENT: canonical_hash(record)}}

    wording.verify_source(evidence(reading), doc.text, doc.source_map, meta, None)
    moved = deepcopy(reading)
    moved["sections"][0]["changes"][0]["locations"][0]["boxes"][0][1] += 1.
    with pytest.raises(ValueError, match="original locations"):
        wording.verify_source(evidence(moved), doc.text, doc.source_map, meta, None)


def test_a_solid_cover_keeps_a_two_column_page_outside_the_comparison(tmp_path):
    path = tmp_path / "covered.pdf"
    _render_pdf(path, cover=True)
    doc = extract(path, ".pdf", allow_ocr=False)
    assert not wording.eligible(".pdf", doc.text, doc.source_map, doc.meta, None, pdf_columns=True)
