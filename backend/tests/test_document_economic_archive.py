from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from docx import Document

from app.core.canonical import canonical_hash
from app.core.config import settings
from app.detectors.text.extract import extract
from app.engine.stages import document as stage
from app.evidence import document_economic_archive as archive
from app.evidence import document_published_wording as wording

BODY = ("The regional research office collected its original records from local community groups and public meeting minutes. "
        "The board recorded the new release on Monday after a separate discussion about the final review. "
        "The regional archive kept a printed record for all interested readers and circulated the approved figures to its libraries. "
        "Several local groups kept copies with their own meeting notes for the next round of public discussion and feedback.")
TITLE = "Review of the regional economic programme"
URL = "https://www.bea.gov/news/2018/regional-economic-programme"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_pinned_release_requires_title_source_and_native_word_replay(tmp_path, monkeypatch):
    original = tmp_path / "release.docx"
    word = Document()
    for paragraph in (TITLE, f"Source: {URL}", BODY.replace("Monday", "Thursday")):
        word.add_paragraph(paragraph)
    word.save(original)
    doc = extract(original, ".docx")
    reference = {"id": digest(URL.encode()), "publisher": "bea", "title": TITLE, "source_url": URL,
                 "published_date": "2018-06-25", "source_record_sha256": "a" * 64,
                 "transcript_sha256": digest(BODY.encode()), "text": BODY}
    path = tmp_path / "release-pack.json"
    path.write_text(json.dumps({"schema": archive.SCHEMA, "references": [reference]}))
    pack = digest(path.read_bytes())
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "document_economic_archive_path", path)
    monkeypatch.setattr(settings, "document_economic_archive_sha256", pack)

    ctx = SimpleNamespace(outputs={"ingest": {"filename": original.name}}, log=SimpleNamespace(warning=lambda *_args, **_kwargs: None))
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "source_map": doc.source_map, "meta": doc.meta}
    record = stage._published_wording(ctx, decoded, None)
    assert record["schema"] == wording.RELEASE_SCHEMA
    assert record["authorship_confidence_contribution"] == 0
    assert record["reference"] == reference
    assert wording.report_view(record, "python-docx")["sections"][0]["source_url"] == URL
    [change] = record["changes"]
    [sentence] = record["changed_sentences"]
    assert change["submitted"] == "Thursday"
    assert sentence["locations"][0]["paragraph"] == 3
    start, end = change["archive_range"]
    assert BODY[start:end] == "Monday"

    def check(value):
        evidence = {"source": {"method": "python-docx", "chars": len(doc.text), wording.COMMITMENT: canonical_hash(value)},
                    "coverage": {wording.KEY: value}}
        wording.verify_source(evidence, doc.text, doc.source_map, doc.meta, doc.word_review, paragraphs=doc.paragraphs)

    check(record)
    changed = deepcopy(record)
    changed["changes"][0]["locations"][0]["paragraph"] = 1
    with pytest.raises(ValueError, match="locations differ"):
        check(changed)
    changed = deepcopy(record)
    changed["reference"]["source_url"] = "https://www.bea.gov.example.invalid/"
    with pytest.raises(ValueError, match="source identity"):
        check(changed)
    assert not archive.select(doc.text.replace(TITLE, "A different programme"), {URL: reference})
    assert not archive.select(doc.text.replace(URL, URL + "?ref=tracking"), {URL: reference})
    monkeypatch.setattr(settings, "document_economic_archive_sha256", "b" * 64)
    assert stage._published_wording(ctx, decoded, None) is None


def test_release_diff_rejects_long_repeated_uploaded_or_archived_ranges(monkeypatch):
    title_header = f"{TITLE}\n\nSource: {URL}\n\n"
    left, right = BODY.split(" The board ", 1)
    archived = {"title": TITLE, "source_url": URL, "text": BODY}

    def diff_must_not_run(*_args, **_kwargs):
        pytest.fail("A comparison outside the size limits must stop before alignment")

    monkeypatch.setattr(archive, "SequenceMatcher", diff_must_not_run)
    repeated = ("repeat " * 850).strip()
    assert archive.align(title_header + left + " " + repeated + " The board " + right, archived) is None
    assert archive.align(title_header + BODY, archived | {"text": left + " " + repeated + " The board " + right}) is None
    assert archive.align(title_header + "a" * 20_001, archived) is None
