import json
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from email.message import EmailMessage

import docx
import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.text.extract import extract, from_string, text_hash
from app.evidence import document_repeat, document_scope
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
)
from app.reports import engine, multimedia_evidence, package, signing, snapshot, verify
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Artefact, ReportRow

PASSAGE = (
    "The review team checked the original receipts, inspected each signed record and matched every recorded date against "
    "the register before preparing the report for the meeting scheduled next Thursday morning."
)


def evidence_for(doc):
    document = {
        "version": document_scope.VERSION,
        "source": {
            "sha256": doc.meta.get("original_sha256", "a" * 64),
            "chars": len(doc.text),
            "extracted_text_sha256": text_hash(doc.text),
            "source_map_sha256": canonical_hash(doc.source_map),
        },
    }
    document["repeated_passages"] = document_repeat.build(document["source"], doc.text, doc.source_map)
    document_repeat.verify_source(document, doc.text, doc.source_map)
    return document["repeated_passages"]


def test_normal_packet_finds_repeat_without_any_classifier_highlight(packet_factory):
    alternate_whitespace = PASSAGE.replace("original receipts", "original  receipts")
    original = f"😀 Opening.\n\n{alternate_whitespace}\n\n{PASSAGE}"
    db, store, analysis, result, decoded, path = packet_factory(scores=(), text=original)
    evidence = build_packet_set(db, analysis, store, result)
    repeat = evidence["repeated_passages"]
    assert evidence["packets"] == []
    assert repeat["status"] == "matched" and repeat["authorship_confidence_contribution"] == 0
    assert repeat["normalization"] == "frozen_extraction_only"
    assert repeat["coverage"]["compared_blocks"] == 2
    assert repeat["coverage"]["excluded_short_or_long_blocks"] == 1
    (group,) = repeat["groups"]
    assert group["id"] == "RP01" and group["text"] == PASSAGE and group["total_occurrences"] == 2
    source = path.read_text()
    for item in group["occurrences"]:
        assert decoded["text"][item["start"] : item["end"]] == PASSAGE
        anchor = item["anchor"]
        assert " ".join(source[anchor["source_start"] : anchor["source_end"]].split()) == PASSAGE
    assert load_packet_set(db, analysis, store) == evidence

    # The false location is structurally possible, so the frozen extraction must reject it.
    changed = deepcopy(evidence)
    changed["repeated_passages"]["groups"][0]["occurrences"][0]["anchor"]["source_start"] -= 1
    changed["sha256"] = packet_hash(changed)
    document_repeat.validate(changed)
    db.get(Artefact, evidence["id"]).sha256 = store.put_bytes(canonical_json(changed)).sha256
    db.flush()
    with pytest.raises(DocumentPacketIntegrityError, match="frozen text or source mapping"):
        load_packet_set(db, analysis, store)


def repeated_pdf(path):
    with pymupdf.open() as pdf:
        for y in (50, 150):
            page = pdf.new_page()
            assert page.insert_textbox(pymupdf.Rect(50, y, 500, y + 150), PASSAGE, fontsize=10) > 0
            if y == 50:
                page.set_rotation(90)
        pdf.save(path)


def test_native_pdf_blocks_bind_pages_and_geometry_and_ocr_stays_unassessed(tmp_path):
    path = tmp_path / "repeat.pdf"
    repeated_pdf(path)
    doc = extract(path, ".pdf")
    repeat = evidence_for(doc)
    a, b = repeat["groups"][0]["occurrences"]
    assert [item["anchor"]["page"] for item in (a, b)] == [1, 2]
    assert [item["anchor"]["bbox"] for item in (a, b)] == [row["bbox"] for row in doc.source_map]
    assert a["anchor"]["coordinate_space"] == "unrotated-page-points"
    doc.source_map[1]["extraction"] = "ocr"
    repeat = evidence_for(doc)
    assert repeat["status"] == "not_assessed" and repeat["groups"] == []
    assert repeat["coverage"]["excluded_ocr_blocks"] == 1


def test_docx_repeats_map_to_body_and_table_locations(tmp_path):
    path = tmp_path / "repeat.docx"
    source = docx.Document()
    source.add_paragraph(PASSAGE)
    source.add_table(rows=1, cols=1).cell(0, 0).text = PASSAGE
    source.save(path)
    repeat = evidence_for(extract(path, ".docx"))
    a, b = [row["anchor"] for row in repeat["groups"][0]["occurrences"]]
    assert a == {"kind": "docx", "paragraph_index": 0, "body_path": [0]}
    assert b == {"kind": "docx", "paragraph_index": 1, "body_path": [1, "row", 0, "cell", 0, 1]}


def test_email_excludes_recognized_quotes_signatures_and_markup(tmp_path):
    path = tmp_path / "repeat.eml"
    email = EmailMessage()
    email.set_content(f"{PASSAGE}\n\n{PASSAGE}\n\n> {PASSAGE}\n\n--\n{PASSAGE}")
    path.write_bytes(email.as_bytes())
    repeat = evidence_for(extract(path, ".eml"))
    assert repeat["status"] == "matched"
    assert repeat["groups"][0]["total_occurrences"] == 2
    assert repeat["coverage"]["excluded_quoted_or_signature_blocks"] == 2
    assert all(item["anchor"]["mime_part"] == "1" for item in repeat["groups"][0]["occurrences"])

    markup = tmp_path / "repeat.html"
    markup.write_text(f"<p>{PASSAGE}</p><p>{PASSAGE}</p>")
    repeat = evidence_for(extract(markup, ".html"))
    assert repeat["status"] == "not_assessed" and repeat["groups"] == []
    assert repeat["coverage"]["excluded_other_blocks"] == 2


def test_requires_exact_text_and_accounts_for_bounded_display():
    repeat = evidence_for(from_string(f"{PASSAGE}\n\n{PASSAGE.lower()}"))
    assert repeat["status"] == "no_match" and repeat["groups"] == []
    blocks = [f"Draft {number}. {PASSAGE}" for number in range(6)]
    repeat = evidence_for(from_string("\n\n".join(blocks * 4)))
    assert repeat["coverage"]["compared_blocks"] == 24
    assert repeat["coverage"]["matching_groups"] == 6
    assert repeat["coverage"]["omitted_groups"] == 2
    assert [group["text"] for group in repeat["groups"]] == blocks[:4]
    assert all(
        group["total_occurrences"] == 4 and len(group["occurrences"]) == 3 for group in repeat["groups"]
    )


@pytest.mark.parametrize("change", ["coverage", "range", "source", "text", "score"])
def test_rejects_inconsistent_offline_repeat_records(packet_factory, change):
    db, store, analysis, result, _, _ = packet_factory(scores=(), text=f"Opening.\n\n{PASSAGE}\n\n{PASSAGE}")
    evidence = build_packet_set(db, analysis, store, result)
    repeat = evidence["repeated_passages"]
    if change == "coverage":
        repeat["coverage"]["compared_blocks"] += 1
    elif change == "range":
        repeat["groups"][0]["occurrences"][1]["start"] += 1
    elif change == "source":
        repeat["source_sha256"] = "b" * 64
    elif change == "text":
        repeat["groups"][0]["text"] += " Extra"
    else:
        repeat["authorship_confidence_contribution"] = 1
    with pytest.raises(ValueError):
        document_repeat.validate(evidence)


def test_native_pdf_repeat_offsets_match_the_rendered_report_and_signed_evidence(
    packet_factory, tmp_path, monkeypatch
):
    path = tmp_path / "repeat.pdf"
    repeated_pdf(path)
    db, store, analysis, source_result, decoded, _ = packet_factory(".pdf", scores=(), source_pdf=path)
    sample = specimen("document", "standard")
    result = deepcopy(sample.result)
    result["input"].update(sha256=analysis.sha256, filename=analysis.filename, size_bytes=analysis.size_bytes)
    result["manifest"].update(sampling=source_result["manifest"]["sampling"], detectors=[])
    result["signals"] = source_result["signals"]
    result["findings"] = []
    result["exhibits"] = []
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    evidence = build_packet_set(db, analysis, store, result)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    now = datetime.now(UTC)
    frozen = sample.model_copy(
        update={
            "result": result,
            "evidence": evidence,
            "case": None,
            "model_cards": [],
            "detector_names": {},
            "analysis": sample.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
            "media": sample.media.model_copy(
                update={
                    "sha256": analysis.sha256,
                    "filename": analysis.filename,
                    "size_bytes": analysis.size_bytes,
                    "content_type": analysis.content_type,
                }
            ),
            "report": sample.report.model_copy(
                update={"template_set": engine.TEMPLATE_SET, "requested_at": now}
            ),
        }
    )
    full = engine.render_pdf(frozen)
    redacted_snapshot = frozen.model_copy(
        update={"report": frozen.report.model_copy(update={"redacted": True})}
    )
    redacted = engine.render_pdf(redacted_snapshot)
    assert not full.refused and not redacted.refused
    full_text, redacted_text = [" ".join(engine.pdf_text(item.pdf).split()) for item in (full, redacted)]
    assert PASSAGE in full_text and PASSAGE not in redacted_text
    assert "verbatim excerpts are withheld" in redacted_text
    for text in (full_text, redacted_text):
        assert "Exact repeated passages" in text and "RP01" in text and "/repeated_passages" in text
        assert "PRIVATE MODEL NAME" not in text and "txt.desklib" not in text
        for row in evidence["repeated_passages"]["groups"][0]["occurrences"]:
            assert f"Page {row['anchor']['page']}, block {row['anchor']['block'] + 1}." in text
            assert f"Extracted characters [{row['start']}, {row['end']})." in text
            assert decoded["text"][row["start"] : row["end"]] == PASSAGE

    key = signing.instance_key(tmp_path / "repeat-signing.pem")
    pdf_art = register_artefact(
        db,
        store,
        full.pdf,
        media_type="application/pdf",
        role="report",
        created_by="repeat-qa",
        analysis_id=analysis.id,
    )
    snap_art = register_artefact(
        db,
        store,
        snapshot.dump(frozen),
        media_type="application/json",
        role="stage_output",
        created_by="report-snapshot",
        analysis_id=analysis.id,
    )
    report = ReportRow(
        id=frozen.report.id,
        display_id=frozen.report.display_id,
        revision=frozen.report.revision,
        analysis_id=analysis.id,
        analysis_revision=analysis.revision,
        profile="standard",
        redacted=False,
        review_state="machine",
        issuable=False,
        template_set=engine.TEMPLATE_SET,
        engine="test",
        status="done",
        snapshot_hash=snapshot.snapshot_hash(frozen),
        snapshot_artefact_id=snap_art.id,
        artefact_id=pdf_art.id,
        sha256=pdf_art.sha256,
        signature=key.sign(full.pdf),
        signing_key_id=key.key_id,
        requested_by="test",
        requested_at=now,
        page_count=full.page_count,
    )
    db.add(report)
    row = package.prepare(
        db,
        analysis,
        report,
        kind="draft",
        include_original=False,
        created_by="test",
        schema_id=package.PUBLIC_SCHEMA,
    )
    bundle = package.build(db, row, store=store, key=key, exported_at=now)
    checked = verify.verify_package(bundle.path, public_key_pem=key.public_pem())
    assert checked.ok, checked.as_dict()
    with zipfile.ZipFile(bundle.path) as archive:
        assert archive.read("report.pdf") == full.pdf
        public = json.loads(archive.read("evidence.json"))
        assert public == evidence and public["repeated_passages"]["groups"][0]["text"] == PASSAGE
        for name in ("evidence.json", "result.json", "bundle.json"):
            assert b"PRIVATE MODEL NAME" not in archive.read(name) and b"txt.desklib" not in archive.read(
                name
            )
