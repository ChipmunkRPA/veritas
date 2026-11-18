# SPDX-License-Identifier: Apache-2.0
"""Verify date evidence through local checkout, public exports and original replay."""

from __future__ import annotations

import argparse
import json
import os
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_native_alignment_panel import digest, save
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def deliveries(examples, folder):
    import docx
    import pymupdf

    plain = examples / "calendar-mismatch.txt"
    original = plain.read_text(encoding="utf-8")
    yield "plain", plain, "text/plain"
    word = folder / "word.docx"
    document = docx.Document()
    for paragraph in original.split("\n\n"):
        document.add_paragraph(paragraph)
    document.save(word)
    yield "word", word, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    email = folder / "email.eml"
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = "editor@example.org", "calendar@example.org", "Meeting dates"
    message.set_content(original + "\n\n> Earlier reply said Tuesday, September 24, 2018.\n")
    email.write_bytes(message.as_bytes())
    yield "email", email, "message/rfc822"
    pdf = folder / "submitted.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        assert page.insert_textbox((60, 65, 540, 400), original, fontsize=11) >= 0
        document.save(pdf)
    yield "pdf", pdf, "application/pdf"
    yield "ordinary", examples / "untouched.txt", "text/plain"


def report(client, route, profile):
    row = body(client.post(f"{route}/reports", json={"profile": profile}))
    wait(client, f"/api/v2/reports/{row['id']}")
    return row["id"], fetch(client, f"/api/v2/reports/{row['id']}/file")


def run(examples, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_PUBLIC_PDF_LAYOUT="standard", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_MODELS_DIR=str(local_model_view(output)),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.text.extract import extract
    from app.evidence import document_calendar as check
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    runs = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("calendar-local-trial", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, path, mime in deliveries(examples, output):
            doc = extract(path, path.suffix)
            expected = check.build({"method": doc.meta["method"]}, doc.text, doc.source_map)
            assert expected["checked_dates"] >= 1
            assert expected["disagreements"] == int(name != "ordinary"), (name, expected)
            aid = checkout(client, stub, path, kind="document", media_type=mime, filename=path.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            assert evidence[check.KEY] == expected and not expected["affects_overall_confidence"]
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for blob in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(blob.decode())
            report_id, unredacted = report(client, route, "standard")
            words = " ".join(pdf_text(unredacted).split())
            scan(words)
            assert "Calendar date check" in words
            for index, item in enumerate(expected["locations"]):
                assert " ".join(item["text"].split()) in words
                assert f"fell on {item['expected_weekday']}" in words
                assert f"Extracted characters [{item['start']}, {item['end']})" in words
                assert f"/calendar_consistency/locations/{index}" in words
                assert doc.text[item["start"]:item["end"]] == item["text"]
            (output / f"{name}-report.pdf").write_bytes(unredacted)
            package_row = body(client.post(f"{route}/packages", json={"report_id": report_id, "kind": "draft", "include_original": False}))
            wait(client, f"/api/v2/packages/{package_row['id']}")
            bundle = output / f"{name}-package.zip"
            bundle.write_bytes(fetch(client, f"/api/v2/packages/{package_row['id']}/file"))
            replay = verify_package(bundle, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                                    source_document=path)
            assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
            assert any("Verified the calendar date check and its source positions" in item.detail for item in replay.checks)
            with ZipFile(bundle) as archive:
                assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == unredacted
            if name == "pdf":
                _, redacted = report(client, route, "newsroom")
                redacted_words = " ".join(pdf_text(redacted).split())
                assert "Calendar date check" in redacted_words
                assert all(" ".join(item["text"].split()) not in redacted_words and item["iso_date"] not in redacted_words
                           for item in expected["locations"])
                (output / "pdf-newsroom-report.pdf").write_bytes(redacted)
            runs[name] = {"input_sha256": digest(path), "calendar": expected, "signed_source_verified": True,
                          "redacted_calendar_phrase_checked": name == "pdf"}
            print("Verified", name, flush=True)
    save(output / "receipt.json", {"schema": "chipmunk.document-calendar-local-checkout/1", "cases": runs,
                                   "usd_cents": 999, "jobs_before_payment": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
