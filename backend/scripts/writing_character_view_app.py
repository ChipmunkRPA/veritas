# SPDX-License-Identifier: MIT
"""Exercise character-adjusted writing evidence through checkout and signed exports."""

from __future__ import annotations

import argparse
import html
import json
import os
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts import writing_character_view_transfer as study

ROOT = Path(__file__).resolve().parents[2]
TEXT = (
    "The regional committee collected the financial records before reviewing the proposal at its regular meeting. "
    "The treasurer examined the ordinary receipts and compared the remaining information against the original invoices. "
    "Several submitted records were discussed in the afternoon while the committee prepared the revised annual budget. "
    "Members returned the documents to the office and agreed to publish the minutes the following week."
)
MIME = {".txt": "text/plain", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".eml": "message/rfc822", ".pdf": "application/pdf"}


def fixture(extension, text, font):
    if extension == ".txt":
        return text.encode()
    if extension == ".docx":
        from scripts.rebuild_writing_word_boundary_sources import docx

        return docx(text)
    if extension == ".eml":
        message = EmailMessage()
        message["Subject"] = "Minutes of the regular meeting"
        message.set_content(f"<html><body><p>{html.escape(text).encode('ascii', 'xmlcharrefreplace').decode()}</p></body></html>",
                            subtype="html", cte="quoted-printable")
        return message.as_bytes()

    import pymupdf

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_font(fontname="TestText", fontfile=str(font))
        assert page.insert_textbox((54, 60, 558, 740), text, fontname="TestText", fontsize=10) >= 0
        return document.tobytes(deflate=True, no_new_id=True)


def tokens_assessed(aid):
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store

    with get_session() as database:
        current = database.get(Analysis, aid)
        result = json.loads(default_store().read_bytes(current.result_hash))
        return result["decision"], result.get("customer_assessment")


def verify_deliveries(client, aid, source, evidence, directory):
    import pymupdf
    from app.evidence import document_character_review as proof
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    report = paid.body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    paid.wait(client, f"/api/v2/reports/{report['id']}")
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    (directory / "report.pdf").write_bytes(pdf)
    target = evidence["coverage"][proof.KEY]
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = " ".join(word for page in document for word in page.get_text().split())
        scan(text)
        assert "Writing response after character adjustment" in text
        for row in target["ranges"]:
            assert f"Extracted passage [{row['spans'][1]['start']}, {row['spans'][1]['end']})" in text
            assert proof.STATE[row["state"]] in text and all(change["codepoint"] in text for change in row["changes"])
    package = paid.body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False}))
    paid.wait(client, f"/api/v2/packages/{package['id']}")
    archive = directory / "evidence.zip"
    archive.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    replay = verify_package(archive, public_key_pem=public, source_document=source)
    assert replay.ok, replay.as_dict()
    assert any("Verified the character-adjusted writing input hashes" in check.detail for check in replay.checks)
    with ZipFile(archive) as packaged:
        assert json.loads(packaged.read("evidence.json"))["coverage"][proof.KEY] == target
        scan(packaged.read("result.json").decode())
    return {"original_verified": True, "pdf_sha256": study.sha(pdf), "json_sha256": evidence["sha256"]}


def run(output, font):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true",
                      VERITAS_TEXT_CHARACTER_ADJUSTED_REVIEW="true", VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.text.extract import extract
    from app.evidence import document_character_review as proof
    from app.main import app
    from app.storage.db import get_session
    from fastapi.testclient import TestClient
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    start = TEXT.index("The treasurer")
    end = TEXT.index(" Several")
    results = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-writing-character-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for ext, media_type in MIME.items():
            directory = output / ext[1:]
            directory.mkdir(mode=0o700)
            altered, marks = study.perturb(TEXT[start:end], "lookalikes" if ext == ".pdf" else "both")
            source = directory / ("authored" + ext)
            source.write_bytes(fixture(ext, TEXT[:start] + altered + TEXT[end:], font))
            doc = extract(source, ext, allow_ocr=False)
            assert altered in doc.text
            aid = paid.checkout(client, stub, source, kind="document", media_type=media_type, filename=source.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            review = evidence["coverage"][proof.KEY]
            assert review["authorship_confidence_contribution"] == 0
            row, = review["ranges"]
            assert row["excerpt"] == altered and len(row["changes"]) == marks
            assert row["location"]["kind"] == {".txt": "text", ".docx": "docx", ".eml": "email", ".pdf": "pdf"}[ext]
            proof.verify_source(evidence, doc.text, doc.source_map, doc.meta)
            checks = verify_deliveries(client, aid, source, evidence, directory)
            results.append({"format": ext, "state": row["state"], "focus": row["spans"][1], "location": row["location"],
                            "points": row["changes"], **checks})
            if ext == ".txt":
                settings.text_character_adjusted_review = False
                previous = paid.checkout(client, stub, source, kind="document", media_type=media_type, filename=source.name)
                before = paid.body(client.get(f"/api/v2/analyses/{previous}/evidence-packets"))
                assert proof.KEY not in before["coverage"] and tokens_assessed(aid) == tokens_assessed(previous)
                settings.text_character_adjusted_review = True
            print(ext, row["state"], row["spans"][1], flush=True)
    (output / "receipt.json").write_text(json.dumps({"schema": "chipmunk.writing-character-view-app/1", "paid_analyses": len(stub.sessions),
                                                     "price_usd": 9.99, "unpaid_analysis_jobs": 0,
                                                     "baseline_assessment_agrees": True, "results": results}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pdf-font", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.pdf_font)


if __name__ == "__main__":
    main()
