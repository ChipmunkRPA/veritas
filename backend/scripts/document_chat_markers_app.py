# SPDX-License-Identifier: Apache-2.0
"""Replay chat-format markers through a local paid document analysis."""

import argparse
import json
import os
from email.message import EmailMessage
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
OPENER, CLOSER = "<|im_start|>assistant", "<|im_end|>"
TEXT = ("Please confirm the opening time for the community room and bring a printed copy of the inventory. "
        "The side entrance is normally open from nine in the morning. The office can print the final list if needed. "
        "There are two wooden boxes on the third shelf near the window. "
        "The cart can stay in the storage area until the next delivery. "
        "Leave the receipt in the folder by the front door. "
        "Maya will check the list when she comes back from the afternoon meeting. "
        "The final count can go into the usual ledger on Friday.")
TURN = OPENER + "\n" + TEXT + CLOSER
TYPES = {
    "plain": "text/plain", "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "email": "message/rfc822", "pdf": "application/pdf", "ordinary": "text/plain",
}


def make_uploads(folder):
    import docx
    import pymupdf

    files = {role: folder / f"{role}.{ext}" for role, ext in
             (("plain", "txt"), ("word", "docx"), ("email", "eml"), ("pdf", "pdf"), ("ordinary", "txt"))}
    files["plain"].write_text("Inventory note.\n\n<｜Assistant｜>" + TEXT + "<｜end▁of▁sentence｜>", encoding="utf-8")
    files["ordinary"].write_text("Legal assistant: " + TEXT, encoding="utf-8")
    word = docx.Document()
    for part in (OPENER, TEXT, CLOSER):
        word.add_paragraph(part)
    word.save(files["word"])
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = "editor@example.org", "desk@example.org", "Inventory note"
    message.set_content(f"<html><body><p>{escape(OPENER)}</p><p>{escape(TEXT)}{escape(CLOSER)}</p></body></html>", subtype="html")
    files["email"].write_bytes(message.as_bytes())
    with pymupdf.open() as doc:
        page = doc.new_page()
        assert page.insert_textbox((64, 64, 540, 420), TURN) >= 0
        doc.save(files["pdf"])
    return files


def proof(client, folder, aid, original, evidence, expected):
    import pymupdf

    from app.evidence import document_chat_markers as chat
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan, sha

    public = fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(public)["evidence"] == evidence
    scan(public.decode())
    (folder / "export.json").write_bytes(public)
    listed = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{listed['id']}")
    pdf = fetch(client, f"/api/v2/reports/{listed['id']}/file")
    (folder / "report.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        words = " ".join(" ".join(page.get_text().split()) for page in doc)
        scan(words)
        assert (chat.TITLE in words) == bool(expected["turns"])
        if expected["turns"]:
            assert chat.LIMITATION in words
            for marker in expected["locations"][0].values():
                assert marker["literal"] in "".join(words.split())
                assert f"[{marker['start']}, {marker['end']})" in words
    listed_package = body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": listed["id"], "kind": "draft", "include_original": False,
    }))
    wait(client, f"/api/v2/packages/{listed_package['id']}")
    archive = folder / "signed.zip"
    archive.write_bytes(fetch(client, f"/api/v2/packages/{listed_package['id']}/file"))
    key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(archive, public_key_pem=key, source_document=original)
    assert checked.outcome == "verified", checked.as_dict()
    assert any("Verified chat formatting markers and retained source positions" in row.detail for row in checked.checks)
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence
        assert bundle.read("report.pdf") == pdf
    return {"source_sha256": sha(original.read_bytes()), "pdf_sha256": sha(pdf), "signed_source_verified": True}


def run(output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="cpu", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_LOG_LEVEL="WARNING", VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.text.extract import extract
    from app.evidence import document_chat_markers as chat
    from app.main import app
    from app.storage.db import get_session

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    receipts = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-document-chat-marker", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        files = make_uploads(output)
        for role, source in files.items():
            native = extract(source, source.suffix)
            expected = chat.measure(native.text, native.source_map, native.meta)
            assert expected["turns"] == (0 if role == "ordinary" else 1), role
            aid = checkout(client, stub, source, kind="document", media_type=TYPES[role], filename=source.name)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["coverage"][chat.KEY] == expected and expected["authorship_confidence_contribution"] == 0, role
            coverage = evidence["coverage"]["writing_check_coverage"]
            assert coverage["checks"]["passage_classifier"] == "completed", coverage
            folder = output / f"{role}-evidence"
            folder.mkdir()
            receipts[role] = proof(client, folder, aid, source, evidence, expected) | {
                "markers": expected, "writing_checks": coverage,
            }
            print("Verified", role, flush=True)
    assert all((session["amount_total"], session["currency"]) == (999, "usd") for session in stub.sessions.values())
    (output / "receipt.json").write_text(json.dumps({"cases": receipts, "device": "cpu",
        "checkout": {
        "paid_analyses": len(receipts), "jobs_before_payment": 0, "usd_cents": 999,
    }}, indent=2, sort_keys=True) + "\n")


def ordinary_controls(roster, output):
    from app.detectors.text.extract import extract
    from app.evidence import document_chat_markers as chat
    from scripts.verify_multimedia_exports import sha
    from scripts.writing_word_localization import input_plan

    rows, _, prior = input_plan(roster)
    controls = [row for row in rows if row["role"] == "calibration" and row["family"] == "original"]
    assert len({r["group"] for r in controls}) == len(controls) == 72
    results = []
    with TemporaryDirectory(prefix="chipmunk-marker-controls-") as folder:
        file = Path(folder) / "ordinary.txt"
        for row in controls:
            raw = row["text"].encode()
            assert sha(raw) == row["sha256"]
            file.write_bytes(raw)
            doc = extract(file, ".txt")
            record = chat.measure(doc.text, doc.source_map, doc.meta)
            assert record and record["turns"] == 0
            results.append({"source_sha256": row["sha256"], "mapped_chars": record["checked_chars"], "turns": 0})
    output.write_text(json.dumps({"source_roster_sha256": prior["roster_sha256"], "files": len(results), "flags": 0,
                                  "rows": results}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ordinary-roster", type=Path, help="Replay the 72 ordinary report excerpts into a JSON file.")
    args = parser.parse_args()
    if args.ordinary_roster:
        ordinary_controls(args.ordinary_roster, args.output)
    else:
        run(args.output)
