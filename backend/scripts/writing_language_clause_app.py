# SPDX-License-Identifier: Apache-2.0
"""Verify clause coverage and fail-closed authorship through the local paid app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts.writing_language_clause_transfer import check, matches_truth, read, sha, write

ROOT = Path(__file__).resolve().parents[2]


def printable_pdf(text):
    try:
        text.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def inputs(frozen, receipt):
    observed = read(receipt)
    check(observed)
    assert observed["freeze_sha256"] == sha(frozen.read_bytes())
    rows = read(frozen)["inputs"]
    assert len(rows) == len(observed["rows"])
    by_key = {(row["group"], row["role"], row["case"]): row for row in rows}
    cases = []
    for role, case in (("original", "substantial_clause"), ("edited", "short_clause")):
        row = next(row for row in observed["rows"] if row["role"] == role and row["case"] == case
                   and matches_truth(row["matches"]["paired"], row["truth"], row["language"])
                   and (role != "original" or printable_pdf(by_key[row["group"], role, case]["source"])))
        key = row["group"], role, case
        chosen = by_key[key]
        assert sha(chosen["source"]) == row["sha256"]
        cases.append((chosen, row["matches"]["paired"]))
    first = cases[0][0]
    return cases, by_key[first["group"], first["role"], "clean"]


def run(args):
    paid.configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_TEXT_WITHIN_SENTENCE_LANGUAGE_REVIEW="true",
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="false",
                      VERITAS_LOG_LEVEL="ERROR")

    import pymupdf
    import torch
    from docx import Document
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.text.extract import text_hash
    from app.evidence.document_check_coverage import CLAUSE_SCHEMA, KEY, witness_description
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()

    def stage(aid, name):
        with get_session() as db:
            row = db.get(AnalysisStage, (aid, name))
            record = db.get(Artefact, row.output_artefact_id)
            return json.loads(default_store().read_bytes(record.sha256))

    def assess(client, path, content_type):
        aid = paid.checkout(client, stub, path, kind="document", media_type=content_type, filename=path.name)
        route = f"/api/v2/analyses/{aid}"
        evidence = paid.body(client.get(f"{route}/evidence-packets"))
        export = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
        scan(json.dumps(export))
        assert export["evidence"] == evidence
        return aid, route, export, evidence

    def verify_report(client, route, path, evidence, witness):
        report = paid.body(client.post(f"{route}/reports", json={"profile": "standard"}))
        paid.wait(client, f"/api/v2/reports/{report['id']}")
        pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
        words = " ".join(pdf_text(pdf).split())
        scan(words)
        assert "Writing not assessed: mixed languages" in words
        assert witness_description(witness) in words
        package = paid.body(client.post(f"{route}/packages", json={
            "report_id": report["id"], "kind": "draft", "include_original": False,
        }))
        paid.wait(client, f"/api/v2/packages/{package['id']}")
        archive = path.with_name(path.name + ".zip")
        archive.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
        pem = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
        verified = verify_package(archive, public_key_pem=pem, source_document=path)
        assert verified.ok, verified.as_dict()
        with ZipFile(archive) as bundle:
            assert json.loads(bundle.read("evidence.json")) == evidence
            assert bundle.read("report.pdf") == pdf
        return {"pdf_sha256": hashlib.sha256(pdf).hexdigest(), "signed_source_verified": True}

    def files(cases):
        substantial, short = cases
        plain = args.output / "substantial.txt"
        plain.write_text(substantial[0]["source"], encoding="utf-8")
        yield plain, "text/plain", substantial
        word_path = args.output / "short.docx"
        word = Document()
        word.add_paragraph(short[0]["source"])
        word.save(word_path)
        yield word_path, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", short
        email_path = args.output / "substantial.eml"
        email = EmailMessage()
        email.set_content(substantial[0]["source"], charset="utf-8", cte="quoted-printable")
        email_path.write_bytes(email.as_bytes())
        yield email_path, "message/rfc822", substantial
        pdf_path = args.output / "substantial.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=1000, height=720)
            assert page.insert_textbox(pymupdf.Rect(35, 40, 965, 675), substantial[0]["source"], fontsize=12) > 0
            pdf.save(pdf_path)
        yield pdf_path, "application/pdf", substantial

    cases, control = inputs(args.frozen, args.receipt)
    notes = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-writing-clause-replay", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for path, content_type, (row, expected) in files(cases):
            aid, route, exported, evidence = assess(client, path, content_type)
            detect = stage(aid, "detect")
            assert detect["writing_language_scope"] == "mixed"
            assert detect["writing_language_clause"] == expected
            assert all(entry["status"] == "not_applicable" for entry in detect["detectors"])
            assessment = exported["customer_assessment"]
            assert assessment["headline"] == "Writing not assessed: mixed languages"
            assert assessment["confidence_pct"] is None and not evidence["packets"]
            writing = evidence["coverage"][KEY]
            witness = writing["language_witness"]
            assert writing["schema"] == CLAUSE_SCHEMA and set(writing["reasons"].values()) == {"mixed_language"}
            extracted = stage(aid, "decode")["text"]
            assert extracted == row["source"] and witness["sentence"] == expected["sentence"] + 1
            assert witness["range"] == {"start": expected["start"], "end": expected["end"],
                                        "sha256": text_hash(extracted[expected["start"]:expected["end"]])}
            notes.append({"source_sha256": sha(path.read_bytes()), "format": path.suffix,
                          "language_witness": witness, "writing_checks": writing["checks"],
                          **verify_report(client, route, path, evidence, witness)})
            print(path.name, "source and reports verified", flush=True)
        path = args.output / "ordinary.txt"
        path.write_text(control["source"], encoding="utf-8")
        aid, _, _, evidence = assess(client, path, "text/plain")
        assert stage(aid, "detect")["writing_language_scope"] == "english"
        assert evidence["coverage"][KEY]["checks"]["passage_classifier"] == "completed"
        assert "language_witness" not in evidence["coverage"][KEY]
        with get_session() as db:
            assert db.get(Analysis, aid).status == "done"
        print(path.name, "writing check completed", flush=True)
    write(args.output / "receipt.json", {"findings": notes, "ordinary_control_completed": True,
                                          "simulated_paid_analyses": len(stub.sessions), "unpaid_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "frozen", "receipt"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
