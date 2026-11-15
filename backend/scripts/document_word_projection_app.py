# SPDX-License-Identifier: MIT
"""Verify inline Word revision coverage through prepaid local analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.document-word-projection-app/1"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PREFIX = "We checked the readings twice before preparing the public log. "
FIRST = "The afternoon session moved to the second floor."
SECOND = "Participants collected the numbered cards before leaving the room."
INSERTED = FIRST + "\t" + SECOND
DELETED = "An earlier timetable placed the afternoon session outdoors. "
AUTHOR = "PRIVATE REVISION AUTHOR FIXTURE"


def create(path, mode):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    if mode != "deleted_only":
        doc.add_paragraph("The community workshop had been scheduled for early spring. Volunteers tested the sign-in process and "
                          "reviewed the printed instructions. Everyone had time to check the new room layout before it opened to visitors.")
    p = doc.add_paragraph(PREFIX)
    if mode == "accepted":
        p.add_run(INSERTED)
    else:
        deletion = OxmlElement("w:del")
        insertion = OxmlElement("w:ins")
        for index, node in enumerate((deletion, insertion), 1):
            node.set(qn("w:id"), str(index))
            node.set(qn("w:author"), AUTHOR)
        delete_run = OxmlElement("w:r")
        removed = OxmlElement("w:delText")
        removed.set(qn("xml:space"), "preserve")
        removed.text = DELETED
        delete_run.append(removed)
        deletion.append(delete_run)
        p._p.append(deletion)
        if mode == "deleted_only":
            doc.save(path)
            return
        insertion.append(p.add_run(FIRST + "\t")._r)
        insertion.append(p.add_run(SECOND)._r)
        p._p.append(insertion)
        if mode == "paragraph_mark":
            prop = OxmlElement("w:rPr")
            mark = OxmlElement("w:del")
            mark.set(qn("w:id"), "3")
            mark.set(qn("w:author"), AUTHOR)
            prop.append(mark)
            p._p.get_or_add_pPr().append(prop)
    p.add_run(" The volunteers recorded attendance on paper so they could check the totals later.")
    doc.add_paragraph("The meeting ended shortly before sunset. Several people stayed to put away the chairs and retrieve the "
                      "extension leads. Two volunteers wrote details on index cards for the organizers who would arrive the following day.")
    doc.add_paragraph("The final account lists the room changes and the order in which the tasks were completed. The committee "
                      "kept the account with the building records and asked a member to read it back before the next workshop.")
    doc.save(path)


def run(output):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(output, ROOT)
    os.environ.update(VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_DEVICE="auto", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", VERITAS_LOG_LEVEL="WARNING")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.text.extract import extract
    from app.engine.models import accelerator_kind
    from app.evidence import document_tracked_changes as tracked
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    observations = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-word-projection", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        for mode in ("accepted", "inline", "paragraph_mark", "deleted_only"):
            path = output / f"{mode}.docx"
            create(path, mode)
            upload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            projected = extract(path, ".docx")
            historical = extract(path, ".docx", project_word_revisions=False)
            assert (SECOND in historical.text) is (mode == "accepted")
            assert (SECOND in projected.text) is (mode in {"accepted", "inline"})
            assert DELETED.strip() not in projected.text and AUTHOR not in projected.text
            aid = checkout(client, stub, path, kind="document", media_type=DOCX, filename=path.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            assert evidence["source"]["sha256"] == upload_hash and AUTHOR.encode() not in exported
            scan(exported.decode())
            status = evidence["coverage"][tracked.KEY]
            writing = evidence["coverage"]["writing_check_coverage"]
            assert status == projected.word_review
            assert evidence["source"]["extracted_text_sha256"] == projected.meta["extracted_text_sha256"]
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            decision = result["decision"]
            skip_reason = {"paragraph_mark": "unsupported_word_revisions", "deleted_only": "too_little_after_word_revisions"}.get(mode)
            assert writing["checks"].get("passage_classifier") == ("not_applicable" if skip_reason else "completed")
            if skip_reason:
                assert decision["code"] == "not_assessable" and not evidence["packets"]
                assert writing["reasons"]["passage_classifier"] == skip_reason
                assert decision["probability_ai_calibrated"] is None and decision["probability_ai_uncalibrated"] is None
            scored = sorted((window["input_sha256"], window["score"]) for signal in result["signals"]
                            for window in (signal.get("details") or {}).get("text_evidence", {}).get("windows", []))
            selected = next((row for row in status["rows"] if row["kind"] == "inserted"), None)
            report_checked = False
            if mode in {"inline", "paragraph_mark"}:
                report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                words = " ".join(pdf_text(pdf).split())
                # This generic public heading is also the registry display label.
                scan(words.replace("Second writing check", "Additional writing response"))
                assert FIRST in words and SECOND in words and AUTHOR not in words
                assert tracked.INPUT_NOTICES[status["writing_input"]] in words
                if mode == "inline":
                    start, end = selected["writing_input_range"]
                    assert (start, end) == (len(PREFIX), len(PREFIX) + len(INSERTED))
                    assert f"Accepted paragraph characters [{start}, {end})." in words
                package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft",
                                                                     "include_original": False}))
                package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
                local = output / f"{mode}.zip"
                local.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                signer = body(client.get("/api/v2/instance/public-key"))
                verified = verify_package(local, public_key_pem=signer["public_key_pem"].encode(),
                                          expect_key_id=signer["key_id"], source_document=path)
                assert verified.outcome == "verified" and verified.signer_authenticated, verified.as_dict()
                (output / f"{mode}.pdf").write_bytes(pdf)
                report_checked = True
            observations[mode] = {"source_sha256": upload_hash, "input_sha256": evidence["source"]["extracted_text_sha256"],
                "historical_input_sha256": historical.meta["extracted_text_sha256"], "writing": writing,
                "status": status["writing_input"], "range": selected.get("writing_input_range") if selected else None,
                "scored_windows": evidence["coverage"]["scored_windows"], "decision": decision["code"],
                "score_input_digest": hashlib.sha256(json.dumps(scored, separators=(",", ":")).encode()).hexdigest() if scored else None,
                "signed_report_checked": report_checked}
            print(mode, observations[mode]["writing"]["checks"], "report verified" if report_checked else "", flush=True)
    receipt = {"schema": SCHEMA, "device": accelerator_kind(), "simulated_price_usd_cents": 999,
               "jobs_started_before_payment": 0, "completed": len(stub.sessions), "rows": observations}
    review(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def review(receipt):
    assert receipt["schema"] == SCHEMA and receipt["simulated_price_usd_cents"] == 999
    assert receipt["completed"] == 4 and receipt["jobs_started_before_payment"] == 0
    rows = receipt["rows"]
    accepted, projected, withheld, short = (rows[key] for key in ("accepted", "inline", "paragraph_mark", "deleted_only"))
    assert len(set(row["source_sha256"] for row in rows.values())) == 4
    assert accepted["input_sha256"] == projected["input_sha256"] != projected["historical_input_sha256"]
    assert accepted["decision"] == projected["decision"] and accepted["scored_windows"] == projected["scored_windows"] > 0
    assert accepted["score_input_digest"] and accepted["score_input_digest"] == projected["score_input_digest"]
    assert accepted["writing"] == projected["writing"] and accepted["writing"]["checks"]["passage_classifier"] == "completed"
    assert projected["status"] == "accepted_inline_changes" and projected["signed_report_checked"] is True
    assert projected["range"] == [len(PREFIX), len(PREFIX) + len(INSERTED)]
    assert withheld["status"] == "unsupported_tracked_changes" and withheld["decision"] == "not_assessable"
    assert withheld["scored_windows"] == 0 and withheld["signed_report_checked"] is True and withheld["range"] is None
    assert short["status"] == "accepted_inline_changes" and short["decision"] == "not_assessable" and short["scored_windows"] == 0
    assert short["writing"]["reasons"]["passage_classifier"] == "too_little_after_word_revisions"
    print("Verified the matched Word inputs, unsupported-revision abstention and two signed source receipts.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    cli.add_argument("--review", action="store_true")
    args = cli.parse_args()
    if args.review:
        review(json.loads(args.path.read_bytes()))
    else:
        run(args.path)
