# SPDX-License-Identifier: MIT
"""Verify accepted Word paragraph breaks through prepaid local analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.document_word_projection_app import AUTHOR, DOCX, ROOT

SCHEMA = "chipmunk.document-word-breaks-app/1"
LEAD = "The local planning office scheduled a public session in April. Everyone received the map and a list of contacts."
LEFT = "The largest room had an inter"
RIGHT = ("ruption during the head count. Volunteers moved the registration desk after noticing the door was blocked. "
         "The coordinator checked that families could reach the desk without waiting in the hallway.")
SEPARATE = "Two volunteers reviewed the log. They checked the dates before handing the records to the office."
FOLLOWING = "The results were reviewed at the next meeting. The signed agenda and numbered exhibits went into the same folder."
ENDING = "The original forms left the building in a sealed envelope. The office stored a second copy in the archive."


def write_fixture(path, layout, *, marked):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    doc.add_paragraph(LEAD)
    if layout == "joined" and not marked:
        doc.add_paragraph(LEFT + RIGHT)
    else:
        p = doc.add_paragraph(LEFT if layout != "split" else SEPARATE)
        if marked:
            properties = OxmlElement("w:rPr")
            change = OxmlElement("w:ins" if layout == "split" else "w:del")
            change.set(qn("w:id"), "27")
            change.set(qn("w:author"), AUTHOR)
            properties.append(change)
            p._p.get_or_add_pPr().append(properties)
        if layout == "table":
            doc.add_table(1, 1).cell(0, 0).text = "Attendance table."
        doc.add_paragraph(RIGHT if layout != "split" else ENDING)
    doc.add_paragraph(FOLLOWING)
    doc.add_paragraph(ENDING if layout != "split" else LEAD)
    doc.save(path)


def receipt_check(data):
    assert data["schema"] == SCHEMA and data["payment_cents"] == 999 and data["unpaid_jobs"] == 0
    cases = data["cases"]
    assert len(cases) == 5
    assert len({row["original"] for row in cases.values()}) == 5
    for name, kind in (("joined", "deleted"), ("split", "inserted")):
        accepted, marked = cases[name], cases[name + "_tracked"]
        assert marked["mark"] == {"kind": kind, "paragraph": 2, "body_index": 1, "child_path": [0, 0, 0]}
        assert marked["input"] == "accepted_paragraph_changes" and marked["prior_input"] == "unsupported_tracked_changes"
        assert marked["signed"] and accepted["signed"]
        for field in ("text", "score_digest", "checks", "windows", "decision"):
            assert marked[field] == accepted[field], (name, field)
        assert marked["checks"] == {"passage_classifier": "completed", "writing_pattern_ratio": "completed"}
        assert marked["windows"] > 0
    assert cases["joined_tracked"]["source_paths"] == [[1], [2]]
    blocked = cases["table"]
    assert blocked["windows"] == 0 and blocked["decision"] == "not_assessable"
    assert blocked["input"] == "unsupported_tracked_changes" and blocked["signed"]
    print("Verified accepted Word inputs, two original body paths, the unassessed control and five signed reports.")


def run(output):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(output, ROOT)
    os.environ.update(VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_DEVICE="auto", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", VERITAS_LOG_LEVEL="WARNING")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.canonical import canonical_hash
    from app.detectors.text.extract import extract
    from app.engine.models import accelerator_kind
    from app.evidence.document_packets import _source_anchors
    from app.evidence.document_tracked_changes import INPUT_NOTICES, KEY
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
    reports = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-word-breaks", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for mode in ("joined", "joined_tracked", "split", "split_tracked", "table"):
            path = output / f"{mode}.docx"
            layout = mode.removesuffix("_tracked")
            write_fixture(path, layout, marked=mode != layout or mode == "table")
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            current = extract(path, ".docx")
            prior = extract(path, ".docx", project_word_breaks=False)
            aid = checkout(client, stub, path, kind="document", media_type=DOCX, filename=path.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence and AUTHOR.encode() not in exported
            scan(exported.decode())
            assert evidence["source"]["sha256"] == original
            assert evidence["source"]["extracted_text_sha256"] == current.meta["extracted_text_sha256"]
            coverage = evidence["coverage"]
            tracked = coverage[KEY]
            assert tracked == current.word_review
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            scored = sorted((window["input_sha256"], window["score"]) for signal in result["signals"]
                            for window in (signal.get("details") or {}).get("text_evidence", {}).get("windows", []))
            paths = []
            if mode == "joined_tracked":
                start = current.text.index("interruption")
                anchors = _source_anchors(current.source_map, start, start + len("interruption"), len(current.text))
                paths = [anchor["body_path"] for anchor in anchors]
                assert [current.text[a["selected_start"]:a["selected_end"]] for a in anchors] == ["inter", "ruption"]
                assert "interruption" not in prior.text
            if mode == "table":
                assert not evidence["packets"] and not scored
                assert coverage["writing_check_coverage"]["reasons"]["passage_classifier"] == "unsupported_word_revisions"
            report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            printed = " ".join(pdf_text(pdf).split())
            scan(printed.replace("Second writing check", "Additional writing response"))
            mark = next(iter(tracked["paragraph_marks"]), None)
            if mark:
                assert INPUT_NOTICES[tracked["writing_input"]] in printed
                assert f"{mark['kind'].capitalize()} paragraph break" in printed and "child path 0 / 0 / 0" in printed
            assert AUTHOR not in printed
            package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft",
                                                                 "include_original": False}))
            package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
            archive = output / f"{mode}.zip"
            archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
            verification = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(),
                                          expect_key_id=signer["key_id"], source_document=path)
            assert verification.outcome == "verified" and verification.signer_authenticated, verification.as_dict()
            reports[mode] = {"original": original, "text": current.meta["extracted_text_sha256"], "source_paths": paths,
                             "decision": result["decision"]["code"], "score_digest": canonical_hash(scored),
                             "windows": coverage["scored_windows"], "checks": coverage["writing_check_coverage"]["checks"],
                             "input": tracked["writing_input"], "prior_input": prior.word_review["writing_input"],
                             "mark": mark, "signed": True}
            print(mode, reports[mode]["checks"], reports[mode]["windows"], "signed report verified", flush=True)
    assert len(stub.sessions) == len(reports)
    receipt = {"schema": SCHEMA, "device": accelerator_kind(), "payment_cents": 999, "unpaid_jobs": 0, "cases": reports}
    receipt_check(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    cli.add_argument("--review", action="store_true")
    args = cli.parse_args()
    if args.review:
        receipt_check(json.loads(args.path.read_bytes()))
    else:
        run(args.path)
