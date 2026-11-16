# SPDX-License-Identifier: MIT
"""Verify paid analysis, text highlights and signed export for a pinned writing pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts import writing_multi_passage_panel as study

ROOT = Path(__file__).resolve().parents[2]
CASE = "6bfce7a0f53514fcc0b227be"


def exports(client, aid, source, record, output):
    import pymupdf

    from app.evidence import document_passage_intervals as proof
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    report = paid.body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    paid.wait(client, f"/api/v2/reports/{report['id']}")
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    (output / "passage-report.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
        text = " ".join(" ".join(page.get_text().split()) for page in rendered)
        scan(text)
        assert proof.TITLE in text and proof.STATUS["flagged"] in text
        for row in record["ranges"]:
            assert f"Original text characters [{row['source_start']}, {row['source_end']})" in text
            assert " ".join(row["preview"].split()[:8]) in text
    public = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
    scan(json.dumps(public))
    package = paid.body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False}))
    paid.wait(client, f"/api/v2/packages/{package['id']}")
    archive = output / "passage-evidence.zip"
    archive.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    replay = verify_package(archive, public_key_pem=key, source_document=source)
    assert replay.ok, replay.as_dict()
    assert any("Verified writing passage selection" in check.detail for check in replay.checks)
    with ZipFile(archive) as packaged:
        frozen = json.loads(packaged.read("evidence.json"))
        assert frozen["coverage"][proof.KEY] == record
        scan(packaged.read("result.json").decode())
    assert record == public["evidence"]["coverage"][proof.KEY]
    return hashlib.sha256(pdf).hexdigest()


def run(sources, output):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
                      VERITAS_WEBHOOKS_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    base, _ = study.load_files(sources, "test")
    case = base[CASE]
    study._validate(case)

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import document_passage_intervals as proof
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_character_view_app import tokens_assessed

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    manipulated, original = output / "two-passages.txt", output / "original.txt"
    manipulated.write_text(case["text"], encoding="utf-8")
    original.write_text(case["source_text"], encoding="utf-8")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-passage-interval-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()

        def analyze(path):
            aid = paid.checkout(client, stub, path, kind="document", media_type="text/plain", filename=path.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            return aid, evidence["coverage"].get(proof.KEY)

        aid, flagged = analyze(manipulated)
        assert flagged is not None and flagged["status"] == "flagged", flagged
        assert flagged["authorship_confidence_contribution"] == 0
        intervals = [(row["source_start"], row["source_end"]) for row in flagged["ranges"]]
        labels = [(row["start"], row["end"]) for row in case["spans"] if row["label"] == "ai"]
        assert len(intervals) == len(labels) == 2
        for (start, end), (a, b) in zip(intervals, labels, strict=True):
            common = min(b, end) - max(a, start)
            assert common >= (b - a) * .5 and common >= (max(b, end) - min(a, start)) * .3
        pdf = exports(client, aid, manipulated, flagged, output)
        _, control = analyze(original)
        assert control is not None and control["status"] == "no_ranges" and control["ranges"] == []
        settings.text_passage_intervals_review = False
        other, disabled = analyze(manipulated)
        assert disabled is None and tokens_assessed(other) == tokens_assessed(aid)
        result = {"case": CASE, "source_sha256": hashlib.sha256(manipulated.read_bytes()).hexdigest(),
                  "source_intervals": intervals, "labeled_intervals": labels, "original_status": control["status"],
                  "authorship_contribution": 0, "same_assessment": True, "usd_test_checkouts": len(stub.sessions),
                  "no_jobs_before_payment": True, "api_json_pdf_and_original_verified": True,
                  "pdf_sha256": pdf}
        (output / "receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.sources, args.output)
