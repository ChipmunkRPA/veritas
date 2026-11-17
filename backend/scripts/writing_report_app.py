"""Check the report calibration, ordinary evidence and exports in the paid local flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_multi_passage_panel as panel
from scripts import writing_native_passage_panel as documents
from scripts.writing_multi_passage_app import ROOT, exports
from scripts.writing_native_passage_app import MIME

CLEAR = "beigebook201404.htm#san_francisco"
RETAIN = "0dc4b726a79138a34eae0b49"


def sample(plan, document, version):
    work = next(work for work in plan["works"] if work["document"] == document)
    return next(item for item in work["variants"] if item["layout"] == "published" and item["version"] == version)["text"]


def run(args):
    paid.configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
                      VERITAS_TEXT_PASSAGE_DOCUMENT_BASELINE_REVIEW="true", VERITAS_TEXT_PASSAGE_REPORT_REVIEW="true",
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="false", VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.text.extract import extract
    from app.evidence import document_passage_intervals as proof
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_character_view_app import tokens_assessed

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    positive = sample(panel.read(args.earlier), RETAIN, "mixed")
    original = sample(panel.read(args.board), CLEAR, "ordinary")
    external = panel.read(args.gao)["works"][0]["variants"][0]["text"]
    service._stripe = stub = paid.PaymentStub()
    rows, browser_cases, assessments = {}, [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-report-calibration", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()

        def analyze(name, text, *, carrier, expected_schema, wanted):
            directory = args.output / name
            directory.mkdir()
            path = directory / ("source" + documents.EXTENSION[carrier])
            path.write_bytes(documents.fixture(carrier, text, args.font))
            aid = paid.checkout(client, stub, path, kind="document",
                                media_type={"text": "text/plain", **MIME}[carrier], filename=path.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            record = evidence["coverage"][proof.KEY]
            assert record["schema"] == expected_schema and record["status"] == wanted, (name, record["schema"], record["status"])
            assert record["authorship_confidence_contribution"] == 0
            assert panel.compact(extract(path, path.suffix, allow_ocr=False).text) == panel.compact(text)
            assessments[name] = tokens_assessed(aid)
            rows[name] = {"schema": record["schema"], "status": record["status"],
                          "locations": [{key: row[key] for key in row if key != "preview"} for row in record["ranges"]],
                          "pdf_sha256": exports(client, aid, path, record, directory)}
            if expected_schema == proof.REPORT_SCHEMA:
                browser_cases.append({"analysis_id": aid, "carrier": name, "previews": [row["preview"] for row in record["ranges"]],
                                      "reportContext": True,
                                      "relativeCount": sum(row.get("basis") == "document_baseline" for row in record["ranges"])})
            print(name, record["schema"], record["status"], flush=True)

        analyze("original-text", original, carrier="text", expected_schema=proof.REPORT_SCHEMA, wanted="no_ranges")
        analyze("original-pdf", original, carrier="pdf", expected_schema=proof.REPORT_SCHEMA, wanted="no_ranges")
        analyze("edited-word", positive, carrier="word", expected_schema=proof.REPORT_SCHEMA, wanted="flagged")
        analyze("other-report", external, carrier="text", expected_schema=proof.DOCUMENT_BASELINE_SCHEMA, wanted="no_ranges")
        settings.text_passage_report_review = False
        analyze("control-text", original, carrier="text", expected_schema=proof.DOCUMENT_BASELINE_SCHEMA, wanted="flagged")
        analyze("control-word", positive, carrier="word", expected_schema=proof.DOCUMENT_BASELINE_SCHEMA, wanted="flagged")
        assert assessments["original-text"] == assessments["control-text"]
        assert assessments["edited-word"] == assessments["control-word"]
        assert rows["edited-word"]["locations"] == rows["control-word"]["locations"]
        receipt = {"cases": rows, "simulated_prepaid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                   "same_overall_assessment": True, "signed_packages_verified": len(rows)}
        if args.browser:
            from scripts.image_jpeg_turned_copy_app import browser_check

            receipt["browser"] = browser_check(args.output, app, key, browser_cases, scenario="writing-local-baseline-replay.mjs")
    (args.output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("board", "earlier", "gao", "font", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
