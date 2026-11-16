# SPDX-License-Identifier: MIT
"""Replay the additional writing ranges through prepaid local analysis and export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_multi_passage_panel as study
from scripts import writing_native_passage_panel as native
from scripts.writing_multi_passage_app import ROOT, exports
from scripts.writing_native_passage_app import MIME

CASE = "c417dd5367fb7cc004b6b912"
FORMATS = {"text": "text/plain", **MIME}


def run(plan_path, receipt, font, output, *, browser=False):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
                      VERITAS_TEXT_PASSAGE_DOCUMENT_BASELINE_REVIEW="true", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="false",
                      VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.text.extract import extract
    from app.evidence import document_passage_intervals as proof
    from app.main import app
    from app.storage.db import get_session
    from fastapi.testclient import TestClient
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_character_view_app import tokens_assessed

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan, scores = study.read(plan_path), study.read(receipt)
    assert scores["inputs"]["input_plan_sha256"] == study.digest(json.dumps(plan, sort_keys=True).encode())
    row = next(work for work in plan["works"] if work["document"] == CASE)
    samples = {v["version"]: v["text"] for v in row["variants"] if v["layout"] == "published"}
    comparison = scores["decisions"][CASE]
    expected = comparison["addition"]["wider"]["published"]
    assert expected["hits"] and not expected["ordinary"] and not comparison["pooled"]["wider"]["published"]["mixed"]
    service._stripe = stub = paid.PaymentStub()
    checked, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-document-baseline-review", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()

        def analyze(path, carrier):
            aid = paid.checkout(client, stub, path, kind="document", media_type=FORMATS[carrier], filename=path.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            return aid, evidence["coverage"][proof.KEY]

        for carrier in FORMATS:
            directory = output / carrier
            directory.mkdir()
            for version in ("mixed", "ordinary"):
                path = directory / (version + native.EXTENSION[carrier])
                path.write_bytes(native.fixture(carrier, samples[version], font))
                aid, record = analyze(path, carrier)
                assert record["schema"] == proof.DOCUMENT_BASELINE_SCHEMA and record["authorship_confidence_contribution"] == 0
                doc = extract(path, path.suffix, allow_ocr=False)
                compact = study._prefix(doc.text)
                observed = [([compact[r["start"]], compact[r["end"]]], r["basis"]) for r in record["ranges"]]
                assert observed == [(r["compact"], r["basis"]) for r in expected[version]], (carrier, version, observed)
                assert record["status"] == ("flagged" if observed else "no_ranges")
                case = {"source_sha256": study.digest(path.read_bytes()), "status": record["status"], "ranges": observed}
                if version == "mixed":
                    case["pdf_sha256"] = exports(client, aid, path, record, directory)
                    case["locations"] = [r.get("locations") or [r["source_start"], r["source_end"]] for r in record["ranges"]]
                    browser_cases.append({"analysis_id": aid, "carrier": carrier, "previews": [r["preview"] for r in record["ranges"]]})
                    if carrier == "text":
                        assessment = tokens_assessed(aid)
                checked[f"{carrier}/{version}"] = case
                print(carrier, version, observed, flush=True)

        settings.text_passage_document_baseline_review = False
        aid, original_policy = analyze(output / "text" / "mixed.txt", "text")
        assert original_policy["schema"] == proof.SCHEMA and original_policy["status"] == "no_ranges"
        assert tokens_assessed(aid) == assessment
        result = {"source_case": CASE, "cases": checked, "simulated_prepaid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                  "same_assessment": True, "signed_packages_verified_against_originals": len(FORMATS),
                  "offline_verified_without_analysis_runtime": True, "separate_checkpoint_agreement_enabled": False}
        if browser:
            from scripts.image_jpeg_turned_copy_app import browser_check

            result["browser"] = browser_check(output, app, key, browser_cases, scenario="writing-local-baseline-replay.mjs")
    (output / "receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan_path", "receipt", "font", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.plan_path, args.receipt, args.font, args.output, browser=args.browser)
