# SPDX-License-Identifier: Apache-2.0
"""Exercise the new saved-picture review through the normal prepaid document flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_native_alignment_panel import read, save
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.pdf_saved_picture_panel import located, sha
from scripts.pdf_saved_wording_app import ROOT, check_exports, local_model_view


def replay(panel, output):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
                      VERITAS_MODELS_DIR=str(local_model_view(output)), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", VERITAS_WEBHOOKS_ENABLED="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    recorded = read(panel / "results.json.gz")["rows"]
    selected = next(row for row in recorded if row["encoding"] == "jpeg" and row["case"] == "later_note"
                    and len(row["regions"]) == 1 and located(row["regions"], row["target"]))
    planned = {row["case"]: row for row in recorded
               if row["source"] == selected["source"] and row["encoding"] == selected["encoding"]}
    stub = paid.PaymentStub()
    service._stripe = stub
    receipts, browser_cases, assessments = {}, [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("saved-picture-replay", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = paid.body(client.get("/api/v2/instance/public-key"))
        for case, expected in (("changed", "later_note"), ("ordinary", "note"), ("recompressed", "reencoded"), ("off", "later_note")):
            settings.document_pdf_saved_wording_review = case != "off"
            row = planned[expected]
            source = panel / row["source"] / row["encoding"] / f"{expected}.pdf"
            assert sha(source.read_bytes()) == row["sha256"]
            aid = paid.checkout(client, stub, source, kind="document", media_type="application/pdf", filename=source.name)
            route = f"/api/v2/analyses/{aid}"
            result = paid.fetch(client, route)
            evidence = paid.body(client.get(f"{route}/evidence-packets"))
            exported = paid.fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for response in (result, exported, paid.fetch(client, f"{route}/detector-runs")):
                scan(response.decode())
            assessments[case] = {label: json.loads(result).get(label) for label in ("decision", "customer_assessment")}
            review = evidence["coverage"].get("pdf_saved_wording")
            receipt = check_exports(client, route, output / case, source, evidence, review, signer)
            (output / f"{case}.json").write_bytes(exported)
            if review is not None:
                reading = review["inspection"]["pictures"]
                assert reading["status"] == row["status"] and reading["saves"] == row["saves"]
                assert receipt["saved_picture_source_verified"]
                browser_cases.append({"analysis_id": aid, "case": case, "picture": reading,
                                      "sizes": {key: [asset["width"], asset["height"]]
                                                for key, asset in review["artifacts"].items() if key.startswith("picture")}})
            else:
                assert case == "off" and not receipt["saved_picture_source_verified"]
            receipts[case] = {"input_sha256": row["sha256"], **receipt}
            print(case, receipts[case], flush=True)
        browser = browser_check(output, app, key, browser_cases, scenario="pdf-saved-pictures.mjs")
    assert assessments["changed"] == assessments["off"] and len(stub.sessions) == len(receipts)
    save(output / "receipt.json", {"schema": "chipmunk.pdf-saved-pictures-app/1", "cases": receipts, "browser": browser,
                                   "same_assessment_with_option_off": True, "simulated_payment_usd_cents": 999,
                                   "jobs_before_payment": 0})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("output", type=Path)
    replay(**vars(cli.parse_args()))
