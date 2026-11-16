# SPDX-License-Identifier: MIT
"""Verify draft figure evidence through checkout, source replay and browser downloads."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from itertools import product
from pathlib import Path

from scripts import video_short_return_encoded_app as paid

ROOT = Path(__file__).resolve().parents[2]
FORMATS = {".txt": "text/plain", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
           ".pdf": "application/pdf", ".eml": "message/rfc822"}
OLDER = (
    "The agency placed the treatment date at March 4, 2020.\n\n"
    "The board set the annual rate at 4.50 percent of loan balances.\n\n"
    "The approved budget for the work was $1,000."
)
NEWER = (OLDER.replace("March 4, 2020", "2020-03-05")
         .replace("4.50 percent", "5.00%").replace("$1,000", "$1.00 thousand"))
EXPECTED = [("date", "March 4, 2020", "2020-03-05"), ("percentage", "4.50 percent", "5.00%")]


def make_inputs(output):
    from email.message import EmailMessage

    from scripts.replay_document_mixed_formats import rendition

    for side, content in (("earlier", OLDER), ("analyzed", NEWER)):
        for extension in FORMATS:
            source = output / (side + extension)
            if extension == ".eml":
                email = EmailMessage()
                email["From"] = "fixture@example.invalid"
                email.set_content(content + "\n\n> The published budget for the work was $2,000.\n")
                data = email.as_bytes()
            else:
                data = rendition(content, extension)
            source.write_bytes(data)


def check_pdf(data):
    import pymupdf

    from scripts.verify_multimedia_exports import scan

    with pymupdf.open(stream=data, filetype="pdf") as document:
        text = " ".join(" ".join(page.get_text().split()) for page in document)
        assert "Figures and dates" in text and "3 matched · 2 changed" in text
        assert "Date changed · March 4, 2020 → 2020-03-05" in text
        assert "Percentage changed · 4.50 percent → 5.00%" in text
        scan(text)


def run(output, *, browser=False):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DOCUMENT_REVISION_COMPARISON_ENABLED="true",
                      VERITAS_LOG_LEVEL="ERROR", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.main import app
    from app.storage.db import get_session
    from scripts.image_jpeg_turned_copy_app import browser_check
    from scripts.review_document_revision_export import review

    make_inputs(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    verified = {}
    browser_result = None
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-document-figure-check", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        analyses = {
            extension: paid.checkout(client, stub, output / ("analyzed" + extension), kind="document",
                                     media_type=media_type, filename="analyzed" + extension)
            for extension, media_type in FORMATS.items()
        }
        counts = paid.counts()
        original = {aid: paid.body(client.get(f"/api/analyses/{aid}/export.json")) for aid in analyses.values()}
        non_email = tuple(ext for ext in FORMATS if ext != ".eml")
        cases = []
        for old_extension, extension in (*product(non_email, repeat=2), (".eml", ".eml")):
            aid = analyses[extension]
            earlier, later = output / ("earlier" + old_extension), output / ("analyzed" + extension)
            route = f"/api/v2/analyses/{aid}/document-revision-comparison"
            files = {"earlier": (earlier.name, earlier.read_bytes(), FORMATS[old_extension])}
            payload = paid.body(client.post(route, files=files, data={"include_pdf": "true"}))
            result = payload["comparison"]
            proof = result["numeric_comparison"]
            assert (proof["compared"], proof["differences"], proof["affects_overall_confidence"]) == (3, 2, False)
            assert [(row["kind"], row["earlier_value"], row["later_value"]) for row in proof["changes"]] == EXPECTED
            comparison = output / f"{old_extension[1:]}-to-{extension[1:]}.json"
            comparison.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            pdf = base64.b64decode(payload["pdf"])
            comparison.with_suffix(".pdf").write_bytes(pdf)
            check_pdf(pdf)
            verification = review(comparison, earlier, later)
            assert verification["changed_figures_located"] == 2
            verified[f"{old_extension}/{extension}"] = verification | {
                "json_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
                "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
            }
            if old_extension == ".txt" and extension == ".pdf":
                cases.append({"analysis_id": aid, "earlier": str(earlier), "comparison": str(comparison)})
            print(f"Verified figures in {old_extension} and {extension}", flush=True)
        if browser:
            browser_result = browser_check(output, app, key, cases, scenario="draft-figures-replay.mjs")
            check_pdf((output / "browser-comparison.pdf").read_bytes())
        assert paid.counts() == counts
        assert all(paid.body(client.get(f"/api/analyses/{aid}/export.json")) == result for aid, result in original.items())
    (output / "receipt.json").write_text(json.dumps({"paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
        "price_usd": 9.99, "preexisting_analysis_preserved": True, "browser": browser_result, "deliveries": verified},
        indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), browser=args.browser)
