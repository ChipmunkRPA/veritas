# SPDX-License-Identifier: MIT
"""Exercise native writing passages through prepaid analysis and signed source replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_native_passage_panel as panel
from scripts.writing_multi_passage_app import CASE, ROOT, exports

MIME = {"word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "email_plain": "message/rfc822", "email_html": "message/rfc822", "pdf": "application/pdf"}


def run(plan, receipt, font, output, *, resume=False):
    paid.configure(output, ROOT, resume=resume)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
                      VERITAS_WEBHOOKS_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient
    from sqlalchemy import select

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.text.extract import extract
    from app.evidence import document_passage_intervals as proof
    from app.main import app
    from app.storage.db import Analysis, BillingOrderRow, PrincipalRow, get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_character_view_app import tokens_assessed

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pinned = panel.source.read(plan)
    candidate = next(work for work in pinned["works"] if work["document"] == CASE and work["split"] == "test")
    variants = {v["version"]: v for v in candidate["variants"] if v["layout"] == "published"}
    compared = panel.source.read(receipt)
    assert compared["input_plan_sha256"] == panel.source.digest(plan.read_bytes())
    expected = {(row["carrier"], row["version"]): row for row in compared["rows"] if row["document"] == CASE}
    service._stripe = stub = paid.PaymentStub()
    result, assessed, submitted = {}, {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = db.scalar(select(PrincipalRow).where(PrincipalRow.username == "native-passage-test")) if resume else None
            if owner is None:
                owner = auth.new_principal("native-passage-test", "examiner")
                db.add(owner)
            db.flush()
            owner_id = owner.id
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()

        def analyze(path, kind, *, reuse=True):
            aid = None
            if resume and reuse:
                digest = panel.source.digest(path.read_bytes())
                with get_session() as db:
                    aid = db.scalar(select(Analysis.id).join(BillingOrderRow, BillingOrderRow.analysis_id == Analysis.id).where(
                        Analysis.sha256 == digest, Analysis.submitted_by == owner_id, Analysis.status == "done",
                        BillingOrderRow.status == "fulfilled", BillingOrderRow.amount_minor == 999,
                    ).order_by(Analysis.created_at).limit(1))
            if aid is None:
                aid = paid.checkout(client, stub, path, kind="document", media_type=MIME[kind], filename=path.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            return aid, evidence["coverage"].get(proof.KEY)

        for kind in MIME:
            folder = output / kind
            folder.mkdir(exist_ok=resume)
            for version in ("mixed", "ordinary"):
                text = variants[version]["text"]
                path = folder / (version + panel.EXTENSION[kind])
                path.write_bytes(panel.fixture(kind, text, font))
                truth = expected[kind, version]
                assert panel.source.digest(path.read_bytes()) == truth["native_sha256"]
                aid, record = analyze(path, kind)
                assert record and record["schema"] == proof.NATIVE_SCHEMA and record["status"] == truth["status"], (kind, version, record)
                assert record["authorship_confidence_contribution"] == 0
                document = extract(path, path.suffix, allow_ocr=False)
                compact = panel.source._prefix(document.text)
                ranges = [[compact[row["start"]], compact[row["end"]]] for row in record["ranges"]]
                assert ranges == truth["selected"]
                case = {"sha256": truth["native_sha256"], "status": record["status"], "selected": ranges,
                        "excluded_characters": len(document.text) - record["checked_end"]}
                if version == "mixed":
                    assert record["status"] == "flagged" and len(panel._match([{"compact": r} for r in ranges], truth["targets"])) >= 1
                    case["pdf_sha256"] = exports(client, aid, path, record, folder)
                    case["locations"] = [{"excerpt_sha256": panel.source.digest(row["preview"].encode()),
                                         "places": [{key: val for key, val in place.items() if key != "boxes"}
                                                    | ({"boxes": len(place["boxes"])} if "boxes" in place else {})
                                                    for place in row["locations"]]} for row in record["ranges"]]
                    submitted[kind], assessed[kind] = path, tokens_assessed(aid)
                    if kind == "email_plain":
                        quote_control = folder / "ordinary-with-quoted-edits.eml"
                        quote_control.write_bytes(panel.fixture(kind, variants["ordinary"]["text"], font, quote=text))
                        _, quoted = analyze(quote_control, kind)
                        assert quoted["status"] == "no_ranges" and quoted["checked_end"] < len(text) + len(variants["ordinary"]["text"])
                        case["quote_control_status"] = quoted["status"]
                result[f"{kind}/{version}"] = case
                print(kind, version, record["status"], ranges, flush=True)
        settings.text_passage_intervals_review = False
        aid, disabled = analyze(submitted["email_plain"], "email_plain", reuse=False)
        assert disabled is None and tokens_assessed(aid) == assessed["email_plain"]
        with get_session() as db:
            fulfilled = db.scalars(select(BillingOrderRow).where(BillingOrderRow.status == "fulfilled")).all()
            assert len(fulfilled) == 10 and all(order.amount_minor == 999 and order.currency == "usd" for order in fulfilled)
    audit = {"source_case": CASE, "cases": result, "same_assessment": True, "simulated_prepaid_analyses": len(fulfilled),
             "unpaid_jobs": 0, "signed_packages_verified_against_originals": len(MIME),
             "pdf_json_and_api_source_locations_agree": True, "offline_verified_without_analysis_runtime": True}
    (output / "receipt.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "receipt", "font", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed prepaid fixtures in the same test directory")
    args = parser.parse_args()
    run(args.plan, args.receipt, args.font, args.output, resume=args.resume)
