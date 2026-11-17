"""Verify the prepaid PDF column comparison in the browser, PDF, and signed JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import pdf_column_wording_panel as panel
from scripts import video_short_return_encoded_app as paid
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.writing_published_wording_app import exports


def fixtures(results):
    rows = {(row["id"], row["layout"], row["version"]): row for row in results["rows"]}
    for row in rows.values():
        if (row["layout"] != "balanced" or row["version"] != "mixed" or row.get("unmapped_truth")
                or not 2 <= len(row["columns"]["native_locations"]) <= 8 or row["columns"]["sources"] != [row["id"]]):
            continue
        control = rows[row["id"], row["layout"], "ordinary"]
        if control["columns"]["sources"] == [row["id"]] and not control["columns"]["native_locations"]:
            return row, control
    raise ValueError("The panel has no eligible edited and unmodified app fixtures")


def execute(result, archive, output):
    cases = fixtures(panel.read(result))
    paid.configure(output, panel.ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
                      VERITAS_DOCUMENT_GAO_AUTOMATIC_REVIEW="true", VERITAS_DOCUMENT_GAO_PDF_COLUMNS_REVIEW="true",
                      VERITAS_DOCUMENT_GAO_ARCHIVE_PATH=str(archive), VERITAS_DOCUMENT_GAO_ARCHIVE_SHA256=panel.hash_bytes(archive.read_bytes()),
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="false", VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import document_published_wording as wording
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub
    receipts, decisions, browser_cases = {}, {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-pdf-columns-test", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        for label, row, active in (("edited", cases[0], True), ("ordinary", cases[1], True), ("disabled", cases[0], False)):
            settings.document_gao_pdf_columns_review = active
            folder = output / label
            folder.mkdir()
            source = folder / "upload.pdf"
            source.write_bytes((result.parent / "files" / row["file"]).read_bytes())
            assert panel.hash_bytes(source.read_bytes()) == row["input_sha256"]
            aid = paid.checkout(client, stub, source, kind="document", media_type="application/pdf", filename=source.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            decisions[label] = paid.body(client.get(f"/api/v2/analyses/{aid}"))["decision"]
            scan(json.dumps(evidence))
            matching = evidence["coverage"].get(wording.KEY)
            assert active == bool(matching)
            receipts[label] = {"input_sha256": row["input_sha256"], "column_comparison": active}
            if matching:
                sections = matching["sections"]
                positions = [[place["page"], place["start"], place["end"], place["boxes"]]
                             for section in sections for change in section["changes"] for place in change["locations"]]
                assert positions == row["columns"]["native_locations"]
                assert [[s["page"], s["column"], s["submitted_range"]] for s in sections] == row["columns"]["sections"]
                assert [s["reference"]["id"] for s in sections] == row["columns"]["sources"]
                count = sum(len(section["changes"]) for section in sections)
                receipts[label].update(locations=[pos[:3] for pos in positions], changes=count,
                                       source=sections[0]["reference"]["id"], pdf_sha256=exports(client, aid, source, folder, evidence))
                (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
                browser_cases.append({"analysis_id": aid, "case": label, "reference": receipts[label]["source"],
                                      "date": sections[0]["reference"]["published_date"], "sections": len(sections),
                                      "locations": receipts[label]["locations"], "changes": count})
            print(label, receipts[label], flush=True)
        browser = browser_check(output, app, key, browser_cases, scenario="writing-published-wording-replay.mjs")
    assert decisions["edited"] == decisions["disabled"]
    assert len(stub.sessions) == 3
    (output / "receipt.json").write_text(json.dumps({"cases": receipts, "browser": browser,
        "unchanged_overall_assessment": True, "USD_cents": 999, "prepayment_jobs": 0, "paid_jobs": 3}, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "archive", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    execute(args.result.resolve(), args.archive.resolve(), args.output.resolve())
