# SPDX-License-Identifier: Apache-2.0
"""Verify saved PDF wording through prepaid analysis, the browser, and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.pdf_saved_wording_panel import _canonical_save

ROOT = Path(__file__).resolve().parents[2]
HEADING = "Changes across saved PDF versions"
ARCHIVED_TOKEN = "ARCHIVEDTOKENPRIVATE"


def _private_fixture(held, name, output):
    import pymupdf

    path = output / "redaction-test.pdf"
    path.write_bytes((held / f"{name}-base.pdf").read_bytes())
    with pymupdf.open(path) as pdf:
        page = pdf[1]
        # The retained bytes contain this token. The submitted page has a
        # real applied redaction, so it must not appear in default exports.
        rotation = page.rotation
        page.set_rotation(0)
        page.insert_text((85, 525), ARCHIVED_TOKEN)
        bbox, = page.search_for(ARCHIVED_TOKEN)
        page.set_rotation(rotation)
        _canonical_save(pdf, path, incremental=True)
    with pymupdf.open(path) as pdf:
        page = pdf[1]
        page.add_redact_annot(bbox)
        page.apply_redactions(images=0, graphics=0)
        _canonical_save(pdf, path, incremental=True)
        assert ARCHIVED_TOKEN not in page.get_text()
    return path


def fixtures(held, output):
    from app.evidence.document_pdf_history import inspect_path

    indexed = {(row["id"], row["role"]): row for row in read(held / "result.json.gz")["rows"]}
    def source_size(row):
        published = indexed[row["id"], "publisher"]
        return published["report"]["saves"][0]["pages"], (held / published["file"]).stat().st_size

    chosen = min((row for row in indexed.values() if row["role"] == "append" and row["report"]["status"] == "changed"
                  and row["report"]["comparisons"][-1]["changes"][0]["current"]["rotation"] == 90), key=source_size)
    publisher = min((row for row in indexed.values() if row["role"] == "publisher" and row["report"]["status"] == "no_change"),
                    key=source_size)
    name = chosen["id"]
    for label, role in (("changed", "append"), ("form", "form"), ("full-save", "full_rewrite"), ("page", "added_page"),
                        ("publisher", "publisher"), ("off", "append")):
        row = publisher if label == "publisher" else indexed[name, role]
        path = held / row["file"]
        assert digest(path) == row["input_sha256"]
        yield label, path, row["report"] if label != "off" else None
    path = _private_fixture(held, name, output)
    reading = inspect_path(path)
    assert reading["status"] == "changed" and reading["comparisons"][-1]["changes"][0]["kind"] == "delete"
    yield "redaction", path, reading


def check_exports(client, route, output, source, evidence, review, signer):
    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    report = paid.body(client.post(f"{route}/reports", json={"profile": "standard"}))
    rid = paid.wait(client, f"/api/v2/reports/{report['id']}")["id"]
    pdf = paid.fetch(client, f"/api/v2/reports/{rid}/file")
    text = " ".join(pdf_text(pdf).split())
    scan(text)
    assert ARCHIVED_TOKEN not in text and (HEADING.casefold() in text.casefold()) == (review is not None)
    images = []
    if review:
        record = review["inspection"]
        if record["status"] == "changed":
            for change in record["comparisons"][-1]["changes"]:
                for side in ("earlier", "current"):
                    if location := change[side]:
                        a, b = location["word_range"]
                        assert f"{side.capitalize()} page {location['page']}, word positions [{a}, {b})." in text
            for asset in review["artifacts"].values():
                raw = paid.fetch(client, asset["url"])
                assert hashlib.sha256(raw).hexdigest() == asset["sha256"]
                if asset["role"] == "overlay":
                    images.append(pixels(raw))
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            embedded = {((pic.width, pic.height), hashlib.sha256(pic.samples).hexdigest()) for page in doc
                        for entry in page.get_images() for pic in [pymupdf.Pixmap(doc, entry[0])] if pic.n == 3}
        assert set(images) <= embedded and "/coverage/pdf_saved_wording" in text
    output.with_suffix(".pdf").write_bytes(pdf)
    draft = paid.body(client.post(f"{route}/packages", json={"report_id": rid, "kind": "draft", "include_original": False}))
    pid = paid.wait(client, f"/api/v2/packages/{draft['id']}")["id"]
    path = output.with_suffix(".zip")
    path.write_bytes(paid.fetch(client, f"/api/v2/packages/{pid}/file"))
    proof = verify_package(path, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                           source_document=source)
    assert proof.outcome == "verified", proof.as_dict()
    assert any("retained PDF saves" in check.detail for check in proof.checks) == (review is not None)
    with ZipFile(path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
        assert all(ARCHIVED_TOKEN.encode() not in archive.read(name) for name in archive.namelist() if name.endswith(".json"))
    return {"pdf_sha256": hashlib.sha256(pdf).hexdigest(), "verified_pdf_crops": len(images), "signed_replay": True}


def execute(held, output):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
                      VERITAS_MODELS_DIR=str(local_model_view(output)), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", VERITAS_WEBHOOKS_ENABLED="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import document_pdf_history_binding as history
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub
    results, decisions, browser_cases = {}, {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-saved-pdf-wording", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        signer = paid.body(client.get("/api/v2/instance/public-key"))
        for name, source, expected in fixtures(held, output):
            settings.document_pdf_saved_wording_review = expected is not None
            aid = paid.checkout(client, stub, source, kind="document", media_type="application/pdf", filename=source.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = paid.body(client.get(f"{route}/evidence-packets"))
            exported = paid.fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for response in (exported, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
                scan(response.decode())
                assert ARCHIVED_TOKEN.encode() not in response
            detailed = paid.body(client.get(route))
            decisions[name] = (detailed["decision"], detailed.get("customer_assessment"))
            review = evidence["coverage"].get(history.KEY)
            assert (review["inspection"] if review else None) == expected
            results[name] = {"input_sha256": digest(source), "status": expected["status"] if expected else "absent",
                             **check_exports(client, route, output / name, source, evidence, review, signer)}
            if name in {"changed", "form", "redaction"}:
                browser_cases.append({"analysis_id": aid, "case": name, "inspection": review["inspection"],
                                      "hashes": {role: [row["width"], row["height"]] for role, row in review["artifacts"].items()}})
            print(name, results[name]["status"], flush=True)
        browser = browser_check(output, app, key, browser_cases, scenario="pdf-saved-wording.mjs")
    assert decisions["changed"] == decisions["off"] and len(stub.sessions) == len(results)
    save(output / "receipt.json", {"schema": "chipmunk.pdf-saved-wording-app/1", "cases": results, "browser": browser,
                                   "same_assessment_with_option_off": True, "simulated_payment_usd_cents": 999,
                                   "jobs_before_payment": 0, "redacted_token_withheld": True})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("held", type=Path)
    cli.add_argument("output", type=Path)
    execute(**vars(cli.parse_args()))
