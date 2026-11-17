# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen JPEG proposals through the prepaid worker and source exports."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.image_previous_grid_panel import digest, read, save, useful
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def cases(panel):
    result = read(panel / "result.json.gz")
    point = result["freeze"]["points"]["90/surrounded_scan"]["stricter"]
    rows = result["rows"]
    flagged = []
    for row in rows:
        candidate = (row["new"] or {}).get("regions", {}).get("surrounded_scan")
        if row["quality"] == 90 and not row["current"] and candidate and candidate["score"] > point:
            x, y, w, h = candidate["box"]
            flagged.append((row, [x, y, x + w, y + h], useful(candidate["numbers"])))
    first = next(row for row, _, correct in flagged if correct)
    old = next(row for row in rows if row["quality"] == 90 and row["role"] == "sharp" and row["current"])
    peers = {(row["source"], row["role"], row["quality"]): row for row in rows}
    for index, (row, region, correct) in enumerate(flagged, 1):
        yield f"proposal-{index:02d}-{row['role']}", row, region, correct, True
    yield "current-region", old, None, False, True
    yield "ordinary", peers[first["source"], "ordinary", 90], None, False, True
    yield "direct-original", peers[first["source"], "direct_original", 90], None, False, True
    yield "quality-75", peers[first["source"], first["role"], 75], None, False, True
    yield "option-off", first, None, False, False


def exported_evidence(client, route, name, path, evidence, check, signer, output):
    import pymupdf
    from PIL import Image

    from app.evidence import image_previous_grid_binding as binding
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report_id = wait(client, f"/api/v2/reports/{report['id']}")["id"]
    pdf = fetch(client, f"/api/v2/reports/{report_id}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(word for page in document for word in page.get_text().split())
        scan(words)
        assert (binding.TITLE in words) == (check is not None)
        if check:
            assert check["record"]["observation"] in words
            if region := check["record"]["region"]:
                assert f"Source columns [{region[0]}, {region[2]}) and rows [{region[1]}, {region[3]})." in words
                assert binding.BASIS in words
                embedded = {(img.width, img.height, hashlib.sha256(img.samples).hexdigest())
                            for xref in {item[0] for page in document for item in page.get_images()}
                            for img in [pymupdf.Pixmap(document, xref)] if img.n == 3}
                for role, asset in check["artifacts"].items():
                    data = fetch(client, asset["url"])
                    assert digest(data) == asset["sha256"]
                    with Image.open(io.BytesIO(data)) as image:
                        assert (image.width, image.height, digest(image.tobytes())) in embedded
                        if role == "crop":
                            with Image.open(path) as source:
                                assert image.tobytes() == source.convert("RGB").crop(tuple(region)).tobytes()
    (output / f"{name}.pdf").write_bytes(pdf)
    package = body(client.post(f"{route}/packages", json={"report_id": report_id, "kind": "draft", "include_original": False}))
    package_id = wait(client, f"/api/v2/packages/{package['id']}")["id"]
    target = output / f"{name}.zip"
    target.write_bytes(fetch(client, f"/api/v2/packages/{package_id}/file"))
    verified = verify_package(target, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                              source_image=path)
    assert verified.outcome == "verified", verified.as_dict()
    if check and check["record"]["region"] is not None:
        assert any("exact source crop" in row.detail for row in verified.checks)
    with ZipFile(target) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
        projected = json.loads(archive.read("result.json"))
        assert projected.get(binding.COMMITMENT) == evidence["source"].get(binding.COMMITMENT)
    return digest(pdf)


def run(panel, output, *, browser):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_IMAGE_REGION_MODE="candidate",
                      VERITAS_IMAGE_REGION_JPEG_REVIEW="true", VERITAS_IMAGE_REGION_JPEG90_REVIEW="true",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import image_previous_grid_binding as binding
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    stub = PaymentStub()
    service._stripe = stub
    result, decisions, samples, browser_cases = [], {}, set(), []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-jpeg-compression-history", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, row, proposed, correct, enabled in cases(panel):
            file = panel / row["file"]
            assert digest(file.read_bytes()) == row["input_sha256"]
            settings.image_jpeg_prior_grid_review = enabled
            aid = checkout(client, stub, file, kind="image", media_type="image/jpeg", filename=file.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            public = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(public)["image_evidence"] == evidence
            for payload in (public, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(payload.decode())
            review = evidence.get(binding.KEY)
            assert (review is not None) == enabled
            observed = review["record"] if review else {}
            status, region = observed.get("status"), observed.get("region")
            if proposed:
                assert (status, region) in (("matched", proposed), ("earlier_region", None)), (name, status, region)
            elif enabled:
                assert status in {"earlier_region", "no_region", "no_prior_grid", "outside_profile"}, (name, status)
                assert region is None
            if name == "quality-75":
                assert status == "outside_profile"
            if name == "current-region":
                assert status == "earlier_region"
            with get_session() as db:
                analysis = db.get(Analysis, aid)
                stored = json.loads(default_store().read_bytes(analysis.result_hash))
            assert evidence.get(binding.KEY, {}).get("record") == binding.from_result(stored)
            decisions[name] = (stored["decision"], stored.get("customer_assessment"))
            disposition = row["role"] if region else name if not proposed else None
            pdf_sha = None
            if disposition is not None and disposition not in samples:
                samples.add(disposition)
                pdf_sha = exported_evidence(client, route, name, file, evidence, review, signer, output)
                if region is not None:
                    browser_cases.append({"name": name, "analysis_id": aid, "review": review})
            result.append({"case": name, "input_sha256": row["input_sha256"], "role": row["role"], "status": status,
                           "region": region, "proposed_region": proposed, "useful_added_region": bool(region and correct),
                           "source_clock": "native image pixels", "pdf_sha256": pdf_sha})
            print(name, status, region, flush=True)
        off = result[-1]
        compare = next(row for row in result if row["input_sha256"] == off["input_sha256"])
        assert decisions["option-off"] == decisions[compare["case"]]
        assert len(result) == len(stub.sessions) and any(row["useful_added_region"] for row in result)
        replay = browser_check(output, app, key, browser_cases, scenario="jpeg-prior-grid-local.mjs") if browser else None
    save(output / "receipt.json", {"schema": "chipmunk.jpeg-prior-grid-local-checkout/1", "cases": result,
                                   "simulated_payment_usd_cents": 999, "jobs_before_payment": 0,
                                   "same_assessment_with_option_off": True, "browser": replay})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--browser", action="store_true")
    run(**vars(cli.parse_args()))
