# SPDX-License-Identifier: Apache-2.0
"""Replay the fine interpolation finding through prepaid checkout and exports."""

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_native_cfa_fine_transfer import ROOT, digest, native_inputs, read, review
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait


def check_exports(client, aid, path, evidence, expected, output):
    import cv2
    import numpy as np
    import pymupdf

    from app.evidence.image_interpolation_binding import KEY
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    export = fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(export)["image_evidence"] == evidence
    scan(export.decode())
    output.with_suffix(".json").write_bytes(export)
    assets = {role: fetch(client, asset["url"]) for role, asset in evidence[KEY]["artifacts"].items()}
    for role, data in assets.items():
        assert digest(data) == evidence[KEY]["artifacts"][role]["sha256"]
    if expected:
        x0, y0, x1, y1 = expected
        original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        crop = cv2.imdecode(np.frombuffer(assets["crop"], np.uint8), cv2.IMREAD_UNCHANGED)
        assert crop.dtype == np.uint16 and np.array_equal(crop, original[y0:y1, x0:x1])

    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    output.with_suffix(".pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
        words = " ".join(" ".join(page.get_text().split()) for page in rendered)
        scan(words)
        assert evidence[KEY]["record"]["observation"] in words
        if expected:
            assert evidence[KEY]["record"]["basis"] in words
            assert f"Source columns [{x0}, {x1}) and rows [{y0}, {y1})." in words
            embedded = {pixels(rendered.extract_image(ref)["image"])
                        for ref in {item[0] for page in rendered for item in page.get_images(full=True)}}
            assert len(assets) == 2 and {pixels(data) for data in assets.values()} <= embedded

    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                               json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = output.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(archive, public_key_pem=public, source_image=path)
    assert checked.ok, checked.as_dict()
    assert any(item.ok and "Recomputed the native interpolation inspection" in item.detail for item in checked.checks)
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence and bundle.read("report.pdf") == pdf
    return {"pdf_sha256": digest(pdf), "images_verified_in_pdf": len(assets), "signed_source_verified": True}


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_NATIVE_INTERPOLATION_REVIEW="true",
                      VERITAS_IMAGE_NATIVE_ROTATION_REVIEW="true", VERITAS_IMAGE_NATIVE_QUANTIZED_REVIEW="true")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_interpolation_binding import FINE_SCHEMA, KEY
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.image_native_cfa_png_transfer import png_body

    def assessment(aid):
        with get_session() as db:
            stored = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return stored["decision"], stored.get("customer_assessment")

    def submit(client, path, enabled):
        settings.image_native_fine_review = enabled
        aid = checkout(client, stub, path, kind="image", media_type="image/png", filename=path.name)
        return aid, body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))

    held, development = read(args.held), read(args.calibration)
    review(development)
    review(held, development)
    proof = {}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-small-png-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        source, deliveries = next(native_inputs(args))
        for role, rgb, _ in deliveries:
            if role not in {"32_sharp", "local_tone"}:
                continue
            expected = next(row for row in held["rows"] if row["id"] == source["id"] and row["role"] == role)
            path = args.output / (role + ".png")
            path.write_bytes(png_body(rgb))
            assert digest(path.read_bytes()) == expected["png_sha256"]
            incumbent, older = submit(client, path, enabled=False)
            assert older[KEY]["record"]["status"] == "no_match"
            aid, evidence = submit(client, path, enabled=True)
            actual = evidence[KEY]["record"]
            region = expected["fine"]["region"] if role == "32_sharp" else None
            assert actual["schema"] == FINE_SCHEMA and actual["region"] == region
            assert actual["status"] == ("fine_shift" if region else "no_match")
            assert actual["affects_overall_confidence"] is False and assessment(aid) == assessment(incumbent)
            proof[role] = {"source": source["id"], "source_sha256": expected["png_sha256"], "record": actual,
                           **check_exports(client, aid, path, evidence, region, path.with_suffix(""))}
    assert set(proof) == {"32_sharp", "local_tone"}
    checkout_proof = {"currency": "USD", "cents": 999, "analyses": 4, "before_payment_jobs": 0}
    (args.output / "receipt.json").write_text(json.dumps({"checkout": checkout_proof, "cases": proof}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for option in ("held", "calibration", "cache", "archive"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.set_defaults(phase="held")
    run(parser.parse_args())
