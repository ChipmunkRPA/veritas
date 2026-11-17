# SPDX-License-Identifier: Apache-2.0
"""Verify paid HEIC analysis, crop exports, PDF, and the ordinary browser view."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pixels(data):
    from PIL import Image

    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        return image.size, digest(image.convert("RGB").tobytes())


def report_and_package(client, folder, name, analysis, source, expected):
    import pymupdf

    from app.evidence.image_editor_preview_binding import CARRIER_SCHEMAS, HEIF_CARRIER, KEY
    from app.evidence.image_editor_preview_source import verify
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    exported = body(client.get(f"/api/analyses/{analysis}/export.json"))
    preview = exported.get(KEY)
    assert bool(preview) == expected
    exhibits = body(client.get(f"/api/v2/analyses/{analysis}/exhibits"))["items"]
    assert any(item["title"] == "Main image preview differences" and item["kind"] == "regions" for item in exhibits) == expected
    source_crops, footprint, plates = 0, 0, {}
    if preview:
        assert preview["authorship_confidence_contribution"] == 0 and preview["preview_carrier"] == HEIF_CARRIER
        assert preview["main_coordinate_space"] == "decoded-heif-main-image"
        assert preview["preview_coordinate_space"] == "decoded-heif-thumbnail"
        for region in preview["regions"]:
            for asset in region["crops"].values():
                raw = fetch(client, asset["url"])
                assert digest(raw) == asset["sha256"]
                assert pixels(raw)[0] == (asset["display_width"], asset["display_height"])
                plates[asset["sha256"]] = raw
        proof = verify(source.read_bytes(), preview, plates.__getitem__)
        source_crops, footprint = proof["source_crops"], proof["footprints"]
        assert source_crops == 2 * len(preview["regions"]) and footprint == len(preview["regions"])
    report = body(client.post(f"/api/v2/analyses/{analysis}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = " ".join(word for page in document for word in page.get_text().split())
        scan(text)
        assert ("Embedded preview differences" in text) == expected
        if preview:
            assert "This finding does not indicate AI authorship" in text
            for region in preview["regions"]:
                for role in ("main_image", "embedded_preview"):
                    box = region[role]
                    assert f"x {box['x']}, y {box['y']}, {box['w']} × {box['h']} px" in text
            images = {pixels(document.extract_image(xref)["image"]) for xref in
                      {row[0] for page in document for row in page.get_images(full=True) if row[8] == 'png'
                       or document.extract_image(row[0])["ext"] == "png"}}
            assert {pixels(raw) for raw in plates.values()} <= images
    (folder / f"{name}.pdf").write_bytes(pdf)
    (folder / f"{name}.json").write_text(json.dumps(exported, sort_keys=True, indent=2) + "\n")
    package = body(client.post(f"/api/v2/analyses/{analysis}/packages",
                                json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}")
    output = folder / f"{name}.zip"
    output.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public_key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    verified = verify_package(output, public_key_pem=public_key, source_image=source)
    assert verified.outcome == "verified", verified.as_dict()
    original = next(item for item in verified.checks if item.name == "original image crops")
    assert original.ok and ("linked HEIC thumbnail" in original.detail) == expected
    with ZipFile(output) as archive:
        projection = json.loads(archive.read("result.json")).get(KEY)
        assert bool(projection) == expected and archive.read("report.pdf") == pdf
        if projection:
            assert projection["schema"] == CARRIER_SCHEMAS[HEIF_CARRIER]
            assert verify(source.read_bytes(), projection, lambda pin: archive.read(f"assets/{pin}.png")) == proof
    return preview, {"source_crops": source_crops, "footprints": footprint, "pdf_sha256": digest(pdf), "signed_replay": True}


def verify_browser(app, root, folder, key, analysis, preview):
    import httpx
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_config=None, access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() >= deadline or not thread.is_alive():
                raise TimeoutError("The isolated browser server did not start")
            time.sleep(.1)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        httpx.get(base, timeout=5).raise_for_status()
        request = {"url": f"{base}/analyses/{analysis}?view=plain", "key": key, "regions": preview["regions"], "folder": str(folder)}
        browser = subprocess.run(["node", str(root / "frontend/e2e/heif-thumbnail-local.mjs")],
                                 input=json.dumps(request), capture_output=True, text=True, timeout=90)
        if browser.returncode:
            raise RuntimeError(browser.stderr.strip())
        return json.loads(browser.stdout)
    finally:
        server.should_exit = True
        thread.join(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("edited", "agreeing", "without_preview", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    configure(folder, root)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_HEIF_THUMBNAIL_REVIEW="true",
                      VERITAS_IMAGE_WHOLE_REVIEW_PYTHON="", VERITAS_IMAGE_REGION_MODE="candidate")
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import get_session

    assert settings.image_heif_thumbnail_review and settings.billing_mode == "test"
    stub = PaymentStub()
    service._stripe = stub
    receipts, decisions = {}, {}
    entries = (("changed", args.edited, True), ("agreeing", args.agreeing, False),
               ("absent", args.without_preview, False), ("opt_out", args.edited, False))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("heic-preview-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for name, source, expected in entries:
            if name == "opt_out":
                settings.image_heif_thumbnail_review = False
            aid = checkout(client, stub, source, kind="image", media_type="image/heic", filename=source.name)
            detail = body(client.get(f"/api/v2/analyses/{aid}"))
            assert detail["profile"] == "full"
            decisions[name] = detail["decision"]
            preview, row = report_and_package(client, folder, name, aid, source, expected)
            receipts[name] = {"prepayment_jobs": 0, "paid_jobs": 1, "preview_regions": len(preview["regions"]) if preview else 0, **row}
            if preview:
                receipts[name]["browser"] = verify_browser(app, root, folder, key, aid, preview)
                (folder / "analysis-id.txt").write_text(aid + "\n")
            print(name, receipts[name], flush=True)
    assert decisions["changed"] == decisions["opt_out"]
    (folder / "receipt.json").write_text(json.dumps({"amount_cents": 999, "currency": "usd", "same_decision_with_check_disabled": True,
                                                    "cases": receipts}, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
