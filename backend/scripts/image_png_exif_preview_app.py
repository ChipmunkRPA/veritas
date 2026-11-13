# SPDX-License-Identifier: Apache-2.0
"""Check opt-in PNG preview evidence through isolated prepaid Full analyses."""

from __future__ import annotations

import argparse
import io
import json
import os
import uuid
from pathlib import Path
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, configure, counts, event, fetch, wait


def checkout(client, stub, path, notice):
    before = counts()
    order = body(client.post("/api/v2/billing/checkout", headers={"Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "image", "profile": "full", "risk_accepted": "true",
              "risk_notice_version": notice.VERSION, "risk_notice_sha256": notice.SHA256},
        files={"file": (path.name, path.read_bytes(), "image/png")}))
    assert counts() == before
    sid = next(reversed(stub.sessions))
    nonce = uuid.uuid4().hex
    assert event(client, sid, nonce + "unpaid")["received"] and counts() == before
    stub.sessions[sid].update(payment_status="paid", status="complete")
    assert event(client, sid, nonce + "paid")["received"] and event(client, sid, nonce + "paid")["received"]
    order = body(client.get(f"/api/v2/billing/orders/{order['order_id']}"))
    assert order["status"] == "fulfilled" and counts() == tuple(value + 1 for value in before)
    aid = order["analysis_id"]
    wait(client, f"/api/analyses/{aid}", timeout=220)
    return aid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = args.folder.resolve()
    configure(folder, root)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_WHOLE_REVIEW_PYTHON="",
                      VERITAS_IMAGE_PNG_EXIF_PREVIEW_REVIEW="true", VERITAS_IMAGE_REGION_MODE="legacy")
    import pymupdf
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.api import auth
    from app.billing import purchase_notice, service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.image_editor_preview_binding import CARRIER_SCHEMAS, KEY, PNG_CARRIER
    from app.evidence.image_editor_preview_source import verify
    from app.main import app
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts import image_png_exif_preview_transfer as transfer
    from scripts.verify_multimedia_exports import scan

    get_registry().enabled = lambda _modality, _profile: []
    assert settings.image_png_exif_preview_review and settings.billing_mode == "test"
    assert settings.data_dir.resolve() == folder / "app"

    data = root / "backend/data/independent-evaluation"
    source_args = argparse.Namespace(lama=data / "image-bg-real-acquisition-v225",
                                     flux_earlier=data / "image-residual-patch-held-2026-11-11/plan.json",
                                     flux_cache=data / "image-fresh-source-cache-v39/private-groups.json")
    fixtures = {}
    for item in transfer.cases(source_args):
        family, source, role, encoded, truth = item
        if family == "lama" and role == "ordinary":
            fixtures[source] = encoded
        if family == "lama" and role == "feather":
            measurement = transfer.measure(item, transfer.binding(item))
            if measurement["area"]["useful"]:
                fixture = ("edited", encoded, True), ("ordinary", fixtures[source], False), ("opt-out", encoded, False)
                break
    else:
        raise AssertionError("No admitted useful holdout edit")
    del truth

    def pixels(data):
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            return image.size, transfer.sha(image.tobytes())

    stub = PaymentStub()
    service._stripe = stub
    records = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("png-preview-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for name, encoded, positive in fixture:
            if name == "opt-out":
                settings.image_png_exif_preview_review = False
            path = folder / f"{name}.png"
            path.write_bytes(encoded)
            aid = checkout(client, stub, path, purchase_notice)
            assert body(client.get(f"/api/v2/analyses/{aid}"))["profile"] == "full"
            exported = body(client.get(f"/api/analyses/{aid}/export.json"))
            thumbnail = exported.get(KEY)
            exhibits = body(client.get(f"/api/v2/analyses/{aid}/exhibits"))["items"]
            assert any(exhibit["title"] == "Embedded preview difference locations" for exhibit in exhibits) == positive
            assert bool(thumbnail) == positive
            targets, crops = set(), {}
            if thumbnail:
                assert thumbnail["preview_carrier"] == PNG_CARRIER and thumbnail["authorship_confidence_contribution"] == 0
                for row in thumbnail["regions"]:
                    for asset in row["crops"].values():
                        raw = fetch(client, asset["url"])
                        assert transfer.sha(raw) == asset["sha256"]
                        crops[asset["sha256"]] = raw
                        targets.add(pixels(raw))
                original = verify(encoded, thumbnail, crops.__getitem__)
                assert original["source_crops"] > 0 and original["footprints"] > 0
            report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                text = " ".join(" ".join(page.get_text().split()) for page in document)
                scan(text)
                assert ("Embedded preview differences" in text) == positive
                readback = {pixels(document.extract_image(xref)["image"])
                            for xref in {item[0] for page in document for item in page.get_images(full=True)}}
                assert targets <= readback
            report_path = folder / f"{name}.pdf"
            report_path.write_bytes(pdf)
            pack = body(client.post(f"/api/v2/analyses/{aid}/packages",
                                    json={"report_id": report["id"], "kind": "draft", "include_original": False}))
            pack = wait(client, f"/api/v2/packages/{pack['id']}", timeout=120)
            signed = folder / f"{name}.zip"
            signed.write_bytes(fetch(client, f"/api/v2/packages/{pack['id']}/file"))
            signer = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
            verified = verify_package(signed, public_key_pem=signer, source_image=path)
            assert verified.outcome == "verified", verified.as_dict()
            check = next(item for item in verified.checks if item.name == "original image crops")
            assert check.ok and ("PNG Exif preview footprint" in check.detail) == positive
            with ZipFile(signed) as archive:
                projection = json.loads(archive.read("result.json")).get(KEY)
                assert bool(projection) == positive and archive.read("report.pdf") == pdf
                if projection:
                    assert projection["schema"] == CARRIER_SCHEMAS[PNG_CARRIER]
                    assert verify(encoded, projection, lambda pin: archive.read(f"assets/{pin}.png")) == original
            records[name] = {"prepayment_jobs": 0, "analysis_jobs": 1, "preview_regions": len(thumbnail["regions"]) if thumbnail else 0,
                             "native_crop_assets": len(crops), "pdf_assets_verified": len(targets),
                             "signed_source_replay": True}
            print(name, records[name], flush=True)
    (folder / "receipt.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
