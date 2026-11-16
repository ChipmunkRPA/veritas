# SPDX-License-Identifier: MIT
"""Replay the earlier-JPEG comparison in the prepaid local app and its exports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts import image_reference_webp_earlier_jpeg as study
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.image_reference_lossy_webp_app import verify
from scripts.image_regional_lossy_webp_transfer import read, save, useful
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("edited", "reencoded", "shifted")


def fixtures(report, sources):
    rows = {(row["source_id"], row["role"]): row for row in report["rows"] if row["profile"] == "progressive"}
    source = next(source for (source, role), row in rows.items() if role == "edited"
                  and useful(row["scores"]["brightness"], row["true_pixels"])
                  and rows[source, "reencoded"]["status"] == "comparable"
                  and rows[source, "reencoded"]["scores"]["brightness"]["pixels"] == 0)
    remaining = set(ROLES)
    for case, earlier, submitted, truth in study.deliveries(sources):
        role = case["role"]
        if case["source_id"] == source and case["profile"] == "progressive" and role in remaining:
            row = rows[source, role]
            assert all(row[key] == value for key, value in case.items())
            yield row, earlier, submitted, truth
            remaining.remove(role)
            if not remaining:
                return
    raise ValueError("The frozen comparison cases could not be rebuilt")


def run(args):
    report = read(args.receipt)
    study.validate(report)
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_IMAGE_REFERENCE_LOSSY_WEBP="true", VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_EARLIER_JPEG="true",
                      VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_COLOR="true")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session

    def result_hash(aid):
        with get_session() as db:
            return db.get(Analysis, aid).result_hash

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    stub = PaymentStub()
    service._stripe = stub
    checked, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-earlier-jpeg-comparison", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        for row, earlier, submitted, truth in fixtures(report, args.sources):
            role = row["role"]
            folder = args.output / role
            folder.mkdir()
            ref, upload = folder / "earlier.jpg", folder / "submitted.webp"
            ref.write_bytes(earlier)
            upload.write_bytes(submitted)
            aid = checkout(client, stub, upload, kind="image", media_type="image/webp", filename=upload.name)
            prior = result_hash(aid)
            endpoint = f"/api/v2/analyses/{aid}/image-reference-comparison"
            assert body(client.get(endpoint))["available"]
            files = {"reference": (ref.name, earlier, "image/jpeg")}
            if role == "edited":
                settings.image_reference_lossy_webp_earlier_jpeg = False
                old = body(client.post(endpoint, files=files))["comparison"]
                assert old["status"] == "webp_lossy_delivery_unassessed" and not old["regions"]
                settings.image_reference_lossy_webp_earlier_jpeg = True
            response = client.post(endpoint, files=files, data={"include_pdf": "true"})
            assert response.headers["cache-control"] == "no-store"
            payload = body(response)
            measurement = row["scores"]["brightness"]
            expected = {"status": row["status"], "selected": measurement["pixels"],
                        "selected_sha256": measurement["sha256"], "overlap": measurement["overlap"]}
            checked[role] = verify(payload, expected, ref, upload, folder, truth, earlier_jpeg=True)
            assert result_hash(aid) == prior
            browser_cases.append({"role": role, "analysis_id": aid, "earlier": str(ref.resolve()),
                                  "expected": row["status"], "regions": len(payload["comparison"]["regions"])})
            print(f"{role}: verified API, exported crops and PDF exhibit pixels", flush=True)
        browser = browser_check(args.output, app, key, browser_cases, scenario="earlier-jpeg-webp-replay.mjs") if args.browser else None
    assert set(checked) == set(ROLES) and len(stub.sessions) == len(checked)
    save(args.output / "receipt.json", {"cases": checked, "browser": browser,
                                       "simulated_payment_usd_cents": 999, "jobs_before_payment": 0,
                                       "overall_analysis_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "sources", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
