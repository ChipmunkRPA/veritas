# SPDX-License-Identifier: Apache-2.0
"""Check held lossy WebP comparisons through the local prepaid app flow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
from pathlib import Path

from scripts import image_reference_webp_residual_holdout as held
from scripts.image_small_region_repeat import read, save, sha, useful
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("sharp", "prior_webp", "translated")


def examples(args, receipt):
    rows = receipt["rows"]
    by_source = {(row["case"]["source"], row["case"]["role"]): row for row in rows if row["case"]["profile"] == "fast_q82"}
    source = next(row["case"]["source"] for row in rows if row["case"]["profile"] == "fast_q82" and row["case"]["role"] == "sharp"
                  and useful((row["selected"], row["overlap"]), row["case"]["changed"])
                  and (control := by_source[row["case"]["source"], "prior_webp"])["status"] == "comparable"
                  and control["selected"] == 0)
    expected = {name: by_source[source, name] for name in ROLES}
    inputs = {}
    for case, ref, body_bytes, truth in held.cases(args):
        role = case["role"]
        if role in ROLES and case == expected[role]["case"]:
            inputs[role] = (expected[role], ref, body_bytes, truth)
        if set(inputs) == set(ROLES):
            return inputs
    raise ValueError("Held app proof cases were incomplete")


def verify(payload, row, ref, submitted, folder, truth):
    import numpy as np
    import pymupdf
    from PIL import Image

    from app.detectors.image import reference_lossy_webp
    from scripts.review_image_reference_export import review
    from scripts.verify_multimedia_exports import scan

    comparison = payload["comparison"]
    assert comparison["schema"] == reference_lossy_webp.SCHEMA
    assert comparison["status"] == row["status"] and comparison["affects_overall_confidence"] is False
    scan(str(comparison))
    pdf = base64.b64decode(payload.pop("pdf"), validate=True)
    export = folder / "comparison.json"
    save(export, payload)
    audit = review(export, ref, submitted)
    assert audit["source_hashes_verified"] == 2 and audit["mask_pixels_counted"] == row["selected"]
    with pymupdf.open(stream=pdf, filetype="pdf") as report:
        pages = [page.get_text() for page in report]
        scan("\n".join(pages))
        if regions := comparison["regions"]:
            assert "allows for lossy WebP compression" in pages[0] and len(regions) + 1 == len(pages)
            mask_bytes = base64.b64decode(payload["pngs"][comparison["mask"]["file"]], validate=True)
            with Image.open(io.BytesIO(mask_bytes)) as image:
                mask = np.asarray(image) == 255
            assert sha(mask.tobytes()) == row["selected_sha256"] and int((mask & truth).sum()) == row["overlap"]
            for text, region in zip(pages[1:], regions, strict=True):
                assert all(region[role]["sha256"] in text for role in ("reference_crop", "submitted_crop", "submitted_overlay"))
        else:
            assert not payload["pngs"] and len(pages) == 1
    (folder / "comparison.pdf").write_bytes(pdf)
    return {"status": comparison["status"], "selected": row["selected"], "overlap": row["overlap"],
            "source_sha256": [sha(ref.read_bytes()), sha(submitted.read_bytes())],
            "pdf_sha256": hashlib.sha256(pdf).hexdigest(), "source_replay": audit}


def run(args):
    receipt = read(args.receipt)
    assert held.readout(receipt["rows"]) == receipt["outcomes"]
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_IMAGE_REFERENCE_LOSSY_WEBP="true")
    fixtures = examples(args, receipt)

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.main import app
    from app.storage.db import Analysis, get_session

    def result_hash(aid):
        with get_session() as db:
            return db.get(Analysis, aid).result_hash

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    verification = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-lossy-webp-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role in ROLES:
            row, earlier, upload, truth = fixtures[role]
            folder = args.output / role
            folder.mkdir()
            ref, submitted = folder / "earlier.png", folder / "submitted.webp"
            ref.write_bytes(earlier)
            submitted.write_bytes(upload)
            aid = checkout(client, stub, submitted, kind="image", media_type="image/webp", filename=submitted.name)
            before = result_hash(aid)
            endpoint = f"/api/v2/analyses/{aid}/image-reference-comparison"
            assert body(client.get(endpoint))["available"]
            response = body(client.post(endpoint, files={"reference": (ref.name, earlier, "image/png")}, data={"include_pdf": "true"}))
            verification[role] = verify(response, row, ref, submitted, folder, truth)
            assert result_hash(aid) == before
    save(args.output / "receipt.json", {"results": verification, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                        "overall_analysis_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--development-sources", type=Path, required=True)
    data = ROOT / "backend/data/independent-evaluation"
    parser.add_argument("--native", type=Path, default=data / "native-regional-serving-v47")
    parser.add_argument("--pairs", type=Path, default=data / "image-positive-pairs-v21-cache")
    parser.add_argument("--background", type=Path, default=data / "image-bg-real-acquisition-v225")
    run(parser.parse_args())
