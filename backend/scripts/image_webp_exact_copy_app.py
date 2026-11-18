# SPDX-License-Identifier: Apache-2.0
"""Replay held lossless WebP samples through local checkout and evidence exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_webp_exact_copy_transfer import digest, read
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("quarter_24x28", "half_36x28", "straight", "ordinary", "lossy")


def export_report(client, aid, source, evidence, pair, output):
    import pymupdf

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    listed = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{listed['id']}")
    pdf = fetch(client, f"/api/v2/reports/{listed['id']}/file")
    crops = {pixels(fetch(client, item["url"])) for item in pair["artifacts"].values()} if pair else set()
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        if pair:
            assert "lossless WebP" in words
            if pair["transform"] != "same_orientation":
                direction = "90° counterclockwise" if pair["transform"] == "rotate_90_ccw" else "180°"
                assert f"Rotating Area B {direction} aligns every RGB pixel with Area A" in words
                assert "Exact rotated pair" in words
            else:
                assert "crops match exactly in their original orientation" in words and "Exact copy pair" in words
            for region in pair["regions"].values():
                coordinates = f"x {region['x']}, y {region['y']}, {region['w']} x {region['h']} px"
                assert coordinates in words.replace("×", "x")
            embedded = {pixels(document.extract_image(ref)["image"])
                        for ref in {image[0] for page in document for image in page.get_images(full=True)}}
            assert crops <= embedded
        elif source.stem == "lossy":
            assert "Exact comparison requires an opaque lossless WebP still" in words
        else:
            assert "The lossless WebP check found no qualifying exact pair" in words
    output.with_suffix(".pdf").write_bytes(pdf)
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                               json={"report_id": listed["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = output.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    verification = verify_package(archive, public_key_pem=public, source_image=source)
    assert verification.ok, verification.as_dict()
    if pair:
        assert any(item.name == "original image crops" and item.ok and "submitted lossless WebP" in item.detail
                   for item in verification.checks)
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence and bundle.read("report.pdf") == pdf
    return {"signed_source_verified": True, "pdf_sha256": digest(pdf), "unique_native_crops_in_pdf": len(crops)}


def run(proofs: Path, output: Path):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_AFFINE_COPY_REVIEW="true",
                      VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW="true", VERITAS_IMAGE_REGION_MODE="legacy",
                      VERITAS_WEBHOOKS_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    import cv2
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_copy_similarity_binding import (
        PNG_COPY_TRANSFORMS,
        WEBP_EXACT_COPY_BASIS,
        WEBP_EXACT_COPY_KEY,
    )
    from app.evidence.image_embedded_pair_binding import base_document
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(aid):
        with get_session() as db:
            result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            return result["decision"], result.get("customer_assessment")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen = {(row["role"], row["sha256"]): row for row in read(proofs / "receipt.json.gz")["rows"]}
    lossy = output / "lossy.webp"
    with Image.open(proofs / "proof-straight.webp") as photo:
        photo.save(lossy, format="WEBP", quality=85)
    service._stripe = stub = PaymentStub()
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-lossless-webp-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role in ROLES:
            print("Checking", role, flush=True)
            path = lossy if role == "lossy" else proofs / f"proof-{role}.webp"
            source_hash = digest(path.read_bytes())
            kwargs = {"kind": "image", "media_type": "image/webp", "filename": path.name}
            incumbent = None
            if role == ROLES[0]:
                settings.image_lossless_webp_copy_review = False
                incumbent = checkout(client, stub, path, **kwargs)
            settings.image_lossless_webp_copy_review = True
            aid = checkout(client, stub, path, **kwargs)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            base = base_document(evidence)
            assert evidence["status"] == "ready" and base["version"] == "image-evidence-33"
            record = base["copy_translation"]
            export = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(export.decode())
            assert json.loads(export)["image_evidence"] == evidence
            (output / f"{role}.json").write_bytes(export)
            proof = {"status": record[WEBP_EXACT_COPY_KEY]["status"], "source_sha256": source_hash, "analysis_id": aid}
            pair = None
            if role in ROLES[:3]:
                expected = frozen[role, source_hash]
                assert record["basis"] == WEBP_EXACT_COPY_BASIS and record["authorship_confidence_contribution"] == 0
                pair, = record["pairs"]
                measurements = record[WEBP_EXACT_COPY_KEY]["measurement"]
                assert measurements == expected["review"]["measurement"]
                assert pair["transform"] == PNG_COPY_TRANSFORMS[expected["turns"]]
                assert [[box[k] for k in "xywh"] for box in pair["regions"].values()] == expected["exact"][0]
                proof.update(regions=pair["regions"], transform=pair["transform"],
                             matching_pixels=measurements["matching_pixels"])
            else:
                assert not record["pairs"]
                status = "unsupported_profile" if role == "lossy" else frozen[role, source_hash]["review"]["status"]
                assert proof["status"] == status
            proof.update(export_report(client, aid, path, evidence, pair, output / role))
            if incumbent:
                assert decision(incumbent) == decision(aid)
                proof["overall_assessment_preserved"] = True
            results[role] = proof
    receipt = {"cases": results, "simulated_paid_analyses": len(stub.sessions), "unpaid_jobs": 0}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.proofs, args.output)
