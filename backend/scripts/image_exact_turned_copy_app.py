# SPDX-License-Identifier: Apache-2.0
"""Verify the rotated PNG check through a local prepaid analysis and report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_exact_turned_copy_transfer import cases, digest, review, source_pixels
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("quarter_24x28", "half_36x28", "ordinary")


def inputs(args):
    document = json.loads(args.receipt.read_bytes())
    review(document)
    manifest = (args.archive / "freeze.json").read_bytes()
    assert digest(manifest) == document["source_freeze_sha256"]
    records = {(row["group"], row["role"]): row for row in document["rows"]}
    for source in json.loads(manifest)["sources"]:
        slot = source["slot"]
        rows = {role: records[slot, role] for role in ROLES}
        if max(rows[ROLES[0]]["size"]) < 513 or not all(
            row["translation_status"] == "no_pair"
            and (row["rotated"]["status"] == ("no_pair" if role == "ordinary" else "supported_pair"))
            for role, row in rows.items()
        ):
            continue
        for role, rgb, _ in cases(source_pixels(args.archive, source), slot):
            if role in rows:
                assert digest(rgb.tobytes()) == rows[role]["rgb_sha256"]
                yield role, rgb, rows[role]
        return
    raise ValueError("No common held source for both rotations and their matched control")


def report(client, aid, input_path, evidence, pair, exports, *, jpeg=False):
    import pymupdf

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    listed = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{listed['id']}")
    pdf = fetch(client, f"/api/v2/reports/{listed['id']}/file")
    exports.with_suffix(".pdf").write_bytes(pdf)
    images = {pixels(fetch(client, asset["url"])) for asset in pair["artifacts"].values()}
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        direction = "90° counterclockwise" if pair["transform"] == "rotate_90_ccw" else "180°"
        summary = "closely matches Area A" if jpeg else "matches Area A exactly"
        assert f"Area B {summary} after rotating it {direction}." in words
        assert ("Rotated JPEG pair" if jpeg else "Exact rotated pair") in words
        for region in pair["regions"].values():
            coordinates = f"x {region['x']}, y {region['y']}, {region['w']} x {region['h']} px"
            assert coordinates in words.replace("×", "x")
        embedded = {pixels(document.extract_image(ref)["image"])
                    for ref in {image[0] for page in document for image in page.get_images(full=True)}}
        assert images <= embedded and len(images) == 2
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                               json={"report_id": listed["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = exports.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    verification = verify_package(archive, public_key_pem=public, source_image=input_path)
    assert verification.ok, verification.as_dict()
    assert any(item.name == "original image crops" and item.ok and "after rotation" in item.detail
               for item in verification.checks)
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence and bundle.read("report.pdf") == pdf
    return {"pdf_sha256": digest(pdf), "signed_source_verified": True, "native_crops_verified_in_pdf": len(images)}


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_SMALL_PNG_COPY_REVIEW="true",
                      VERITAS_IMAGE_AFFINE_COPY_REVIEW="true", VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW="true")

    import cv2
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_copy_similarity_binding import PNG_COPY_TRANSFORMS, TURNED_PNG_BASIS
    from app.evidence.image_copy_similarity_binding import ROLES as AREAS
    from app.evidence.image_embedded_pair_binding import base_document
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(aid):
        with get_session() as db:
            stored = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return stored["decision"], stored.get("customer_assessment")

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    proofs = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-rotated-png-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role, rgb, row in inputs(args):
            input_path = args.output / (role + ".png")
            Image.fromarray(rgb).save(input_path)
            expected = next(iter(row["rotated"]["pairs"]), None)
            if role == ROLES[0]:
                settings.image_turned_png_copy_review = False
                incumbent = checkout(client, stub, input_path, kind="image", media_type="image/png", filename=input_path.name)
                baseline = body(client.get(f"/api/v2/analyses/{incumbent}/evidence-packets"))
                assert base_document(baseline)["copy_translation"]["basis"] != TURNED_PNG_BASIS
            settings.image_turned_png_copy_review = True
            aid = checkout(client, stub, input_path, kind="image", media_type="image/png", filename=input_path.name)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            check = base_document(evidence)["copy_translation"]
            assert evidence["status"] == "ready" and check["status"] == row["rotated"]["status"]
            export = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(export.decode())
            assert json.loads(export)["image_evidence"] == evidence
            (args.output / (role + ".json")).write_bytes(export)
            proof = {"source_sha256": digest(input_path.read_bytes()), "status": check["status"], "source_group": row["group"]}
            if expected is not None:
                pair, = check["pairs"]
                assert check["basis"] == TURNED_PNG_BASIS and check["authorship_confidence_contribution"] == 0
                assert pair["transform"] == PNG_COPY_TRANSFORMS[expected["turns"]]
                live_boxes = [[pair["regions"][area][key] for key in "xywh"] for area in AREAS]
                assert live_boxes == expected["pair"]
                assert check["small_png_copy_review"]["measurement"] == expected["measurement"]
                proof.update(regions=pair["regions"], transform=pair["transform"], **report(
                    client, aid, input_path, evidence, pair, args.output / role))
            if role == ROLES[0]:
                assert decision(incumbent) == decision(aid)
                proof["overall_assessment_preserved"] = True
            proofs[role] = proof
    result = {"cases": proofs, "simulated_paid_analyses": len(stub.sessions), "unpaid_jobs": 0}
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("archive", type=Path)
    run(parser.parse_args())
