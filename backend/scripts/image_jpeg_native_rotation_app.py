# SPDX-License-Identifier: Apache-2.0
"""Replay two large JPEG profiles and a medium JPEG through prepaid analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.image_jpeg_native_rotation_transfer import read, review, sha, useful, write

ROOT = Path(__file__).resolve().parents[2]


def selected(inputs, receipt):
    review(receipt)
    frozen = receipt["freeze"]
    assert read(inputs / "freeze.json") == frozen
    entries = list(zip(frozen["rows"], receipt["observed"], strict=True))
    controls = {(r["geometry"], r["group"]): (r, o["native"]) for r, o in entries if r["role"] == "ordinary"}
    for geometry, quality in (("large", 90), ("large", 95), ("medium", 90)):
        row, expected = next(
            (r, o["native"]) for r, o in entries
            if r["geometry"] == geometry and r["quality"] == quality and r["truth"]
            and o["existing"]["status"] != "supported_pair"
            and useful([item["pair"] for item in o["native"]["pairs"]], r["truth"])
        )
        yield (row, expected), controls[(geometry, row["group"])]


def run(args):
    from scripts.image_exact_turned_copy_app import report
    from scripts.image_jpeg_turned_copy_app import browser_check
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_AFFINE_COPY_REVIEW="true",
                      VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW="true", VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW="true",
                      VERITAS_LOG_LEVEL="ERROR")

    import cv2
    import torch
    from fastapi.testclient import TestClient
    from pymupdf import open as open_pdf

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_copy_similarity_binding import (
        EXPANDED_JPEG_ROTATION_AFFINE_VERSION,
        NATIVE_JPEG_ROTATION_AFFINE_VERSION,
        PNG_COPY_TRANSFORMS,
        ROLES,
        TURNED_JPEG_BASIS,
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

    def analyze(client, path, enabled):
        settings.image_jpeg_native_rotation_review = enabled
        aid = checkout(client, stub, path, kind="image", media_type="image/jpeg", filename=path.name)
        route = f"/api/v2/analyses/{aid}/evidence-packets"
        evidence = body(client.get(route))
        return aid, route, evidence

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    observations, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-native-jpeg-check", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for matched in selected(args.inputs, read(args.receipt)):
            for row, expected in matched:
                path = args.inputs / row["file"]
                assert sha(path) == row["sha256"]
                before, before_route, old = analyze(client, path, False)
                aid, _, evidence = analyze(client, path, True)
                earlier = base_document(old)
                base = base_document(evidence)
                assert earlier["version"] == EXPANDED_JPEG_ROTATION_AFFINE_VERSION
                assert evidence["status"] == "ready" and base["version"] == NATIVE_JPEG_ROTATION_AFFINE_VERSION
                check = base["copy_translation"]
                assert earlier["copy_translation"]["basis"] != TURNED_JPEG_BASIS
                assert base["copy_affine"]["status"] == earlier["copy_affine"]["status"]
                assert check["jpeg_offgrid_review"] is None and check["jpeg_block_review"] is None
                rotation = check["jpeg_turned_review"]
                assert rotation["status"] == expected["status"] and decision(before) == decision(aid)
                assert body(client.get(before_route, params={"set_id": old["id"]})) == old
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                scan(exported.decode())
                assert json.loads(exported)["image_evidence"] == evidence
                note = {"source_sha256": row["sha256"], "size": row["size"], "quality": row["quality"],
                        "status": rotation["status"], "assessment_preserved": True, "earlier_evidence_readable": True}
                if expected["pairs"]:
                    match, = expected["pairs"]
                    pair, = check["pairs"]
                    assert check["basis"] == TURNED_JPEG_BASIS and check["authorship_confidence_contribution"] == 0
                    assert pair["transform"] == PNG_COPY_TRANSFORMS[match["turns"]]
                    assert [[pair["regions"][area][axis] for axis in "xywh"] for area in ROLES] == match["pair"]
                    assert rotation["measurement"] == match["measurement"]
                    output = args.output / path.stem
                    note.update(regions=pair["regions"], transform=pair["transform"],
                                **report(client, aid, path, evidence, pair, output, jpeg=True))
                    with open_pdf(output.with_suffix(".pdf")) as pdf:
                        assert "The rotation check used the original pixel grid." in " ".join(
                            " ".join(page.get_text().split()) for page in pdf
                        )
                    browser_cases.append({"analysis_id": aid, "name": path.stem, "pair": pair})
                else:
                    assert not check["pairs"] and rotation["measurement"] is None
                observations[path.stem] = note
                print(path.stem, rotation["status"], flush=True)
        browser = browser_check(args.output, app, key, browser_cases) if args.browser else None
    write(args.output / "receipt.json", {"cases": observations, "simulated_paid_analyses": len(stub.sessions),
                                         "unpaid_jobs": 0, "browser": browser,
                                         "evaluation_sha256": sha(args.receipt)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "receipt", "inputs"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
