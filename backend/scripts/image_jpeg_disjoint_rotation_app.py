# SPDX-License-Identifier: Apache-2.0
"""Verify two distinct rotated copy pairs through local checkout and signed exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import image_jpeg_disjoint_rotation_transfer as study

ROOT = Path(__file__).resolve().parents[2]


def fixtures(args):
    import numpy as np
    from PIL import Image

    study.review(args.receipt)
    receipt = study.prior.read(args.receipt)
    plan = study.inputs(args.inputs, "held")
    earlier = {row["sha256"]: row for row in plan["rows"]}
    choices = []
    for role, quality in (("quarter_and_half", 90), ("two_quarter", 95)):
        winner = next(row for row in receipt["rows"] if row["role"] == role and row["quality"] == quality
                      and len(row["candidate"]["pairs"]) == 2 and all(study.prior.useful(
                          [item["pair"] for item in row["candidate"]["pairs"]], known) for known in row["truth"]))
        choices.append(winner)
    anchor = choices[0]["source_sha256"]
    choices.extend([next(row for row in receipt["rows"] if row["role"] == "ordinary" and row["source_sha256"] == anchor),
                    next(row for row in receipt["rows"] if row["role"] == "shared_source" and row["candidate"]["status"] == "ambiguous")])
    for expected in choices:
        row = earlier[expected["source_sha256"]]
        path = args.inputs / row["file"]
        study.require(study.prior.sha(path) == row["sha256"], "Prepared source differs from its receipt")
        if expected["role"] != "ordinary":
            with Image.open(path, formats=["JPEG"]) as source:
                pixels = np.asarray(source)
            data, truth = study.delivered(pixels, study.sources(pixels, int(row["group"])), expected["role"], row["quality"])
            study.require(study.prior.digest(data) == expected["sha256"] and truth == expected["truth"], "Staged JPEG differs from the receipt")
            path = args.output / (expected["role"] + ".jpg")
            path.write_bytes(data)
        yield path, expected


def run(args):
    from scripts.image_exact_turned_copy_app import report
    from scripts.image_jpeg_turned_copy_app import browser_check
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_AFFINE_COPY_REVIEW="true",
                      VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW="true", VERITAS_IMAGE_NATIVE_ROTATION_REVIEW="true",
                      VERITAS_LOG_LEVEL="ERROR")

    import cv2
    import torch
    from fastapi.testclient import TestClient
    from pymupdf import open as open_pdf

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_copy_similarity_binding import (
        DISJOINT_JPEG_ROTATION_AFFINE_VERSION,
        NATIVE_JPEG_ROTATION_AFFINE_VERSION,
        PNG_COPY_TRANSFORMS,
        ROLES,
    )
    from app.evidence.image_embedded_pair_binding import base_document
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decide(identifier):
        with get_session() as db:
            result = json.loads(default_store().read_bytes(db.get(Analysis, identifier).result_hash))
        return result["decision"], result.get("customer_assessment")

    def analyze(client, file, enabled):
        settings.image_jpeg_disjoint_rotation_review = enabled
        identifier = checkout(client, stub, file, kind="image", media_type="image/jpeg", filename=file.name)
        route = f"/api/v2/analyses/{identifier}/evidence-packets"
        return identifier, route, body(client.get(route))

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    measured, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-disjoint-rotation", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for file, row in fixtures(args):
            name = row["role"]
            previous = None
            if name == "quarter_and_half":
                previous = analyze(client, file, False)
                older = base_document(previous[2])
                assert older["version"] == NATIVE_JPEG_ROTATION_AFFINE_VERSION
                assert older["copy_translation"]["jpeg_turned_review"]["status"] == row["prior"]["status"] == "ambiguous"
            identifier, _, evidence = analyze(client, file, True)
            base = base_document(evidence)
            assert base["version"] == DISJOINT_JPEG_ROTATION_AFFINE_VERSION and evidence["status"] == "ready"
            if previous:
                assert decide(previous[0]) == decide(identifier)
                assert body(client.get(previous[1], params={"set_id": previous[2]["id"]})) == previous[2]
            check = base["copy_translation"]
            pairs = check["pairs"]
            rotation = check["jpeg_turned_review"]
            assert rotation["status"] == row["candidate"]["status"] and check["authorship_confidence_contribution"] == 0
            assert len(pairs) == len(row["candidate"]["pairs"])
            for pair, expected in zip(pairs, row["candidate"]["pairs"], strict=True):
                assert [[pair["regions"][role][axis] for axis in "xywh"] for role in ROLES] == expected["pair"]
                assert pair["transform"] == PNG_COPY_TRANSFORMS[expected["turns"]]
            exported = fetch(client, f"/api/analyses/{identifier}/export.json")
            scan(exported.decode())
            assert json.loads(exported)["image_evidence"] == evidence
            note = {"sha256": row["sha256"], "status": rotation["status"], "regions": [pair["regions"] for pair in pairs]}
            if len(pairs) == 2:
                measurements = [item["measurement"] for item in row["candidate"]["pairs"]]
                assert rotation["measurement"] is None and rotation["measurements"] == measurements
                output = args.output / name
                note.update(report(client, identifier, file, evidence, pairs, output, jpeg=True))
                with open_pdf(output.with_suffix(".pdf")) as pdf:
                    words = " ".join(word for page in pdf for word in page.get_text().split())
                    assert "Two separate pairs of rotated areas share closely matching detail." in words
                    for index, measurement in enumerate(measurements, 1):
                        assert f"T{index:02d} · Rotated JPEG pair" in words
                        assert f"Mean RGB difference inside {measurement['interior_mean']:.2f} of 255 levels." in words
                browser_cases.append({"analysis_id": identifier, "name": name, "pairs": pairs})
            measured[name] = note
            print(name, rotation["status"], len(pairs), flush=True)
        browser = browser_check(args.output, app, key, browser_cases) if args.browser else None
    assert len(stub.sessions) == 5
    study.prior.write(args.output / "receipt.json", {"cases": measured, "browser": browser, "simulated_paid_analyses": 5,
                                                    "unpaid_jobs": 0, "previous_evidence_readable": True,
                                                    "same_overall_assessment": True, "study_sha256": study.prior.sha(args.receipt)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for argument in ("output", "receipt", "inputs"):
        cli.add_argument(argument, type=Path)
    cli.add_argument("--browser", action="store_true")
    run(cli.parse_args())
