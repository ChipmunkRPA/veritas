# SPDX-License-Identifier: Apache-2.0
"""Check expanded JPEG rotation evidence through prepaid local analyses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts.image_jpeg_rotation_profile_transfer import QUALITIES, digest, read, review, useful, write

ROOT = Path(__file__).resolve().parents[2]


def selected(directory, receipt):
    summary = review(receipt)
    assert digest((directory / "inputs.json").read_bytes()) == receipt["input_plan_sha256"]
    rows = receipt["rows"]
    controls = {(row["quality"], row["group"]): row for row in rows if row["role"] == "ordinary"
                and row["rotation"]["status"] == "no_pair"}
    for quality in QUALITIES:
        if not summary[str(quality)]["advances"]:
            continue
        positive = next(row for row in rows if row["quality"] == quality and (quality, row["group"]) in controls
                        and row["truth"] and useful([match["pair"] for match in row["rotation"]["matches"]], row["truth"]))
        yield positive, controls[quality, positive["group"]]


def run(args):
    from scripts.image_exact_turned_copy_app import report
    from scripts.image_jpeg_turned_copy_app import browser_check
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_JPEG_OFFGRID_COPY_REVIEW="true",
                      VERITAS_IMAGE_AFFINE_COPY_REVIEW="true", VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW="true",
                      VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW="true", VERITAS_LOG_LEVEL="ERROR")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence import image_packets_v7
    from app.evidence.image_copy_similarity_binding import (
        EXPANDED_JPEG_ROTATION_AFFINE_VERSION,
        PNG_COPY_TRANSFORMS,
        ROLES,
        TURNED_JPEG_BASIS,
        TURNED_JPEG_COPY_AFFINE_VERSION,
    )
    from app.evidence.image_embedded_pair_binding import base_document
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(aid):
        with get_session() as db:
            return json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))["decision"]

    def inspect(client, row):
        path = args.inputs / row["file"]
        assert digest(path.read_bytes()) == row["sha256"]
        aid = checkout(client, stub, path, kind="image", media_type="image/jpeg", filename=path.name)
        route = f"/api/v2/analyses/{aid}/evidence-packets"
        return path, aid, route, body(client.get(route))

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    observations, browser_cases = {}, []
    receipt = read(args.receipt)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-jpeg-profile-check", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for positive, control in selected(args.inputs, receipt):
            quality = positive["quality"]
            with patch.object(image_packets_v7, "_analysis_version", return_value=TURNED_JPEG_COPY_AFFINE_VERSION):
                _, before, before_route, old = inspect(client, positive)
            assert base_document(old)["copy_translation"]["basis"] != TURNED_JPEG_BASIS
            for row in (positive, control):
                path, aid, _, evidence = inspect(client, row)
                base = base_document(evidence)
                assert evidence["status"] == "ready" and base["version"] == EXPANDED_JPEG_ROTATION_AFFINE_VERSION
                translation = base["copy_translation"]
                assert translation["jpeg_offgrid_review"]["status"] == ("no_pair" if quality == 85 else "unsupported_profile")
                rotation = translation["jpeg_turned_review"]
                assert rotation["status"] == row["rotation"]["status"]
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                scan(exported.decode())
                assert json.loads(exported)["image_evidence"] == evidence
                note = {"source_sha256": row["sha256"], "source_group": row["group"], "quality": quality,
                        "rotation_status": rotation["status"], "between_block_status": translation["jpeg_offgrid_review"]["status"]}
                if row is positive:
                    (expected,) = row["rotation"]["matches"]
                    (pair,) = translation["pairs"]
                    assert translation["basis"] == TURNED_JPEG_BASIS and translation["authorship_confidence_contribution"] == 0
                    assert pair["transform"] == PNG_COPY_TRANSFORMS[expected["turns"]]
                    assert [[pair["regions"][area][axis] for axis in "xywh"] for area in ROLES] == expected["pair"]
                    assert rotation["measurement"] == expected["measurement"]
                    assert decision(before) == decision(aid)
                    assert body(client.get(before_route, params={"set_id": old["id"]})) == old
                    note.update(regions=pair["regions"], transform=pair["transform"], earlier_evidence_verified=True,
                                assessment_preserved=True, **report(client, aid, path, evidence, pair, args.output / path.stem, jpeg=True))
                    browser_cases.append({"analysis_id": aid, "name": path.stem, "pair": pair})
                else:
                    assert not translation["pairs"] and rotation["measurement"] is None
                observations[path.stem] = note
                print(path.stem, rotation["status"], flush=True)
        browser = browser_check(args.output, app, key, browser_cases) if args.browser else None
    write(args.output / "receipt.json", {"cases": observations, "simulated_paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                         "browser": browser, "evaluation_sha256": digest(args.receipt.read_bytes())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
