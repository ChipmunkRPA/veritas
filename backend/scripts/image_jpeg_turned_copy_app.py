# SPDX-License-Identifier: Apache-2.0
"""Recreate and check paid rotated-JPEG evidence through the browser and exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.image_exact_turned_copy_app import report
from scripts.image_exact_turned_copy_transfer import digest, useful
from scripts.image_jpeg_turned_copy_transfer import deliveries, encode_jpeg, review, source_pixels
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

ROOT = Path(__file__).resolve().parents[2]
CASES = (("quarter", 75), ("half", 90), ("ordinary", 90))


def inputs(args):
    receipt = json.loads(args.receipt.read_bytes())
    review(receipt)
    manifest = (args.archive / "freeze.json").read_bytes()
    assert digest(manifest) == receipt["source_freeze_sha256"]
    by_slot = {(row["source"], row["role"], row["quality"]): row for row in receipt["rows"]}
    for source in json.loads(manifest)["sources"]:
        slot = source["slot"]
        if (slot, *CASES[0]) not in by_slot:
            continue
        rows = [by_slot[slot, *profile] for profile in CASES]
        if not all(row["prior"]["status"] == "no_pair" for row in rows):
            continue
        if not all(useful([match["pair"] for match in row["rotation"]["matches"]], row["known_boxes"]) for row in rows[:2]):
            continue
        if rows[-1]["rotation"]["status"] != "no_pair":
            continue
        images = {name: pixels for name, pixels, _ in deliveries(source_pixels(args.archive, source), slot)}
        for role, quality in CASES:
            row = by_slot[slot, role, quality]
            encoded = encode_jpeg(images[role], quality)
            assert digest(encoded) == row["encoded_sha256"]
            yield role, encoded, row
        return
    raise ValueError("No held source provided both rotations and the matched ordinary control")


def browser_check(output, app, auth_key, cases, *, scenario="jpeg-rotation-replay.mjs"):
    import socket
    import subprocess
    import threading
    import time

    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_config=None, access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(.05)
            if not server.started:
                raise RuntimeError("The isolated local web server did not start")
            result = subprocess.run(
                ["node", str(ROOT / "frontend/e2e" / scenario)], check=True, text=True, timeout=85,
                input=json.dumps({"url": f"http://127.0.0.1:{port}", "key": auth_key, "output": str(output), "cases": cases}),
                stdout=subprocess.PIPE,
            )
            return json.loads(result.stdout)
        finally:
            server.should_exit = True
            thread.join(timeout=8)


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_JPEG_OFFGRID_COPY_REVIEW="true",
                      VERITAS_IMAGE_AFFINE_COPY_REVIEW="true", VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW="true",
                      VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW="false")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_copy_similarity_binding import (
        EXPANDED_JPEG_ROTATION_AFFINE_VERSION,
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
            stored = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return stored["decision"], stored.get("customer_assessment")

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    proofs, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-rotated-jpeg-check", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        for role, encoded, row in inputs(args):
            upload = args.output / f"{role}.jpg"
            upload.write_bytes(encoded)
            kwargs = {"kind": "image", "media_type": "image/jpeg", "filename": upload.name}
            if role == CASES[0][0]:
                settings.image_turned_jpeg_copy_review = False
                incumbent = checkout(client, stub, upload, **kwargs)
                historical = body(client.get(f"/api/v2/analyses/{incumbent}/evidence-packets"))
                assert base_document(historical)["copy_translation"]["basis"] != TURNED_JPEG_BASIS
            settings.image_turned_jpeg_copy_review = True
            aid = checkout(client, stub, upload, **kwargs)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            base = base_document(evidence)
            check = base["copy_translation"]
            rotation = check["jpeg_turned_review"]
            assert evidence["status"] == "ready" and base["version"] == EXPANDED_JPEG_ROTATION_AFFINE_VERSION
            assert rotation["status"] == row["rotation"]["status"]
            export = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(export.decode())
            assert json.loads(export)["image_evidence"] == evidence
            (args.output / f"{role}.json").write_bytes(export)
            proof = {"source_sha256": digest(encoded), "status": rotation["status"], "source_group": row["source"],
                     "quality": row["quality"]}
            if row["rotation"]["matches"]:
                expected, = row["rotation"]["matches"]
                pair, = check["pairs"]
                assert check["basis"] == TURNED_JPEG_BASIS and check["authorship_confidence_contribution"] == 0
                assert pair["transform"] == PNG_COPY_TRANSFORMS[expected["turns"]]
                assert [[pair["regions"][area][axis] for axis in "xywh"] for area in ROLES] == expected["pair"]
                for field in ("interior_mean", "interior_p90"):
                    assert rotation["measurement"][field] == expected[field]
                assert [v for v in rotation["measurement"]["perimeter_means"] if v is not None] == expected["exterior_means"]
                proof.update(regions=pair["regions"], transform=pair["transform"], **report(
                    client, aid, upload, evidence, pair, args.output / role, jpeg=True))
                browser_cases.append({"analysis_id": aid, "name": role, "pair": pair})
            else:
                assert not check["pairs"] and rotation["measurement"] is None
            if role == CASES[0][0]:
                assert decision(incumbent) == decision(aid)
                assert historical == body(client.get(f"/api/v2/analyses/{incumbent}/evidence-packets",
                                                      params={"set_id": historical["id"]}))
                proof["overall_assessment_preserved"] = True
                proof["earlier_evidence_preserved"] = True
            proofs[role] = proof
        browser = browser_check(args.output, app, key, browser_cases) if args.browser else None
    result = {"cases": proofs, "simulated_paid_analyses": len(stub.sessions), "unpaid_jobs": 0}
    if browser is not None:
        result["browser"] = browser
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--browser", action="store_true", help="Also check the built frontend in headless Chromium")
    run(parser.parse_args())
