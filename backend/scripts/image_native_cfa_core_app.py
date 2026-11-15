# SPDX-License-Identifier: Apache-2.0
"""Replay compact interpolation evidence through local checkout and exports."""

import argparse
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import image_native_cfa_core_transfer as comparison
from scripts.image_native_cfa_fine_app import check_exports
from scripts.image_native_cfa_fine_transfer import ROOT, digest, native_photos
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_NATIVE_INTERPOLATION_REVIEW="true",
                      VERITAS_IMAGE_NATIVE_ROTATION_REVIEW="true", VERITAS_IMAGE_NATIVE_QUANTIZED_REVIEW="true",
                      VERITAS_IMAGE_NATIVE_FINE_REVIEW="true", VERITAS_IMAGE_NATIVE_EXTENT_REVIEW="true")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.image_interpolation_binding import CORE_SCHEMA, KEY
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.image_native_cfa_png_transfer import png_body

    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        receipt = json.load(stream)
    comparison.review(receipt)
    assert receipt["points"]["core"]["tight"] == 0.
    rows = receipt["readings"]["held"]
    edit = next(row for row in rows if row["role"] == "offset_feather9"
                and comparison.useful(row["proposed"]["core"], row["changed"]))
    group = edit["id"]
    control = next(row for row in rows if row["role"] == "offset_even_feather" and row["id"] == group)
    with_photo = SimpleNamespace(phase="held", cache=args.cache, archive=args.archive)
    source, index, pixels = next((photo, index, pixels) for photo, index, pixels in native_photos(with_photo) if photo["id"] == group)
    fixtures = {role: im for role, im, _ in comparison.new_cases(pixels, source["id"], index, "held")
                if role in {edit["role"], control["role"]}}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub

    def assessment(aid):
        with get_session() as db:
            saved = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return saved["decision"], saved.get("customer_assessment")

    proof = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-compact-png-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for row in (edit, control):
            role = row["role"]
            path = args.output / f"{role}.png"
            path.write_bytes(png_body(fixtures[role]))
            assert digest(path.read_bytes()) == row["png_sha256"]
            decisions = []
            for enabled in (False, True):
                settings.image_native_core_review = enabled
                aid = checkout(client, stub, path, kind="image", media_type="image/png", filename=path.name)
                evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
                record = evidence[KEY]["record"]
                decisions.append(assessment(aid))
                selected = row["proposed"]["core"] if enabled else None
                expected = selected["region"] if selected else None
                assert record["region"] == expected and record["affects_overall_confidence"] is False
                assert record["status"] == ("compact_shift" if expected else "no_match")
                if enabled:
                    assert record["schema"] == CORE_SCHEMA
                    proof[role] = {"source": group, "record": record,
                                   **check_exports(client, aid, path, evidence, expected, args.output / role)}
            assert decisions[0] == decisions[1]
            print(f"Verified {role} with the compact check off and on.", flush=True)
    summary = {"cases": proof, "analyses": 4, "before_payment_jobs": 0, "usd_cents": 999,
               "automated_assessments_unchanged": True}
    (args.output / "receipt.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "receipt", "cache", "archive"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
