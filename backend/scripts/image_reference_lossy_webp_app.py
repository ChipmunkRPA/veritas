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
    if args.cropped:
        from scripts import image_reference_cropped_webp_transfer as benchmark

        profile = "tight_corner"
        roles = ("small_feathered", "prior_jpeg", "mirrored")
        measured = {(r["case"]["source"], r["case"]["role"]): r for r in receipt["rows"] if r["case"]["profile"] == profile}
        chosen = next(source for (source, role), row in measured.items() if role == roles[0]
                      and row["registered"] == row["case"]["crop"]
                      and useful((row["selected"], row["overlap"]), row["case"]["changed"])
                      and measured[source, roles[1]]["registered"] and not measured[source, roles[1]]["selected"]
                      and not measured[source, roles[2]]["registered"])
        inputs = {}
        for case, ref, upload, truth in benchmark.cases(args, "held", identity=chosen):
            role = case["role"]
            if role in roles and profile == case["profile"]:
                row = measured[chosen, role]
                assert case == row["case"]
                inputs[role] = (row, ref, upload, truth)
        assert set(inputs) == set(roles)
        return inputs
    if args.resized:
        from scripts import image_reference_resized_webp_transfer as benchmark

        profile = "bicubic_webp70"
        roles = ("small_sharp", "prior_webp", "translated")
        rows = receipt["rows"]
        measured = {(r["case"]["source"], r["case"]["role"]): r for r in rows if r["case"]["profile"] == profile}
        chosen = next(source for (source, role), row in measured.items() if role == "small_sharp"
                      and useful((row["selected"], row["overlap"]), row["case"]["changed"])
                      and measured[source, "prior_webp"]["status"] == "comparable" and not measured[source, "prior_webp"]["selected"]
                      and measured[source, "translated"]["status"] != "comparable")
        inputs = {}
        for case, ref, upload, truth in benchmark.deliveries(args, "held", identity=chosen):
            role = case["role"]
            if role in roles and profile == case["profile"]:
                row = measured[chosen, role]
                assert case == row["case"]
                inputs[role] = (row, ref, upload, truth)
        assert set(inputs) == set(roles)
        return inputs
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


def verify(payload, row, ref, submitted, folder, truth, *, resized, cropped):
    import numpy as np
    import pymupdf
    from PIL import Image

    from app.detectors.image import reference_lossy_webp, reference_lossy_webp_crop, reference_lossy_webp_resize
    from scripts.review_image_reference_export import review
    from scripts.verify_multimedia_exports import scan

    comparison = payload["comparison"]
    schema = reference_lossy_webp_resize.SCHEMA if resized or cropped else reference_lossy_webp.SCHEMA
    status = row.get("status", reference_lossy_webp.UNASSESSED)
    summary = "proportional downsampling and WebP compression" if resized else "allows for lossy WebP compression"
    field = "selection_sha256" if resized else "selected_sha256"
    if cropped and row["registered"]:
        schema, status = reference_lossy_webp_crop.SCHEMA, "comparable"
        summary, field = "Matched an unscaled crop after WebP compression", "selection"
        bounds = comparison["lossy_webp_crop"]["reference_crop"]
        assert [bounds[key] for key in ("x", "y", "w", "h")] == row["registered"]
    assert comparison["schema"] == schema
    assert comparison["status"] == status and comparison["affects_overall_confidence"] is False
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
            assert summary in pages[0] and len(regions) + 1 == len(pages)
            mask_bytes = base64.b64decode(payload["pngs"][comparison["mask"]["file"]], validate=True)
            with Image.open(io.BytesIO(mask_bytes)) as image:
                mask = np.asarray(image) == 255
            assert sha(mask.tobytes()) == row[field] and int((mask & truth).sum()) == row["overlap"]
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
    if args.cropped:
        from scripts import image_reference_cropped_webp_transfer as benchmark

        benchmark.review(receipt)
    elif args.resized:
        from scripts import image_reference_resized_webp_transfer as benchmark

        benchmark.verify(receipt)
    else:
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
        for role, (row, earlier, upload, truth) in fixtures.items():
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
            verification[role] = verify(response, row, ref, submitted, folder, truth, resized=args.resized, cropped=args.cropped)
            assert result_hash(aid) == before
    save(args.output / "receipt.json", {"results": verification, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                        "overall_analysis_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--resized", action="store_true")
    geometry.add_argument("--cropped", action="store_true")
    parser.add_argument("--development-sources", type=Path, required=True)
    data = ROOT / "backend/data/independent-evaluation"
    parser.add_argument("--native", type=Path, default=data / "native-regional-serving-v47")
    parser.add_argument("--pairs", type=Path, default=data / "image-positive-pairs-v21-cache")
    parser.add_argument("--background", type=Path, default=data / "image-bg-real-acquisition-v225")
    run(parser.parse_args())
