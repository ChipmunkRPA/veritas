# SPDX-License-Identifier: Apache-2.0
"""Replay held WebP color evidence through checkout, the API, PDF and JSON."""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

from scripts import image_reference_webp_color_transfer as panel
from scripts.image_small_region_repeat import save, sha, useful
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]


def fixtures(packet, sources, camera_dir):
    held = [spec for spec in panel.source_specs() if spec["split"] == packet["split"]]
    rows = {(row["case"]["profile"], row["case"]["group"], row["case"]["role"]): row for row in packet["rows"]}

    def matches(row, role):
        states = row["measurements"]
        if role == "unrelated":
            return states is None
        if states is None or row["measured_crop"] != row["case"]["expected_crop"]:
            return False
        color = states["color_gamut"]["color"]
        if role in panel.CONTROLS:
            return not states["color_gamut"]["selected"]
        return useful((color["selected"], color["overlap"]), row["case"]["changed"])

    targets = [(profile, "small_recolor") for profile in panel.PROFILES]
    targets += [("crop_resize", "separate_luma_and_color"), ("same", "whole_tint"), ("same", "unrelated")]
    for profile, role in targets:
        index, spec = next((i, item) for i, item in enumerate(held)
                           if (row := rows.get((profile, item["group"], role))) and matches(row, role))
        paired = held[(index + len(held) // 2) % len(held)]
        foreign, _ = panel.source_pixels(sources, camera_dir, paired)
        delivery, earlier, submitted, truth = next(panel.deliveries(spec, sources, camera_dir, foreign, profile=profile, role=role))
        expected = rows[profile, spec["group"], role]
        assert expected["case"] == delivery
        yield f"{profile}-{role}", earlier, submitted, truth, expected


def verify(payload, folder, earlier, submitted, truth, expected):
    import numpy as np
    import pymupdf
    from PIL import Image

    from app.detectors.image import reference_webp_color
    from scripts.review_image_reference_export import review
    from scripts.verify_multimedia_exports import scan

    comparison = payload["comparison"]
    states = expected["measurements"]
    measured = states["color_gamut"] if states else None
    regions = comparison["regions"]
    assert comparison["status"] == expected["status"] and comparison["affects_overall_confidence"] is False
    if measured:
        assert comparison["schema"] == reference_webp_color.SCHEMA and comparison["color_review"] == measured["proof"]
        for actual, wanted in zip(regions, measured["regions"], strict=True):
            assert {key: value for key, value in actual.items() if not key.endswith(("_crop", "_overlay"))} == wanted
    else:
        assert not regions and "color_review" not in comparison
    pdf = base64.b64decode(payload.pop("pdf"), validate=True)
    export = folder / "comparison.json"
    save(export, payload)
    source_replay = review(export, earlier, submitted)
    assert source_replay["source_hashes_verified"] == 2 and source_replay["crops_verified"] == 2 * len(regions)
    assert source_replay["mask_pixels_counted"] == (measured["selected"] if measured else 0)
    if regions:
        mask_bytes = base64.b64decode(payload["pngs"][comparison["mask"]["file"]], validate=True)
        with Image.open(io.BytesIO(mask_bytes), formats=["PNG"]) as image:
            mask = np.asarray(image) == 255
        assert sha(mask.tobytes()) == measured["mask_sha256"] and int((mask & truth).sum()) == measured["overlap"]
    with pymupdf.open(stream=pdf, filetype="pdf") as report:
        assert len(report) == len(regions) + 1 and report.metadata["author"] == "Chipmunk Verify"
        pages = [page.get_text() for page in report]
        scan(str(comparison) + "\n" + "\n".join(pages))
        for i, (page, region) in enumerate(zip(pages[1:], regions, strict=True), 1):
            if region.get("difference_basis") == "color":
                assert f"Color difference {i}" in page and "Brightness remains similar" in page
            hashes = [region[key]["sha256"] for key in ("reference_crop", "submitted_crop", "submitted_overlay")]
            assert all(digest in page for digest in hashes)
            for box in (region, region.get("reference_region", region)):
                assert f"x {box['x']}, y {box['y']}, {box['w']} x {box['h']} px" in page
    (folder / "comparison.pdf").write_bytes(pdf)
    return {"schema": comparison["schema"], "status": comparison["status"],
            "comparison_schema": comparison.get("comparison_schema"), "regions": len(regions),
            "color_regions": sum(region.get("difference_basis") == "color" for region in regions),
            "source_replay": source_replay}


def run(args):
    packet = panel.read(args.receipt)
    assert panel.review(packet) == packet["results"]
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_IMAGE_REFERENCE_LOSSY_WEBP="true", VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_CROP_RESIZE="true",
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
    torch.set_num_interop_threads(1)
    simulation = PaymentStub()
    service._stripe = simulation
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-webp-color-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for name, earlier, submitted, truth, expected in fixtures(packet, args.sources, args.cameras):
            folder = args.output / name
            folder.mkdir()
            ref, upload = folder / "earlier.png", folder / "submitted.webp"
            ref.write_bytes(earlier)
            upload.write_bytes(submitted)
            aid = checkout(client, simulation, upload, kind="image", media_type="image/webp", filename=upload.name)
            before = result_hash(aid)
            endpoint = f"/api/v2/analyses/{aid}/image-reference-comparison"
            assert body(client.get(endpoint))["available"]
            files = {"reference": (ref.name, earlier, "image/png")}
            if name == "same-small_recolor":
                settings.image_reference_lossy_webp_color = False
                baseline = body(client.post(endpoint, files=files))["comparison"]
                assert not baseline["regions"] and "color_review" not in baseline
                settings.image_reference_lossy_webp_color = True
            response = client.post(endpoint, files=files, data={"include_pdf": "true"})
            assert response.headers["cache-control"] == "no-store"
            results[name] = verify(body(response), folder, ref, upload, truth, expected)
            assert result_hash(aid) == before
            print(f"{name}: API, PDF and JSON replay verified", flush=True)
    save(args.output / "receipt.json", {"results": results, "paid_analyses": len(simulation.sessions), "unpaid_jobs": 0,
                                        "overall_analysis_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    run(parser.parse_args())
