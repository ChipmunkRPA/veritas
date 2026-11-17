"""Replay an earlier HEIC and edited JPEG through the prepaid local workflow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

from scripts import image_reference_perspective_panel as shared
from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check

ROOT = Path(__file__).resolve().parents[2]


def held_cases(panel):
    receipt = shared.load(panel / "receipt.json.gz")
    point = str(receipt["point"])
    edited = next(row for row in receipt["rows"] if row["role"] == "feather" and row["geometry"] == "tilted_crop"
                  and row["quality"] == 75 and len(row["observations"][point]["regions"]) == 1
                  and any(shared.useful(area["intersection"], area["selected"], row["truth_pixels"])
                          for area in row["observations"][point]["regions"]))
    key = ("source", "geometry", "quality")
    ordinary = next(row for row in receipt["rows"] if row["role"] == "ordinary"
                    and all(row[k] == edited[k] for k in key))
    assert ordinary["observations"][point] == {"assessed": True, "regions": []}
    files = []
    for role, row in (("edit", edited), ("ordinary", ordinary)):
        upload = panel / row["file"]
        reference = upload.with_name("earlier.heic")
        assert shared.file_sha(upload) == row["jpeg_sha256"] and shared.file_sha(reference) == row["earlier_sha256"]
        files.append((role, upload, reference, row["observations"][point]["regions"]))
    return files


def verify_outputs(payload, role, earlier, upload, output):
    import pymupdf
    from PIL import Image

    from scripts.review_image_reference_perspective_export import review
    from scripts.verify_multimedia_exports import scan

    record = payload["comparison"]
    scan(json.dumps(record))
    export = output / f"{role}.json"
    export.write_text(json.dumps({"comparison": record, "pngs": payload["pngs"]}))
    source_replay = review(export, earlier, upload)
    assert source_replay["source_hashes_verified"] == 2
    pdf = base64.b64decode(payload["pdf"], validate=True)
    (output / f"{role}.pdf").write_bytes(pdf)

    def pixels(image):
        rgb = image.convert("RGB")
        return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()

    crops = [region[key] for region in record["regions"] for key in ("reference_crop", "submitted_crop")]
    target_pixels = set()
    for crop in crops:
        data = base64.b64decode(payload["pngs"][crop["file"]], validate=True)
        assert hashlib.sha256(data).hexdigest() == crop["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            target_pixels.add(pixels(image))

    with pymupdf.open(stream=pdf, filetype="pdf") as report:
        assert len(report) == 1 + len(record["regions"])
        assert report.metadata["author"] == "Chipmunk Verify"
        words = " ".join(" ".join(page.get_text().split()) for page in report)
        scan(words)
        if record["status"] == "comparable":
            assert "Matched HEIC and JPEG framing. Crops use sRGB display colors." in words
        assert all(crop["sha256"] in words for crop in crops)
        for region in record["regions"]:
            for box in (region, region["reference_region"]):
                assert f"x {box['x']}, y {box['y']}, {box['w']} x {box['h']} px" in words
        embedded = set()
        for page in report:
            for xref, *_ in page.get_images():
                with Image.open(io.BytesIO(report.extract_image(xref)["image"])) as image:
                    embedded.add(pixels(image))
        assert target_pixels <= embedded
    return {"status": record["status"], "regions": len(record["regions"]), "pdf_crops_verified": len(target_pixels),
            "reference_sha256": record["reference"]["sha256"], "submitted_sha256": record["submitted"]["sha256"],
            "verified_assets": source_replay["verified_assets"]}


def run(panel, output, camera, camera_jpeg):
    files = held_cases(panel) + [("camera_export", camera_jpeg, camera, [])]
    paid.configure(output, ROOT)
    models = local_model_view(output)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED="true", VERITAS_IMAGE_REFERENCE_HEIF_REVIEW="true",
                      VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"),
                      VERITAS_MODELS_DIR=str(models), VERITAS_HF_HOME=str(models / "hf"))

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.image import reference_heif
    from app.main import app
    from app.storage.db import Analysis, get_session

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub

    def result_hash(aid):
        with get_session() as db:
            return db.get(Analysis, aid).result_hash

    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-earlier-heic", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for role, upload, earlier, expected_regions in files:
            aid = paid.checkout(client, stub, upload, kind="image", media_type="image/jpeg", filename=upload.name)
            before = result_hash(aid)
            endpoint = f"/api/v2/analyses/{aid}/image-reference-comparison"
            source = {"reference": (earlier.name, earlier.read_bytes(), "image/heic")}
            assert paid.body(client.get(endpoint)) == {"available": True, "earlier_heif": True}
            if role == "edit":
                settings.image_reference_heif_review = False
                try:
                    assert paid.body(client.get(endpoint)) == {"available": True}
                    assert client.post(endpoint, files=source).status_code == 422
                finally:
                    settings.image_reference_heif_review = True
            response = client.post(endpoint, files=source, data={"include_pdf": "true"})
            assert response.headers["cache-control"] == "no-store"
            payload = paid.body(response)
            comparison = payload["comparison"]
            assert comparison["schema"] == reference_heif.SCHEMA and comparison["affects_overall_confidence"] is False
            assert comparison["status"] == "comparable"
            actual = [([region[k] for k in "xywh"], region["selected_pixels"]) for region in comparison["regions"]]
            assert actual == [(row["box"], row["selected"]) for row in expected_regions]
            results[role] = verify_outputs(payload, role, earlier, upload, output)
            assert before == result_hash(aid)
            if role == "edit":
                wrong = next(path for path in sorted((panel / "deliveries").glob("*/earlier.heic")) if path != earlier)
                unrelated = paid.body(client.post(endpoint, files={"reference": (wrong.name, wrong.read_bytes(), "image/heic")},
                                                  data={"include_pdf": "true"}))
                assert unrelated["comparison"]["status"] == reference_heif.UNASSESSED and unrelated["pngs"] == {}
                results["unrelated"] = verify_outputs(unrelated, "unrelated", wrong, upload, output)
                fixture = {"analysis_id": aid, "reference": str(earlier), "comparison": comparison,
                           "native_reference_hash": results[role]["reference_sha256"]}
            print(role, results[role], flush=True)
        browser = browser_check(output, app, key, [fixture], scenario="earlier-heic.mjs")
    assert len(stub.sessions) == len(files)
    (output / "receipt.json").write_text(json.dumps({"results": results, "browser": browser,
        "unpaid_jobs": 0, "simulated_usd_cents": 999, "simulated_paid_analyses": len(files),
        "earlier_heic_without_option_status": 422, "saved_result_preserved": True}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "output", "camera", "camera_jpeg"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
