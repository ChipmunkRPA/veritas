"""Verify the local small-copy check through prepaid analysis and signed evidence."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts import image_small_affine_jpeg as study
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def fixtures(receipt):
    assert study.review(receipt) == receipt["summary"]
    entries = list(zip(receipt["freeze"]["rows"], receipt["observed"], strict=True))
    ordinary = {(row["group"], row["quality"]): row for row, _ in entries if row["role"] == "original"}
    for long_edge, quality in ((768, 75), (1024, 90)):
        row, selected = next(
            (row, result["proposed"][-1]) for row, result in entries
            if max(row["size"]) == long_edge and row["quality"] == quality and result["proposed"]
            and (detail := result["proposed"][-1]["detail"]) and detail[0] <= study.MAX_COLOR_ERROR
            and detail[1] >= study.MIN_DETAIL and study.useful(result["proposed"][-1], result["truth_pixels"])
        )
        yield row, selected
        yield ordinary[row["group"], quality], None


def report(client, identifier, source, evidence, pair, output):
    import pymupdf
    from PIL import Image

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    record = body(client.post(f"/api/v2/analyses/{identifier}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{record['id']}")
    pdf = fetch(client, f"/api/v2/reports/{record['id']}/file")
    output.with_suffix(".pdf").write_bytes(pdf)
    assert pair["id"] == "G01"
    expected_images = set()
    with Image.open(source) as original:
        for area, bounds in pair["regions"].items():
            x, y, w, h = (bounds[key] for key in "xywh")
            raw_crop = fetch(client, pair["artifacts"][area]["url"])
            mask = fetch(client, pair["artifacts"]["mask_" + area[-1]]["url"])
            expected_images.update(pixels(data) for data in (raw_crop, mask))
            with Image.open(io.BytesIO(raw_crop)) as crop, Image.open(io.BytesIO(mask)) as selected:
                assert crop.tobytes() == original.crop((x, y, x+w, y+h)).tobytes()
                assert selected.mode == "L" and selected.size == crop.size == (w, h)
                assert selected.histogram()[255] == pair["selected_pixels"][area]
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        assert "Geometric copy comparison" in words and "Matching detail was found after geometric alignment." in words
        for bounds in pair["regions"].values():
            assert f"x {bounds['x']}, y {bounds['y']}, {bounds['w']} × {bounds['h']} px" in words
        images = {pixels(document.extract_image(image)["image"])
                  for image in {row[0] for page in document for row in page.get_images()}}
        assert expected_images <= images and len(expected_images) == 4
    package = body(client.post(f"/api/v2/analyses/{identifier}/packages",
                               json={"report_id": record["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = output.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    proof = verify_package(archive, public_key_pem=key, source_image=source)
    assert proof.ok, proof.as_dict()
    assert any(check.ok and "Recomputed 2 geometric copy masks" in check.detail for check in proof.checks)
    with ZipFile(archive) as verified:
        assert json.loads(verified.read("evidence.json")) == evidence and verified.read("report.pdf") == pdf
    return {"pdf_sha256": study.sha(pdf), "native_crop_and_mask_images": 4, "original_source_replay": True}


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_IMAGE_AFFINE_COPY_REVIEW="true", VERITAS_LOG_LEVEL="ERROR")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(identifier):
        with get_session() as db:
            result = json.loads(default_store().read_bytes(db.get(Analysis, identifier).result_hash))
        return result["decision"], result.get("customer_assessment")

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    replay, cases = {}, []
    receipt = study.read(args.receipt)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-small-image-copy-review", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for row, matches in fixtures(receipt):
            path = args.inputs / row["file"]
            assert study.sha(path.read_bytes()) == row["sha256"]
            ids, packages = [], []
            for enabled in (False, True):
                settings.image_small_affine_copy_review = enabled
                identifier = checkout(client, stub, path, kind="image", media_type="image/jpeg", filename=path.name)
                ids.append(identifier)
                packages.append(body(client.get(f"/api/v2/analyses/{identifier}/evidence-packets")))
            baseline, evidence = packages
            check = evidence["copy_affine"]
            export = fetch(client, f"/api/analyses/{ids[1]}/export.json")
            scan(export.decode())
            assert json.loads(export)["image_evidence"] == evidence
            assert baseline["copy_affine"]["status"] == "no_pair" and baseline["version"] == evidence["version"]
            assert decision(ids[0]) == decision(ids[1])
            earlier = body(client.get(f"/api/v2/analyses/{ids[0]}/evidence-packets", params={"set_id": baseline["id"]}))
            assert earlier == baseline
            note = {"source_sha256": row["sha256"], "profile": row["quality"], "status": check["status"],
                    "earlier_readable": True, "decision_preserved": True}
            assert check["authorship_confidence_contribution"] == 0
            if matches:
                assert check["status"] == "supported_pair"
                pair, = check["pairs"]
                assert pair["matrix"] == matches["matrix"]
                assert list(pair["selected_pixels"].values()) == matches["pixels"]
                x0, y0, x1, y1 = matches["seed"]
                assert pair["seed"] == {"x": x0, "y": y0, "w": x1-x0, "h": y1-y0}
                note["regions"] = pair["regions"]
                note.update(report(client, ids[1], path, evidence, pair, args.output / path.stem))
                cases.append({"analysis_id": ids[1], "name": path.stem, "pair": pair})
            else:
                assert check["status"] == "no_pair" and not check["pairs"]
            replay[path.stem] = note
            print(path.name, check["status"], flush=True)
        browser = browser_check(args.output, app, key, cases) if args.browser else None
    study.write(args.output / "receipt.json", {"cases": replay, "browser": browser, "unpaid_jobs": 0,
                                               "simulated_paid_analyses": len(stub.sessions),
                                               "comparison_sha256": study.sha(args.receipt.read_bytes())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "receipt", "inputs"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
