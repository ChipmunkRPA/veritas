# SPDX-License-Identifier: MIT
"""Replay the local paid app with restored image and video checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.image_whole_jpeg_transfer import ROOT, evaluate, read, require, sha, sources, write_delivery
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

FACE_INPUTS = ("rtfs-13-original_videos", "rtfs-13-inswapper", "rtfs-13-uniface")
FACE_MANIFEST = "a86cc49c7b0a40815e0e5921fdf1ccb7260f174d6d937d4e6ed2d684223977eb"
FACE_SEMANTICS = "paired-whole-face-samples/1"


def fixtures(images, videos, output, compression, modality):
    if compression:
        from PIL import Image

        reference = read(compression)
        require(evaluate(reference) == reference["outcomes"], "The documented compression test does not replay")
        measured = next(row for row in reference["samples"]
                        if row["label"] == "synthetic" and row["deliveries"]["15"]["mean"] <= reference["point"])
        source = next(row for row in sources(images) if row["source_index"] == measured["index"])
        delivery = output / "heavily-compressed.jpg"
        with Image.open(source["input_path"], formats=["JPEG"]) as image:
            write_delivery(image, delivery, 15)
        digest = sha(delivery)
        require(digest == measured["deliveries"]["15"]["input_sha256"], "The tested JPEG bytes differ")
        yield "image", "compressed-generated", delivery, digest
        return
    if modality in {"both", "video"}:
        manifest = videos / "manifest.json"
        require(sha(manifest) == FACE_MANIFEST, "Earlier matched video inputs changed")
        face_rows = {row["id"]: row["matched"] for row in read(manifest)["items"] if row["id"] in FACE_INPUTS}
        for name in FACE_INPUTS:
            row = face_rows[name]
            source = videos / row["path"]
            require(sha(source) == row["sha256"], "The signed video input differs from the matched roster")
            yield "video", name, source, row["sha256"]
    if modality == "video":
        return
    matches = {}
    for row in sources(images):
        matches.setdefault(row["label"], row)
    for label in ("ordinary", "synthetic"):
        row = matches[label]
        yield "image", label, Path(row["input_path"]), row["sha256"]


def visual_receipt(client, aid, kind, source, original_sha, *, compression=False):
    import pymupdf
    from PIL import Image

    from app.reports.verify import verify_package
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import pixels, scan

    public = body(client.get(f"/api/analyses/{aid}/export.json"))
    scan(json.dumps(public))
    if kind == "video":
        evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
        require(evidence == public["evidence"] and evidence["status"] == "ready", "Video evidence surfaces differ")
        [packet] = [p for p in evidence["packets"] if p["semantics"] == FACE_SEMANTICS]
        measure = packet["whole_face_review"]
        require(packet["affects_overall_confidence"] is False and len(measure["samples"]) == 6 and len(packet["frames"]) == 3,
                "Video restored check did not retain six source positions")
        selected = [packet["artifacts"][frame["artifact_refs"][role]]
                    for frame in packet["frames"] for role in ("original", "crop")]
        heading = "Planned native frame indices"
        outcome = {"flagged": measure["flagged"], "planned_source_indices": packet["window"]["frame_indices"],
                   "source_time_base": measure["time_base"]}
    else:
        review = public["whole_image_review"]
        require(review["source_sha256"] == original_sha and review["authorship_confidence_contribution"] == 0
                and review["status"] in {"flagged", "no_flag"}, "Image restored check did not run")
        selected = review["crops"]
        require(len(selected) == (3 if review["status"] == "flagged" else 0), "Image views are incomplete")
        heading = "Supplementary whole-image check"
        outcome = {"flagged": review["status"] == "flagged", "source_crop_bounds": [row["bounds"] for row in selected]}
        if compression:
            from app.evidence.image_whole_review import COMPRESSION_LIMIT

            require(review["status"] == "no_flag" and COMPRESSION_LIMIT.strip() in review["interpretation"],
                    "The known missed image does not show the compression limitation")
            heading = COMPRESSION_LIMIT.strip()
            outcome["compression_note_verified"] = True
    assets = [fetch(client, item["url"]) for item in selected]
    require(all(hashlib.sha256(data).hexdigest() == item["sha256"] for item, data in zip(selected, assets, strict=True)),
            "An API image payload differs from the recorded hash")
    if kind == "image" and selected:
        from scripts.image_smaug_views_v7 import views

        with Image.open(source) as image:
            expected, _ = views(image, 3)
        require(all(pixels(data) == (crop.size, hashlib.sha256(crop.tobytes()).hexdigest())
                    for crop, data in zip(expected, assets, strict=True)), "An inspected view differs from the exact source pixels")
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(word for page in document for word in page.get_text().split())
        scan(words)
        require(heading in words, "The PDF does not show the restored check")
        extracted = {pixels(document.extract_image(ref)["image"])
                     for ref in {item[0] for page in document for item in page.get_images(full=True)}}
        require({pixels(data) for data in assets} <= extracted, "The PDF images differ from the inspected source views")
    signed = body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{signed['id']}")
    exported = fetch(client, f"/api/v2/packages/{signed['id']}/file")
    package_sha256 = hashlib.sha256(exported).hexdigest()
    archive = default_store().path_for(package_sha256)
    require(sha(archive) == package_sha256, "The stored signed package differs from its API export")
    instance = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    replay = verify_package(archive, public_key_pem=instance, **{f"source_{kind}": source})
    require(replay.ok, f"Signed source verification failed for {kind}")
    return outcome | {"source_sha256": original_sha, "api_pdf_images_verified": len(assets), "signed_source_verified": True}


def run(output, images, videos, compression, modality):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_WHOLE_FACE_REVIEW="true",
                      VERITAS_IMAGE_WHOLE_REVIEW_PYTHON=str(ROOT / "tmp/image-smaug-v7-env/bin/python"),
                      VERITAS_DEVICE="auto", VERITAS_JOB_TIMEOUT_S="150")

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.main import app
    from app.storage.db import get_session

    stub = PaymentStub()
    service._stripe = stub
    receipts = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("visual-coverage-replay", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for kind, name, source, digest in fixtures(images, videos, output, compression, modality):
            mime = "video/mp4" if kind == "video" else "image/jpeg"
            aid = checkout(client, stub, source, kind=kind, media_type=mime, filename=f"{kind}-evidence{source.suffix}")
            receipt = visual_receipt(client, aid, kind, source, digest, compression=compression is not None)
            receipts.append({"case": name, "kind": kind, **receipt})
            print(f"Verified {kind} check for {name}.", flush=True)
    with (output / "receipt.json").open("x") as stream:
        json.dump({"payment": "simulated USD 9.99", "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                   "cases": receipts}, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("output", "images", "videos"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--compression-receipt", type=Path)
    parser.add_argument("--modality", choices=("both", "image", "video"), default="both")
    args = parser.parse_args()
    if args.compression_receipt and args.modality == "video":
        parser.error("The compression receipt selects an image case")
    run(args.output.resolve(), args.images, args.videos, args.compression_receipt, args.modality)
