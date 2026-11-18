# SPDX-License-Identifier: Apache-2.0
"""Exercise local prepaid single-face transitions and verify the original input."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import ALLOW, PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
POSITIVES = (
    ("tandem", "witness/tandem.mp4", (31, 32, 43), 1.280),
    ("direct", "witness-prefix-distinct/direct.mp4", (56, 57, 69), 2.280),
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def case_inputs(witnesses, controls):
    for name, relative, frames, time_s in POSITIVES:
        media = witnesses / relative
        witness = read(media.parent / "receipt.json")
        assert witness["reported_boundary"] == frames[1] and witness["input_sha256"] == digest(media.read_bytes())
        yield name, media, witness["input_sha256"], frames, time_s, "localized", None, witness
        if name == "tandem":
            yield "off", media, witness["input_sha256"], (), None, None, None, None
    manifest = read(controls / "receipt.json")
    for name, status, reason in (("ordinary", "no_match", None), ("hard-cut", "unavailable", "shot_change")):
        item = manifest[name]
        media = controls / item["file"]
        assert digest(media.read_bytes()) == item["sha256"]
        yield name, media, item["sha256"], (), None, status, reason, None


def packet_images(client, packet):
    from PIL import Image

    images = {}
    for frame in packet["frames"]:
        role, refs = frame["interval_role"], frame["artifact_refs"]
        assets = {name: packet["artifacts"][refs[name]] for name in ("original", "crop")}
        source_digest = frame["anchor"]["decoded_rgb_sha256"]
        original, crop = (fetch(client, entry["url"]) for entry in assets.values())
        assert digest(original) == assets["original"]["sha256"] and digest(crop) == assets["crop"]["sha256"]
        with Image.open(io.BytesIO(original)) as picture, Image.open(io.BytesIO(crop)) as exhibit:
            rgb = picture.convert("RGB")
            assert rgb.size == (640, 360) and digest(rgb.tobytes()) == source_digest
            region = frame["context_region"]
            x, y, width, height = (region[key] for key in ("x", "y", "w", "h"))
            assert exhibit.size == (width, height)
            cropped = rgb.crop((x, y, x + width, y + height))
            assert exhibit.convert("RGB").tobytes() == cropped.tobytes()
            for name, value in (("original", picture), ("crop", exhibit)):
                images[role, name] = (value.size, digest(value.convert("RGB").tobytes()))
    assert len(images) == 6
    return images


def verify_exports(client, aid, source, evidence, output, signer, *, packet, images, coverage_copy):
    import pymupdf
    from PIL import Image

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    route = f"/api/v2/analyses/{aid}"
    initialized = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{initialized['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        words = " ".join(" ".join(page.get_text().split()) for page in doc)
        scan(words)
        if coverage_copy is not None:
            assert coverage_copy in words
        if packet:
            assert "Face appearance transition" in words and "Surrounding picture similarity at the transition" in words
            similarity = packet["face_single_boundary"]["fine_scene"]
            assert f"Surrounding picture similarity at the transition {similarity} correlation" in words
            for frame in packet["frames"]:
                assert f"frame {frame['anchor']['frame_index']}" in words.casefold()
            extracted = set()
            for xref in {img[0] for page in doc for img in page.get_images(full=True)}:
                with Image.open(io.BytesIO(doc.extract_image(xref)["image"])) as image:
                    extracted.add((image.size, digest(image.convert("RGB").tobytes())))
            assert set(images.values()) <= extracted
        else:
            assert "VFB01" not in words and "Surrounding picture similarity at the transition" not in words
    output.mkdir()
    (output / "report.pdf").write_bytes(pdf)
    request = body(client.post(f"{route}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False,
    }))
    package = wait(client, f"/api/v2/packages/{request['id']}", timeout=120)
    signed = output / "signed-package.zip"
    signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    result = verify_package(signed, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                            source_video=source)
    assert result.outcome == "verified", result.as_dict()
    if packet:
        assert any(row.ok and row.name == "original video frames" for row in result.checks)
    with ZipFile(signed) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return {"pdf_sha256": digest(pdf), "matching_pdf_plates": len(images), "signed_original_replay": True}


def inspect_positive(client, packet, source, declared_frames, time_s, witness):
    from app.evidence.face_single_boundary_binding import KEY, MIN_SCENE, PACKET_ID, TITLE

    transition = packet[KEY]
    assert packet["id"] == PACKET_ID and packet["title"] == TITLE
    assert packet["affects_overall_confidence"] is transition["affects_overall_confidence"] is False
    assert packet["signal_refs"] == packet["finding_ids"] == []
    assert transition["boundary"] == declared_frames[1] and math.isclose(transition["time_s"], time_s, abs_tol=1e-6)
    assert transition["pts"] == witness["reported_pts"] and transition["time_base"] == witness["time_base"]
    assert transition["fine_scene"] >= MIN_SCENE
    assert round(float.fromhex(transition["readings_hex"]["fine_scene"]), 6) == transition["fine_scene"]
    assert {"label": "Surrounding picture similarity at the transition",
            "value": transition["fine_scene"], "unit": "correlation"} in packet["measurements"]
    assert "cause" in packet["interpretation"] and "zero weight" in " ".join(packet["limitations"])
    assert tuple(frame["anchor"]["frame_index"] for frame in packet["frames"]) == declared_frames
    for frame in packet["frames"]:
        role = frame["interval_role"]
        anchor = frame["anchor"]
        expected = witness["verified_source_frames"][role]
        assert anchor["original_sha256"] == source["sha256"]
        assert anchor["pts"] == expected["pts"] and anchor["decoded_rgb_sha256"] == expected["decoded_rgb_sha256"]
        crop = transition["source_frames"][role]
        assert [[float.fromhex(value) for value in eye] for eye in crop["eyes_hex"]] == expected["eyes"]
        assert crop["aligned_rgb_sha256"] == expected["aligned_rgb_sha256"]
    return packet_images(client, packet)


def run(witnesses, controls, output, *, browser):
    cases = list(case_inputs(witnesses, controls))
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
                      VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="false", VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_REVIEW="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.face_single_boundary_binding import COVERAGE, KEY, PACKET_ID, coverage_text
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert ALLOW.issubset(enabled("video", "full"))
    registry.enabled = lambda modality, profile: [name for name in enabled(modality, profile) if name in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    rows, browser_cases, decisions = [], [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-single-face-transition", "examiner")
            db.add(owner)
            db.flush()
            access, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = access
        signer = body(client.get("/api/v2/instance/public-key"))
        assert settings.video_face_single_boundary_review is False
        for name, media, source_sha, frames, time_s, status, reason, witness in cases:
            settings.video_face_single_boundary_review = name != "off"
            aid = checkout(client, stub, media, filename=media.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            export = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(export)["evidence"] == evidence and evidence["source"]["sha256"] == source_sha
            for public in (export, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(public.decode())
            coverage = evidence["coverage"].get(COVERAGE)
            if status is None:
                assert coverage is None
            else:
                assert coverage["status"] == status and coverage["reason"] == reason
                if status in {"localized", "no_match"}:
                    assert coverage["frames_scanned"] == 100 and coverage["measured_face_frames"] >= 38
            found = [item for item in evidence["packets"] if KEY in item or item["id"] == PACKET_ID]
            assert len(found) == int(bool(frames))
            packet = found[0] if found else None
            images = inspect_positive(client, packet, evidence["source"], frames, time_s, witness) if packet else {}
            with get_session() as db:
                analysis = db.get(Analysis, aid)
                result = read(default_store().path_for(analysis.result_hash))
            decisions[name] = (result["decision"], result.get("customer_assessment"))
            signals = [item for item in result.get("signals", []) if item["detector_id"] == "vid.review.face_appearance"]
            assert all(item["effective_weight"] == 0 and item["calibrated_probability"] is None for item in signals)
            message = coverage_text(coverage) if coverage else None
            proof = verify_exports(client, aid, media, evidence, output / name, signer, packet=packet,
                                   images=images, coverage_copy=message)
            browser_cases.append({"id": aid, "name": name, "status": status, "reason": reason,
                                  "background": packet[KEY]["fine_scene"] if packet else None,
                                  "packet": PACKET_ID, "frames": [] if packet is None else [
                                      {"index": row["anchor"]["frame_index"], "seek": row["anchor"]["playback_time_s"]}
                                      for row in packet["frames"]]})
            rows.append({"name": name, "source_sha256": source_sha, "coverage": coverage,
                         "source_frames": list(frames), "transition_time_s": time_s,
                         "background_similarity": packet[KEY]["fine_scene"] if packet else None,
                         "original_crop_pixels_verified": len(images), **proof})
            print(name, status or "disabled", list(frames), flush=True)
        assert decisions["off"] == decisions["tandem"] and len(stub.sessions) == len(rows)
        browser_results = browser_check(output, app, access, browser_cases, scenario="single-face-transition.mjs") if browser else None
    receipt = {"schema": "chipmunk.face-single-boundary-local-checkout/1", "cases": rows, "browser": browser_results,
               "confidence_parity": True, "job_created_before_paid": False, "simulated_payment_usd_cents": 999}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witnesses", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.witnesses, args.controls, args.output, browser=args.browser)


if __name__ == "__main__":
    main()
