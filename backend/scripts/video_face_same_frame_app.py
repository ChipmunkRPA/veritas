# SPDX-License-Identifier: Apache-2.0
"""Exercise both face checks through an isolated prepaid analysis and its exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.video_face_paired_edge_hevc_app import image_signature, sha, sha_path, write_json
from scripts.video_face_temporal_fit import read
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait
from scripts.writing_character_view_app import tokens_assessed

CHECK = "vid.review.whole_face"
ADDITIONS = ("09-inswapper-tandem.mp4", "20-uniface-tandem.mp4")
CONTROL = "09-ordinary-tandem.mp4"
EXISTING = "03-uniface-direct.mp4"


def verify_deliveries(client, aid, source, evidence, packet, output):
    import pymupdf

    from app.detectors.video.decode import decode_frames, probe_frames
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    output.mkdir(mode=0o700)
    table = probe_frames(source)
    indices = [row["anchor"]["frame_index"] for row in packet["frames"]]
    native = {frame.index: frame for frame in decode_frames(source, table, indices)}
    signatures = set()
    for frame, index in zip(packet["frames"], indices, strict=True):
        pixel = native[index]
        anchor = frame["anchor"]
        assert pixel.pts_verified and anchor["pts"] == table.pts[index]
        assert anchor["decoded_rgb_sha256"] == sha(pixel.rgb.tobytes())
        x, y, w, h = (frame["context_region"][key] for key in "xywh")
        for role, expected in (("original", ((table.width, table.height), anchor["decoded_rgb_sha256"])),
                               ("crop", ((w, h), sha(pixel.rgb[y:y+h, x:x+w].tobytes())))):
            asset = packet["artifacts"][frame["artifact_refs"][role]]
            image = fetch(client, asset["url"])
            assert sha(image) == asset["sha256"] and image_signature(image) == expected
            signatures.add(expected)
    assert len(signatures) == 6

    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    scan(exported.decode())
    assert json.loads(exported)["evidence"] == evidence
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        assert "Both readings were combined at each jointly measured face." in words
        assert "Planned native frame indices 14, 28, 42, 57, 71, 85." in words
        xrefs = {item[0] for page in document for item in page.get_images(full=True)}
        assert signatures <= {image_signature(document.extract_image(xref)["image"]) for xref in xrefs}
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                               json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    zip_path = output / "signed-package.zip"
    zip_path.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = body(client.get("/api/v2/instance/public-key"))
    replay = verify_package(zip_path, public_key_pem=key["public_key_pem"].encode(),
                            expect_key_id=key["key_id"], source_video=source)
    assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
    assert any(result.name == "original video frames" and result.ok for result in replay.checks), replay.as_dict()
    with ZipFile(zip_path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read("report.pdf") == pdf
    (output / "report.pdf").write_bytes(pdf)
    (output / "export.json").write_bytes(exported)
    return {"six_exact_images": True, "pdf_api_json_agree": True, "signed_source_verification": True,
            "source_samples": indices, "evidence_window_s": [packet["window"][f"{k}_stream_s"] for k in ("start", "end")]}


def run(clips, receipt, output, device):
    protocol = read(receipt)
    held = {row["file"]: row for row in protocol["held"]}
    output = output.resolve()
    configure(output, Path(__file__).resolve().parents[2])
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE=device, VERITAS_VIDEO_WHOLE_FACE_REVIEW="true",
                      VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence import whole_face_binding as binding
    from app.main import app
    from app.storage.db import get_session

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert CHECK in enabled("video", "full")
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key == CHECK]
    assert settings.data_dir == output / "app" and settings.video_whole_face_review
    stub = PaymentStub()
    service._stripe = stub
    cases, original_assessments = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            examiner = auth.new_principal("same-frame-local-test", "examiner")
            db.add(examiner)
            db.flush()
            key, _ = auth.issue_api_key(db, examiner)
            db.commit()
        client.headers["X-API-Key"] = key
        for name, on in [(name, False) for name in ADDITIONS] + [(name, True) for name in (*ADDITIONS, CONTROL, EXISTING)]:
            settings.video_whole_face_same_frame_review = on
            source = clips / name
            original = held[name]
            assert sha_path(source) == original["sha256"]
            aid = checkout(client, stub, source)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == original["sha256"]
            [packet] = [packet for packet in evidence["packets"] if packet.get("whole_face_review")]
            record = packet["whole_face_review"]
            new_positive = on and name in ADDITIONS
            assert bool(record["flagged"]) == (original["reference"]["flagged"] or new_positive)
            assert record["schema"] == (binding.SAME_FRAME_METHOD if new_positive else binding.METHOD)
            assert packet["affects_overall_confidence"] is False
            assessment = tokens_assessed(aid)
            if not on:
                original_assessments[name] = assessment
            elif new_positive:
                assert assessment == original_assessments[name]
            log_key = f"{source.stem}-{'on' if on else 'off'}"
            case = {"sha256": original["sha256"], "flagged": record["flagged"], "method": record["schema"]}
            print(log_key, json.dumps(case, sort_keys=True), flush=True)
            if new_positive:
                case.update(verify_deliveries(client, aid, source, evidence, packet, output / log_key))
            cases[log_key] = case
    write_json(output / "receipt.json", {"cases": cases, "unpaid_jobs": 0, "simulated_paid_jobs": len(cases),
                                         "overall_assessments_unchanged": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()
    run(args.clips, args.receipt, args.output, args.device)
