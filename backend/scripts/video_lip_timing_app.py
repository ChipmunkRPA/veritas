# SPDX-License-Identifier: MIT
"""Replay the local prepaid timing trial using pinned publisher recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

SOURCE = "s41_p_bbbt7p"


def fixtures(output, sources, receipt):
    from scripts import video_lip_viseme_transfer as study
    from scripts.video_lip_event_alignment import aac_clocked

    row = next(item for item in study.load(receipt)["rows"] if item["source"] == SOURCE)
    payloads = {}
    for role in ("audio", "front"):
        member = row["members"][role]
        with ZipFile(sources / f"lombardgrid_{role}.zip") as archive:
            payloads[role] = archive.read(member["name"])
        if hashlib.sha256(payloads[role]).hexdigest() != member["sha256"]:
            raise ValueError(f"Publisher {role} input differs from the held receipt")
    picture = output / "source.mov"
    picture.write_bytes(payloads["front"])
    waveforms = study.parent.treatments(study.phone.pcm(payloads["audio"]), row["edit"])
    for role in ("late", "original"):
        aac = output / f"{role}.m4a"
        aac_clocked(waveforms[role], aac)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(picture), "-i", str(aac),
                        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(output / f"{role}.mp4")], check=True)
    return picture


def run(args):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    output = args.output.resolve()
    configure(output, Path(__file__).resolve().parents[2])
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_LIP_TIMING_REVIEW="true",
                      VERITAS_VIDEO_LIP_TIMING_MODEL_DIR=str(args.checkpoint.resolve()),
                      VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str(args.geometry.resolve()),
                      VERITAS_VIDEO_LIP_TIMING_RUNTIME=str(args.runtime.resolve()))
    if args.standard_checks:
        os.environ.update(VERITAS_DEVICE="auto", VERITAS_JOB_TIMEOUT_S="480")
    picture = fixtures(output, args.sources, args.receipt)

    import pymupdf
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence import video_viseme_binding as binding
    from app.main import app
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import pixels, scan, sha

    def decision(aid):
        with get_session() as db:
            result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return result["decision"], result.get("customer_assessment")

    if not args.standard_checks:
        registry = get_registry()
        enabled = registry.enabled
        registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile)
                                                       if registry.detector_family(key) == "provenance"]
    stub = PaymentStub()
    service._stripe = stub
    outcomes = []
    off = None
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("speech-timing-smoke", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        cases = [(False, output / "late.mp4"), (True, output / "late.mp4"),
                 (True, output / "original.mp4"), (True, picture)]
        for active, source in cases:
            settings.video_lip_timing_review = active
            media_type = "video/quicktime" if source.suffix == ".mov" else "video/mp4"
            aid = checkout(client, stub, source, filename=source.name, media_type=media_type)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = body(client.get(f"/api/analyses/{aid}/export.json"))
            assert evidence["status"] == "ready" and evidence == exported["evidence"], evidence.get("status")
            scan(json.dumps(exported))
            review = evidence.get(binding.KEY)
            status = review["status"] if review else "off"
            print(source.name, status, flush=True)
            proof = {"file_role": source.stem, "sha256": sha(source.read_bytes()), "status": status}
            matched = [item for item in evidence["packets"] if item["id"] == binding.PACKET_ID]
            assert bool(matched) == (status == "matched")
            if not active:
                assert status == "off"
                off = decision(aid)
            elif source == picture:
                assert status == "unsupported"
            elif source.stem == "original":
                assert status == "no_match", status
            else:
                assert status == "matched" and decision(aid) == off, status
                [packet] = matched
                witness = packet[binding.KEY]
                assert witness == {"audio_source_pts": [10880, 22400], "picture_source_samples": [8320, 19840],
                                   "baseline_lag_samples": 0, "change_samples": 2560,
                                   "picture_frames": {"before": 12, "during": 21, "after": 30}}
                assets = {role: fetch(client, asset["url"]) for role, asset in packet["artifacts"].items()}
                assert all(sha(data) == packet["artifacts"][role]["sha256"] for role, data in assets.items())
                report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
                wait(client, f"/api/v2/reports/{report['id']}")
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                (output / "timing.pdf").write_bytes(pdf)
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    words = " ".join(word for page in document for word in page.get_text().split())
                    scan(words)
                    words = words.replace("–", "-")
                    assert all(text in words for text in ("Local speech and mouth timing change", "0.680-1.400 s", "0.520-1.240 s",
                                                          "Before timing range", "During timing range", "After timing range")), words[-4000:]
                    embedded = {pixels(document.extract_image(ref)["image"])
                                for ref in {image[0] for page in document for image in page.get_images(full=True)}}
                    assert {pixels(data) for role, data in assets.items() if role != binding.CLIP} <= embedded
                package = body(client.post(f"/api/v2/analyses/{aid}/packages", json={
                    "report_id": report["id"], "kind": "draft", "include_original": False}))
                wait(client, f"/api/v2/packages/{package['id']}")
                archive = output / "timing.zip"
                archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
                checked = verify_package(archive, public_key_pem=public, source_video=source)
                assert checked.ok, checked.as_dict()
                replay = next(check.detail for check in checked.checks if check.name == "original video frames")
                assert "timing review's soundtrack" in replay
                with ZipFile(archive) as bundle:
                    assert json.loads(bundle.read("evidence.json")) == evidence
                    assert bundle.read(f"assets/{packet['artifacts'][binding.CLIP]['sha256']}.wav") == assets[binding.CLIP]
                proof.update(witness=witness, verified_assets=len(assets), source_replay=replay,
                             pdf_sha256=sha(pdf), same_assessment_as_off=True)
            outcomes.append(proof)
    (output / "receipt.json").write_text(json.dumps({"paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                                     "payment_mode": "simulated USD 9.99", "standard_checks": args.standard_checks,
                                                     "outcomes": outcomes}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "sources", "receipt", "checkpoint", "geometry", "runtime"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--standard-checks", action="store_true", help="Also run the configured Full video detectors")
    run(parser.parse_args())
