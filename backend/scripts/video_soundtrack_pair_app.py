# SPDX-License-Identifier: Apache-2.0
"""Verify same-file soundtrack evidence through the local prepaid app and signed replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def run(bank, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_SOUNDTRACK_PAIR_REVIEW="true")

    import pymupdf
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.api import auth
    from app.billing import service
    from app.evidence.video_soundtrack_pair_binding import KEY, assets
    from app.main import app
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    stub = PaymentStub()
    service._stripe = stub
    tested = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-two-track-review", "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        for role, expected in (("later", "local_shift"), ("different_speech", "insufficient_same_time_support")):
            source = bank / f"aac-{role}.mp4"
            aid = checkout(client, stub, source, filename=source.name)
            assert stub.sessions[next(reversed(stub.sessions))]["amount_total"] == 999
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(exported.decode())
            assert json.loads(exported)["evidence"] == evidence
            case = evidence["audio"][KEY]
            assert case["status"] == expected and case["stream_indices"] == [1, 2] and case["affects_overall_confidence"] is False
            exhibits = {role: fetch(client, asset["url"]) for role, asset in assets(case).items()}
            assert all(hashlib.sha256(data).hexdigest() == assets(case)[role]["sha256"] for role, data in exhibits.items())
            with get_session() as db:
                row = db.get(Analysis, aid)
                result = json.loads(default_store().read_bytes(row.result_hash))
                frozen = result["manifest"]["sampling"]["video_audio_evidence"]["input"][KEY]
                assert KEY not in result["decision"]
            assert case["status"] == frozen["status"]
            scan(json.dumps(case))
            report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                words = " ".join(" ".join(page.get_text().split()) for page in document)
                scan(words)
                assert "Timing between audio tracks" in words and "The second track was unassessed" not in words
                if case["location"]:
                    assert len(exhibits) == 4 and "200.0 ms later" in words
                    originals = {Image.open(io.BytesIO(data)).convert("RGB").tobytes()
                                 for role, data in exhibits.items() if role.endswith("waveform")}
                    extracted = {Image.open(io.BytesIO(document.extract_image(xref)["image"])).convert("RGB").tobytes()
                                 for xref in {img[0] for page in document for img in page.get_images()}}
                    assert originals <= extracted
            (output / f"{role}.pdf").write_bytes(pdf)
            package = body(client.post(f"/api/v2/analyses/{aid}/packages", json={"report_id": report["id"],
                                 "kind": "draft", "include_original": False}))
            package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
            destination = output / f"{role}.zip"
            destination.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
            key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
            verify = verify_package(destination, public_key_pem=key, source_video=source, source_audio=source)
            assert verify.outcome == "verified", verify.as_dict()
            assert any(item.ok and "Recomputed the comparison of two audio streams" in item.detail for item in verify.checks)
            with ZipFile(destination) as archive:
                assert json.loads(archive.read("evidence.json")) == evidence
            location = case["location"]
            tested[role] = {"status": expected, "source_sha256": case["source_sha256"], "coverage": case["coverage"],
                            "shift_samples": location["shift_samples"] if location else None,
                            "source_intervals": [track["source_timestamp_ranges"][0] for track in location["excerpts"]]
                            if location else [], "exhibits_verified": len(exhibits), "original_replay_verified": True}
    receipt = {"simulated_paid_runs": len(tested), "unpaid_jobs": 0, "cases": tested}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.bank, args.output)
