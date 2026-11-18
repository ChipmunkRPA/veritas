# SPDX-License-Identifier: Apache-2.0
"""Replay verified speech-activity ranges in the local prepaid app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait
from scripts.video_silent_mouth_app import nested
from scripts.video_stereo_covariance_panel import read

ROOT = Path(__file__).resolve().parents[2]


def report(client, analysis, source, output, evidence, label):
    import math

    import numpy as np
    import pymupdf

    from app.detectors.audio.decode import decode_audio
    from app.evidence import video_activity_levels
    from app.evidence import video_viseme_binding as binding
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan, sha

    review = evidence[binding.KEY]
    witness = review["witness"]
    assert review["schema"] == binding.SCHEMA_WITH_ACTIVITY_LEVELS and review["affects_overall_confidence"] is False
    packet = next(entry for entry in evidence["packets"] if entry["id"] == binding.PACKET_ID)
    assert packet[binding.KEY] == witness and packet["affects_overall_confidence"] is False
    submitted = decode_audio(source, source_sha256=sha(source.read_bytes()), retain_stereo=True).stereo_pcm
    levels = witness[binding.ACTIVITY_KEY][video_activity_levels.KEY]
    assert levels[1]["source_samples"] == witness["audio_source_pts"]
    for row in levels:
        start, end = row["source_samples"]
        left, right = submitted[start:end].astype(np.float64).T / 32768
        reference = [round(10 * math.log10(power), 2) if power else None
                     for power in (np.mean(view**2) for view in (left, right, (left - right) / 2))]
        assert reference == row["rms_dbfs"]
    assets = {role: fetch(client, asset["url"]) for role, asset in packet["artifacts"].items()}
    assert all(sha(value) == packet["artifacts"][role]["sha256"] for role, value in assets.items())
    report_id = body(client.post(f"/api/v2/analyses/{analysis}/reports", json={"profile": "standard"}))["id"]
    wait(client, f"/api/v2/reports/{report_id}")
    pdf = fetch(client, f"/api/v2/reports/{report_id}/file")
    (output / f"{label}.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = " ".join(word for page in document for word in page.get_text().split())
        scan(text)
        assert binding.ACTIVITY_TITLE in text
        assert all(f"{role} reduced speech activity range" in text for role in ("Before", "During", "After"))
        assert all(name in text for name in video_activity_levels.CHANNELS)
        lo, hi = (value / 16000 for value in witness["audio_source_pts"])
        assert f"between {lo:.3f} and {hi:.3f} s" in text
        embedded = {pixels(document.extract_image(ref)["image"])
                    for ref in {image[0] for page in document for image in page.get_images(full=True)}}
        assert {pixels(data) for role, data in assets.items() if role != binding.CLIP} <= embedded
    package_id = body(client.post(f"/api/v2/analyses/{analysis}/packages", json={
        "report_id": report_id, "kind": "draft", "include_original": False}))["id"]
    wait(client, f"/api/v2/packages/{package_id}")
    archive = output / f"{label}.zip"
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package_id}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(archive, public_key_pem=public, source_video=source)
    assert checked.ok, checked.as_dict()
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence
    return {"range": witness["audio_source_pts"], "frames": witness["picture_frames"],
            "asset_count": len(assets), "source_replay": True, "pdf_sha256": sha(pdf), "stereo_levels": levels}


def process(held, media, assets, enhancer, output):
    results = read(held)
    by_name = {row["delivery"]: row for row in results["rows"]}
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="auto", VERITAS_JOB_TIMEOUT_S="360",
                      VERITAS_VIDEO_LIP_TIMING_REVIEW="false", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="false",
                      VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str(assets / "face_landmarker.task"),
                      VERITAS_VIDEO_LIP_TIMING_RUNTIME=str(assets / "runtime"))
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.video_viseme_binding import ACTIVITY_POINTS, KEY
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    stub = PaymentStub()
    assert {carrier: choices["selected"]["point"] for carrier, choices in results["decisions"].items()} == ACTIVITY_POINTS
    service._stripe = stub
    decisions, outcome, browser_case = {}, {}, None
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as session:
            owner = auth.new_principal("speech-activity-local-review", "examiner")
            session.add(owner)
            session.flush()
            key, _ = auth.issue_api_key(session, owner)
            session.commit()
        client.headers["X-API-Key"] = key
        for label, carrier, role, active in (("aac", "aac48", "hard", True), ("pcm", "pcm", "hard", True),
                                             ("ordinary", "aac48", "unmodified_mix", True),
                                             ("disabled", "aac48", "hard", False)):
            settings.video_speech_activity_gap_enhancer = enhancer if active else None
            row = by_name[f"source-08--5-{carrier}-{role}.mov"]
            source = media / row["delivery"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["input_sha256"]
            analysis = checkout(client, stub, source, filename=source.name, media_type="video/quicktime")
            evidence, exported = nested(client, analysis)
            scan(json.dumps(exported))
            decisions[label] = body(client.get(f"/api/v2/analyses/{analysis}"))["decision"]
            expected = row["candidate"]["both"]["reading"]
            outcome[label] = {"input_sha256": row["input_sha256"], "prepayment_jobs": 0, "paid_jobs": 1}
            if label in {"aac", "pcm"}:
                assert evidence[KEY]["status"] == "matched", evidence.get(KEY)
                assert evidence[KEY]["measurement"] == expected
                outcome[label].update(report(client, analysis, source, output, evidence, label))
                if label == "aac":
                    browser_case = {"analysis_id": analysis, "frames": expected["picture_frames"], "activity": True, "stereo_levels": True}
            elif label == "ordinary":
                assert evidence[KEY]["status"] == "no_match" and evidence[KEY]["witness"] is None
            else:
                assert KEY not in evidence
            print(label, outcome[label], flush=True)
        assert decisions["aac"] == decisions["disabled"]
        outcome["browser"] = browser_check(output, app, key, [browser_case], scenario="video-silent-mouth-local.mjs")
    (output / "receipt.json").write_text(json.dumps(outcome | {"decision_unchanged": True, "USD_cents": 999},
                                                   indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("held", "media", "assets", "enhancer", "output"):
        cli.add_argument(field, type=Path)
    process(**{name: path.resolve() for name, path in vars(cli.parse_args()).items()})
