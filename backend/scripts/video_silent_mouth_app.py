# SPDX-License-Identifier: Apache-2.0
"""Verify the near-silent soundtrack finding in the prepaid local app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Mouth movement with a near-silent soundtrack"


def nested(client, analysis):
    evidence = body(client.get(f"/api/v2/analyses/{analysis}/evidence-packets"))
    exported = body(client.get(f"/api/analyses/{analysis}/export.json"))
    assert evidence["status"] == "ready" and exported["evidence"] == evidence
    return evidence, exported


def proof_files(client, analysis, path, output, evidence, expected):
    import pymupdf

    from app.evidence import video_viseme_binding as binding
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan, sha

    review = evidence[binding.KEY]
    assert review["status"] == "matched" and review["schema"] == binding.SCHEMA_WITH_SILENCE
    assert review["measurement"] == expected["measurements"] and review["witness"]["picture_frames"] == expected["frames"]
    [item] = [entry for entry in evidence["packets"] if entry["id"] == binding.PACKET_ID]
    assert item.get("title") == TITLE and item["affects_overall_confidence"] is False
    assert item[binding.KEY] == review["witness"]
    assert "change_samples" not in item[binding.KEY]
    assets = {role: fetch(client, asset["url"]) for role, asset in item["artifacts"].items()}
    assert all(sha(value) == item["artifacts"][role]["sha256"] for role, value in assets.items())
    report = body(client.post(f"/api/v2/analyses/{analysis}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    (output / "silent-interval.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = " ".join(word for page in document for word in page.get_text().split())
        scan(text)
        assert TITLE in text and all(f"{role} near-silent range" in text for role in ("Before", "During", "After"))
        lo, hi = (value / 16000 for value in review["witness"]["audio_source_pts"])
        assert f"near-silent from {lo:.3f} to {hi:.3f} s" in text
        embedded = {pixels(document.extract_image(ref)["image"])
                    for ref in {image[0] for page in document for image in page.get_images(full=True)}}
        assert {pixels(data) for role, data in assets.items() if role != binding.CLIP} <= embedded
    package = body(client.post(f"/api/v2/analyses/{analysis}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = output / "silent-interval.zip"
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(archive, public_key_pem=public, source_video=path)
    assert checked.ok, checked.as_dict()
    replay = next(check.detail for check in checked.checks if check.name == "original video frames")
    assert "timing review's soundtrack" in replay
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence
    return {"range": review["witness"]["audio_source_pts"], "frames": expected["frames"],
            "asset_count": len(assets), "source_replay": True, "pdf_sha256": sha(pdf)}


def choose_fixture(rows):
    index = {(row["group"], row["delivery"], row["role"]): row for row in rows}
    for row in rows:
        if row["role"] in {"muted", "muted_faded"} and row["delivery"] == "aac48" and row["reading"]["added"]:
            control = index[row["group"], row["delivery"], "original"]
            if not control["reading"]["baseline"] and not control["reading"]["added"]:
                return row, control
    raise ValueError("No positive AAC fixture has a matching clean control")


def process(panel, assets, output):
    import gzip

    selected, clean = choose_fixture(json.loads(gzip.decompress((panel / "result.json.gz").read_bytes()))["rows"])
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="auto", VERITAS_JOB_TIMEOUT_S="360",
                      VERITAS_VIDEO_LIP_TIMING_REVIEW="true", VERITAS_VIDEO_LIP_TIMING_RANKED="true",
                      VERITAS_VIDEO_LIP_ARTICULATION_REVIEW="true", VERITAS_VIDEO_LIP_AAC48_ARTICULATION_REVIEW="true",
                      VERITAS_VIDEO_LIP_SILENT_MOUTH_REVIEW="true", VERITAS_VIDEO_LIP_TIMING_MODEL_DIR=str(assets / "checkpoint"),
                      VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str(assets / "face_landmarker.task"),
                      VERITAS_VIDEO_LIP_TIMING_RUNTIME=str(assets / "runtime"))
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.video_viseme_binding import KEY, SCHEMA_WITH_SILENCE
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    stub = PaymentStub()
    service._stripe = stub
    decisions, outcome = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as session:
            owner = auth.new_principal("near-silent-soundtrack-smoke", "examiner")
            session.add(owner)
            session.flush()
            key, _ = auth.issue_api_key(session, owner)
            session.commit()
        client.headers["X-API-Key"] = key
        for title, active, row in (("positive", True, selected), ("control", True, clean), ("disabled", False, selected)):
            settings.video_lip_silent_mouth_review = active
            source = panel / "playable" / row["playable"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["playable_sha256"]
            analysis = checkout(client, stub, source, filename=source.name, media_type="video/quicktime")
            evidence, exported = nested(client, analysis)
            decisions[title] = body(client.get(f"/api/v2/analyses/{analysis}"))["decision"]
            sounded = evidence.get(KEY, {}).get("schema") == SCHEMA_WITH_SILENCE
            assert sounded == (title == "positive"), evidence.get(KEY, {}).get("status")
            outcome[title] = {"audio_finding": sounded, "prepayment_jobs": 0, "paid_jobs": 1}
            if sounded:
                scan(json.dumps(exported))
                (output / "evidence.json").write_text(json.dumps(exported, indent=2) + "\n")
                outcome[title].update(proof_files(client, analysis, source, output, evidence, row["reading"]))
                outcome["browser"] = browser_check(
                    output, app, key, [{"analysis_id": analysis, "frames": outcome[title]["frames"]}],
                    scenario="video-silent-mouth-local.mjs")
            print(title, outcome[title], flush=True)
    assert decisions["positive"] == decisions["disabled"]
    (output / "receipt.json").write_text(json.dumps(outcome | {"decision_unchanged": True, "USD_cents": 999}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "assets", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    process(args.panel.resolve(), args.assets.resolve(), args.output.resolve())
