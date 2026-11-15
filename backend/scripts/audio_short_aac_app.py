# SPDX-License-Identifier: MIT
"""Exercise the short AAC review and signed exports through the local paid worker."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_short_aac_transfer import replay
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_downloads(client, aid, evidence, source, output):
    import pymupdf

    from scripts.audio_focused_window_app import signature
    from scripts.verify_multimedia_exports import scan
    from scripts.verify_short_speech_coverage import verify_coverage

    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    scan(exported.decode())
    assert json.loads(exported)["evidence"] == evidence
    review = evidence["short_voice_review"]
    wav, waveform = (fetch(client, review["artifacts"][role]["url"]) for role in ("selected_clip", "waveform"))
    assert sha(wav) == review["artifacts"]["selected_clip"]["sha256"]
    assert sha(waveform) == review["artifacts"]["waveform"]["sha256"]
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    start, end = (review["interval"][key] for key in ("start_sample", "end_sample"))
    with pymupdf.open(stream=pdf, filetype="pdf") as pages:
        text = " ".join(" ".join(page.get_text().split()) for page in pages)
        scan(text)
        assert f"Flagged window {start / 16000:.3f}–{end / 16000:.3f} s." in text
        assert f"Source PTS [{start}, {end}) at 1/16000." in text
        hashes = {signature(pages.extract_image(image[0])["image"]) for page in pages for image in page.get_images(full=True)}
        assert signature(waveform) in hashes
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
        json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    path = output / "signed-package.zip"
    path.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = output / "instance-public-key.pem"
    key.write_text(body(client.get("/api/v2/instance/public-key"))["public_key_pem"])
    coverage = verify_coverage(path, source, key, ROOT / "models/weights/audio-upgrade/silero_vad.jit")
    with ZipFile(path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence
    for name, data in (("export.json", exported), ("report.pdf", pdf), ("clip.wav", wav), ("waveform.png", waveform)):
        (output / name).write_bytes(data)
    return {"coverage": coverage["windows"], "signed_source_replay": True,
            "wav_sha256": sha(wav), "waveform_sha256": sha(waveform), "pdf_sha256": sha(pdf)}


def run(native, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_SHORT_ADMISSION_REVIEW="true")
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.audio_short_review_binding import AAC_REVIEW_POINT, SAMPLES
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = replay.load(native / "held.json")
    points = replay.load(native / "points.json")
    assert points["combined"]["1"] == AAC_REVIEW_POINT
    staged = next(row for row in rows if row["id"] == "held-11")
    ordinary = next(row for row in rows if row["group"] == staged["group"] and row["condition"] == "untouched")
    score, span = replay.strongest(staged["scores"]["aac"], "combined")
    assert score > AAC_REVIEW_POINT and span == (48000, 80000)
    transport = PaymentStub()
    service._stripe = transport
    summary, decisions = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-short-aac", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        cases = (("disabled", staged, "skipped"), ("staged_edit", staged, "flagged"), ("ordinary", ordinary, "no_flag"))
        for role, row, expected in cases:
            settings.audio_short_aac_review = role != "disabled"
            source = native / row["path"]
            assert sha(source.read_bytes()) == row["encoded_sha256"]
            aid = checkout(client, transport, source, kind="audio", media_type="audio/mp4", filename=source.name)
            public = fetch(client, f"/api/v2/analyses/{aid}/evidence-packets")
            scan(public.decode())
            packet = json.loads(public)
            review = packet["short_voice_review"]
            assert review["status"] == expected and review["affects_overall_confidence"] is False
            assert packet["source"]["samples"] == SAMPLES + 256
            with get_session() as db:
                snapshot = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
                decisions[role] = snapshot["decision"]
            proof = {"status": expected, "known_edit": row["truth"], "matched_interval": review["interval"],
                     "source_clock": review["source_timestamp_ranges"], "short_check_sample_limit": SAMPLES,
                     "native_presentation_end": packet["source"]["source_duration_pts"]}
            if expected == "flagged":
                assert review["interval"] == dict(zip(("start_sample", "end_sample"), span, strict=True))
                proof.update(check_downloads(client, aid, packet, source, output))
            summary[role] = proof
            print(f"Verified paid {role} analysis.", flush=True)
    assert decisions["disabled"] == decisions["staged_edit"]
    receipt = {"cases": summary, "usd_cents": 999, "jobs_before_payment": 0, "primary_assessment_preserved": True}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.native, args.output)
