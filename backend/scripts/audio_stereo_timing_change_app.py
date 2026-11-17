# SPDX-License-Identifier: Apache-2.0
"""Check stereo timing evidence through local checkout, analysis, and exports."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import wave
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_native_alignment_panel import digest, read, save
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
HEADING = "Temporary change in stereo timing"


def local_model_view(output):
    """Share installed weight files while keeping verification records in this test."""
    installed = ROOT / "models"
    view = output.resolve() / "models"
    (view / "weights").mkdir(parents=True)
    for path in installed.iterdir():
        if path.name != "weights":
            (view / path.name).symlink_to(path)
    for weight in (installed / "weights").iterdir():
        if not weight.name.startswith(".veritas-verified.json"):
            (view / "weights" / weight.name).symlink_to(weight)
    return view


def cases(panel, output):
    from app.detectors.audio.decode import decode_audio

    records = {row["delivery"]: row for row in read(panel / "result.json.gz")["rows"]}
    stem = "DWASHING-0-alternate_voice_with_delay_change"
    wav, aac96, aac160, ordinary, unclear = (panel / name for name in (
        f"{stem}.wav", f"{stem}-aac96.m4a", f"{stem}-aac160.m4a", "DWASHING-0-ordinary-aac160.m4a",
        "DWASHING-0-recorded_environment.wav",
    ))
    for path in (wav, aac96, aac160, ordinary, unclear):
        assert digest(path) == records[path.name]["input_sha256"]
    video, different_rate = output / "offset-edit.mp4", output / "different-rate.m4a"
    commands = (
        ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4.4", "-itsoffset", "0.4", "-i", str(aac160),
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(video)],
        ["-i", str(aac160), "-c:a", "aac", "-ar", "44100", str(different_rate)],
    )
    for command in commands:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *command], check=True, timeout=20)
    mapped = decode_audio(video, source_sha256=digest(video)).mapping
    assert 17500 < mapped["source_start_pts"] < 19200 and mapped["source_duration_pts"] > 180000
    for name, file, mime in (("wav", wav, "audio/wav"), ("aac96", aac96, "audio/mp4"), ("aac160", aac160, "audio/mp4"),
                             ("ordinary", ordinary, "audio/mp4"), ("unassessed", unclear, "audio/wav")):
        yield name, file, "audio", mime, records[file.name]
    yield "offset-video", video, "video", "video/mp4", records[aac160.name]
    yield "different-rate", different_rate, "audio", "audio/mp4", None
    yield "off", aac160, "audio", "audio/mp4", None


def verify_exports(client, route, name, original, evidence, check, signer, output):
    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report_id = wait(client, f"/api/v2/reports/{report['id']}")["id"]
    pdf = fetch(client, f"/api/v2/reports/{report_id}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    assert (HEADING.casefold() in words.casefold()) == (check is not None)
    charts = []
    if check is not None:
        if check["measurement"]:
            origin = evidence["source"]["timeline_origin_s"] if name == "offset-video" else None
            for clip in check["clips"]:
                asset = check["artifacts"][clip["role"]]
                data = fetch(client, asset["url"])
                assert hashlib.sha256(data).hexdigest() == asset["sha256"]
                with wave.open(io.BytesIO(data)) as wav:
                    assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (48000, 2, 2, asset["samples"])
                    assert hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() == clip["pcm_s16le_sha256"]
                time, = clip["source_timestamp_ranges"]
                low, high = time["source_start_pts"], time["source_end_pts"]
                assert f"Source PTS [{low}, {high}) at 1/48000." in words
                if origin is not None:
                    assert f"Video playback {low / 48000 - origin:.3f}–{high / 48000 - origin:.3f} s." in words
            charts = [pixels(fetch(client, check["artifacts"][role]["url"])) for role in ("timing_plot", "waveform")]
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                embedded = {((pix.width, pix.height), hashlib.sha256(pix.samples).hexdigest())
                            for page in document for image in page.get_images()
                            for pix in [pymupdf.Pixmap(document, image[0])] if pix.n == 3}
            assert set(charts) <= embedded
        elif check["status"] == "measured":
            assert "No qualifying temporary delay change was found" in words
        else:
            assert "Too little sound was shared clearly" in words
    (output / f"{name}.pdf").write_bytes(pdf)
    draft = body(client.post(f"{route}/packages", json={"report_id": report_id, "kind": "draft", "include_original": False}))
    package_id = wait(client, f"/api/v2/packages/{draft['id']}")["id"]
    package = output / f"{name}.zip"
    package.write_bytes(fetch(client, f"/api/v2/packages/{package_id}/file"))
    result = verify_package(package, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                            source_audio=original)
    assert result.outcome == "verified", result.as_dict()
    if check is not None:
        assert any("Recomputed the stereo timing change" in row.detail for row in result.checks)
    with ZipFile(package) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return {"pdf_sha256": hashlib.sha256(pdf).hexdigest(), "pdf_plots_verified": len(charts),
            "signed_source_replay": True}


def run(panel, output, *, browser):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
                      VERITAS_MODELS_DIR=str(local_model_view(output)), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.audio_stereo_timing_binding import KEY
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    stub = PaymentStub()
    service._stripe = stub
    rows, decisions, browser_cases = [], {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-stereo-timing", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, media, kind, mime, expected in cases(panel, output):
            settings.audio_stereo_timing_change_review = name != "off"
            # Verify both coexistence and independence from stereo repeat evidence.
            settings.audio_stereo_native_repeat_review = True
            settings.audio_stereo_aac_repeat_review = name != "aac96"
            aid = checkout(client, stub, media, kind=kind, media_type=mime, filename=media.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for public in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(public.decode())
            with get_session() as db:
                stored = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            decisions[name] = (stored["decision"], stored.get("customer_assessment"))
            audio = evidence if kind == "audio" else evidence["audio"]
            assert ("stereo_repetition" in audio) == (name not in {"different-rate", "aac96"})
            check = audio.get(KEY)
            assert (check is not None) == (expected is not None)
            if check is not None:
                assert check["affects_overall_confidence"] is False
                assert (check["status"] == "measured") == expected["assessed"]
                if kind == "audio":
                    assert check["measurement"] == expected["measurement"]
                    assert check["stereo_pcm_sha256"] == expected["presented_pcm_sha256"]
                else:
                    inside = check["clips"][1]["source_timestamp_ranges"][0]
                    for role in ("start", "end"):
                        nominal = expected["measurement"][f"{role}_sample"] + 19200
                        assert abs(inside[f"source_{role}_pts"] - nominal) <= 480
                if check["measurement"]:
                    browser_cases.append({"analysis_id": aid, "name": name, "clips": check["clips"],
                                          "video_origin_s": evidence["source"].get("timeline_origin_s")})
            report = verify_exports(client, route, name, media, evidence, check, signer, output)
            rows.append({"case": name, "source_sha256": digest(media), "status": check["status"] if check else "absent",
                         "measurement": check["measurement"] if check else None,
                         "clips": check["clips"] if check else [], **report})
            print(name, rows[-1]["status"], "browser case" if check and check["measurement"] else "", flush=True)
        assert decisions["off"] == decisions["aac160"] and len(rows) == len(stub.sessions)
        browser_result = browser_check(output, app, key, browser_cases, scenario="stereo-timing-change.mjs") if browser else None
    save(output / "receipt.json", {"schema": "chipmunk.stereo-timing-local-checkout/1", "cases": rows,
                                   "same_assessment_with_option_off": True, "simulated_payment_usd_cents": 999,
                                   "jobs_before_payment": 0, "browser": browser_result})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--browser", action="store_true")
    run(**vars(cli.parse_args()))
