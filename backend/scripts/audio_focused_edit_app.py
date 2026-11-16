# SPDX-License-Identifier: MIT
"""Replay the frozen focused speech fit through prepaid uploads and signed exports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from zipfile import ZipFile

from app.evidence import audio_focused_edit_binding as binding
from scripts.audio_focused_edit_carrier import render
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def cases(study):
    def row(panel, kind):
        return next(row for row in study["rows"] if (row["panel"], row["kind"], row["group"]) == (panel, kind, 7))

    loud = row("vc_-10dB", "edited/ChatterboxVC")
    yield "louder-edit", loud, True, "wav", "flagged"
    yield "louder-disabled", loud, False, "wav", None
    yield "moderate-edit", row("vc_-5dB", "edited/ChatterboxVC"), True, "flac", "flagged"
    yield "ordinary", row("vc_-10dB", "untouched"), True, "wav", "no_flag"
    yield "isolated-background", row("vc_-10dB", "background"), True, "wav", "no_flag"
    flagged = next(row for row in study["rows"] if row["panel"] == "vc_dry" and row["current"]["range"] is not None)
    yield "existing-flag", flagged, True, "wav", None
    yield "extra-sample", loud, True, "wav", None


def fixture(output, media, name, row, container, lossless_48k):
    pin = row["pcm_sha256"]
    raw = (media / "pcm" / f"{pin}.pcm").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin and len(raw) == 2 * binding.SAMPLES
    if lossless_48k:
        import io

        from app.detectors.audio.decode import decode_audio

        with tempfile.TemporaryDirectory(dir=output) as scratch:
            assets = {path.suffix[1:]: path for path in render(raw, Path(scratch))}
            file = output / f"{name}.{container}"
            if name == "extra-sample":
                with wave.open(str(assets["wav"]), "rb") as stream:
                    params, pcm = stream.getparams(), stream.readframes(stream.getnframes())
                with wave.open(str(file), "wb") as stream:
                    stream.setparams(params)
                    stream.writeframes(pcm + pcm[-2:])
            else:
                assets[container].replace(file)
        frozen = decode_audio(file, source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), max_seconds=7)
        with wave.open(io.BytesIO(frozen.wav), "rb") as stream:
            return file, stream.readframes(stream.getnframes())
    wav = output / f"{name}.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, binding.RATE, binding.SAMPLES, "NONE", "not compressed"))
        stream.writeframes(raw + raw[-2:] if name == "extra-sample" else raw)
    if container == "wav":
        return wav, raw
    target = wav.with_suffix(".flac")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "flac", str(target)], check=True)
    return target, raw


def verify_report(client, route, file, evidence, signer, raw):
    import io

    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    result = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{result['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{result['id']}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    assert binding.TITLE in words
    review = evidence[binding.KEY]
    assert review["reason"] in words
    if review["status"] == "flagged":
        a, b = (review["interval"][key] for key in ("start_sample", "end_sample"))
        assert f"{a / 16000:.3f}–{b / 16000:.3f} s" in words
        if source_range := review.get("original_timestamp_range"):
            assert source_range == binding.original_range(a, b)
            assert f"Original samples [{3 * a}, {3 * b}) at 48000 Hz." in words
            assert f"Decoded samples [{a}, {b}) at 16000 Hz." in words
        else:
            assert f"Source PTS [{a}, {b}) at 1/16000." in words
        clip = fetch(client, review["artifacts"]["selected_clip"]["url"])
        with wave.open(io.BytesIO(clip), "rb") as stream:
            assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (16000, 1, 2)
            assert stream.readframes(stream.getnframes()) == raw[2 * a:2 * b]
        chart = pixels(fetch(client, review["artifacts"]["waveform"]["url"]))
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            assert chart in {((pix.width, pix.height), hashlib.sha256(pix.samples).hexdigest())
                             for page in document for image in page.get_images()
                             for pix in [pymupdf.Pixmap(document, image[0])] if pix.n == 3}
    record = body(client.post(f"{route}/packages", json={"report_id": result["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{record['id']}", timeout=120)
    file.with_suffix(".pdf").write_bytes(pdf)
    archive = file.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{record['id']}/file"))
    verified = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(),
                              expect_key_id=signer["key_id"], source_audio=file)
    assert verified.outcome == "verified", verified.as_dict()
    if review["status"] == "flagged":
        assert any("Recomputed the marked focused speech waveform" in check.detail for check in verified.checks)
    with ZipFile(archive) as document:
        assert json.loads(document.read("evidence.json")) == evidence
        assert document.read("report.pdf") == pdf
    return hashlib.sha256(pdf).hexdigest()


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_SHORT_ADMISSION_REVIEW="true", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_AUDIO_SHORT_LOSSLESS_48K_REVIEW=str(args.lossless_48k).lower(),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    study = json.loads(gzip.decompress(args.study.read_bytes()))
    stub = PaymentStub()
    service._stripe = stub
    observed, assessments, browser_cases = [], {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-focused-speech-edit-review", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, row, enabled, container, expected in cases(study):
            file, raw = fixture(args.output, args.media, name, row, container, args.lossless_48k)
            settings.audio_focused_edit_review = enabled or args.lossless_48k
            settings.audio_focused_edit_lossless_48k_review = enabled and args.lossless_48k
            aid = checkout(client, stub, file, kind="audio", media_type=f"audio/{container}", filename=file.name)
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
                assessments[name] = (result["decision"], result.get("customer_assessment"))
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for public in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                content = public.decode()
                scan(content)
                assert binding.DETECTOR_ID not in content and "score_units" not in content
            previous, focused = evidence["short_voice_review"], evidence.get(binding.KEY)
            if name == "extra-sample":
                if args.lossless_48k:
                    assert evidence["source"]["source_duration_pts"] == 6 * binding.LOSSLESS_RATE + 1
                else:
                    assert evidence["source"]["samples"] == binding.SAMPLES + 1
                assert previous["status"] == "skipped"
            else:
                assert previous["status"] == ("flagged" if row["current"]["range"] is not None else "no_flag")
            assert (focused["status"] if focused else None) == expected
            report_sha = None
            if focused:
                assert focused["affects_overall_confidence"] is False
                assert focused["schema"] == (binding.RESAMPLED_SCHEMA if args.lossless_48k else binding.SCHEMA)
                binding.validate(focused, evidence["source"])
                if expected == "flagged":
                    a, b = row["candidates"]["both"]["range"]
                    assert focused["interval"] == {"start_sample": a, "end_sample": b}
                    browser_cases.append({"analysis_id": aid, "name": name, "review": focused})
                if expected == "flagged" or name == "ordinary":
                    report_sha = verify_report(client, route, file, evidence, signer, raw)
            observed.append({"case": name, "native_samples": evidence["source"]["samples"],
                             "panel_pcm_sha256": row["pcm_sha256"],
                             "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                             "primary": previous["status"], "focused": expected, "interval": focused["interval"] if focused else None,
                             "checked_windows": focused["checked_windows"] if focused else None,
                             "signed_pdf_sha256": report_sha,
                             **({"original_timestamp_range": focused.get("original_timestamp_range") if focused else None,
                                 "original_sample_count": evidence["source"]["source_duration_pts"]}
                                if args.lossless_48k else {})})
            print(name, previous["status"], expected, flush=True)
        assert assessments["louder-edit"] == assessments["louder-disabled"]
        browser = browser_check(args.output, app, token, browser_cases, scenario="focused-speech-replay.mjs") if args.browser else None
    assert len(stub.sessions) == len(observed)
    output = {"schema": "chipmunk.focused-speech-edit-app/1", "cases": observed, "same_assessment_with_option_off": True,
              "simulated_payment_usd_cents": 999, "jobs_before_payment": 0, "browser": browser}
    (args.output / "receipt.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "media", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--lossless-48k", action="store_true")
    run(parser.parse_args())
