# SPDX-License-Identifier: Apache-2.0
"""Exercise paid 48 kHz stereo repetition evidence in the local worker and exports."""

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
DESCRIPTION = "Times and excerpts use the original 48 kHz stereo recording."
AAC_DESCRIPTION = "This comparison uses lower frequency bands of the presented 48 kHz stereo audio."
RATE_DESCRIPTION = "The waveforms show the aligned excerpts at their recorded speeds on the source clock."


def fixtures(panel, output):
    rows = {row["delivery"]: row for row in read(panel / "result.json.gz")["rows"]}
    name = "NPARK-3-repeat_18db.wav"
    edited = panel / name
    control = panel / "NPARK-3-ordinary_panned.wav"
    for path in (edited, control):
        assert digest(path) == rows[path.name]["input_sha256"]
    expected = rows[name]["responses"]["native_48k"]["spans"]
    assert len(expected) == 1
    flac = output / "same-edit.flac"
    video = output / "offset-edit.mov"
    lossy = output / "same-edit.m4a"
    commands = [
        ["-i", str(edited), "-c:a", "flac", str(flac)],
        ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4.4",
         "-itsoffset", "0.4", "-i", str(edited), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(video)],
        ["-i", str(edited), "-c:a", "aac", str(lossy)],
    ]
    for command in commands:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *command], check=True, timeout=20)
    yield "wav", edited, "audio", "audio/wav", True, expected
    yield "flac", flac, "audio", "audio/flac", True, expected
    yield "video", video, "video", "video/quicktime", True, expected
    yield "ordinary", control, "audio", "audio/wav", True, []
    yield "aac", lossy, "audio", "audio/mp4", True, None
    yield "off", edited, "audio", "audio/wav", False, None


def aac_fixtures(panel, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.evidence.audio_aac_alignment import search

    rows = {row["delivery"]: row for row in read(panel / "result.json.gz")["rows"]}
    edited, compressed, ordinary = (panel / name for name in (
        "STRAFFIC-1-repeat_10db-160.m4a", "STRAFFIC-7-repeat_18db-96.m4a", "STRAFFIC-1-ordinary_panned-160.m4a"))
    for path in (edited, compressed, ordinary):
        assert digest(path) == rows[path.name]["input_sha256"]
    main = rows[edited.name]["responses"]["lower_band_stereo"]["spans"]
    quiet = rows[compressed.name]["responses"]["lower_band_stereo"]["spans"]
    assert len(main) == len(quiet) == 1 and not rows[ordinary.name]["responses"]["lower_band_stereo"]["spans"]
    video, resampled = output / "offset-edit.mp4", output / "same-edit-44100.m4a"
    commands = [
        ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4.4",
         "-itsoffset", "0.4", "-i", str(edited), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(video)],
        ["-i", str(edited), "-c:a", "aac", "-ar", "44100", str(resampled)],
    ]
    for command in commands:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *command], check=True, timeout=20)
    mapped = decode_audio(video, source_sha256=digest(video)).mapping
    data = decode_native_stereo(video, mapped, aac=True)
    _, reason, pairs = search(np.frombuffer(data, dtype="<i2").reshape(-1, 2))
    assert reason is None and len(pairs) == 1
    for pair, expected in zip(pairs[0], main[0], strict=True):
        assert all(abs(sample + mapped["source_start_pts"] - nominal - 19200) <= 480
                   for sample, nominal in zip(pair, expected, strict=True))
    yield "aac160", edited, "audio", "audio/mp4", True, main
    yield "aac96", compressed, "audio", "audio/mp4", True, quiet
    yield "offset-video", video, "video", "video/mp4", True, [[list(span) for span in pair] for pair in pairs]
    yield "ordinary", ordinary, "audio", "audio/mp4", True, []
    yield "different-rate", resampled, "audio", "audio/mp4", True, None
    yield "off", edited, "audio", "audio/mp4", False, None


def rate_fixtures(panel, output):
    rows = {row["delivery"]: row for row in read(panel / "result.json.gz")["rows"]}
    edited, second_edit, ordinary, same_speed = (panel / name for name in (
        "STRAFFIC-0-repeat_18db.wav", "STRAFFIC-3-repeat_10db.wav", "STRAFFIC-0-ordinary_mix.wav", "NFIELD-0-same_speed.wav"))
    for path in (edited, second_edit, ordinary, same_speed):
        assert digest(path) == rows[path.name]["input_sha256"]
    main, second, control, unchanged_speed = (rows[path.name]["matches"] for path in (edited, second_edit, ordinary, same_speed))
    assert len(main) == len(second) == 1 and not control and not unchanged_speed
    assert rows[same_speed.name]["baseline"]
    flac, video, resampled = (output / name for name in ("rate-edit.flac", "offset-rate-edit.mov", "rate-edit-44100.wav"))
    commands = [
        ["-i", str(edited), "-c:a", "flac", str(flac)],
        ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4.4", "-itsoffset", "0.4", "-i", str(second_edit),
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(video)],
        ["-i", str(edited), "-ar", "44100", str(resampled)],
    ]
    for command in commands:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *command], check=True, timeout=20)
    yield "wav", edited, "audio", "audio/wav", True, main
    yield "flac", flac, "audio", "audio/flac", True, main
    yield "offset-video", video, "video", "video/quicktime", True, second
    yield "ordinary", ordinary, "audio", "audio/wav", True, []
    yield "same-speed", same_speed, "audio", "audio/wav", True, []
    yield "different-rate", resampled, "audio", "audio/wav", True, None
    yield "off", edited, "audio", "audio/wav", False, None


def verify_exports(client, route, name, source_file, evidence, check, signer, output, origin, *, description):
    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    created = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report_id = wait(client, f"/api/v2/reports/{created['id']}")["id"]
    pdf = fetch(client, f"/api/v2/reports/{report_id}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    assert (description in words) == (check is not None)
    if check:
        charts = []
        for pair in check["pairs"]:
            if "faster_occurrence" in pair:
                speed = pair["duration_denominator"] / pair["duration_numerator"]
                assert f"The {pair['faster_occurrence']} occurrence plays about {speed:.2f}× as fast" in words
            for side in ("first", "second"):
                elapsed = pair[side]["source_timestamp_ranges"][0]
                a, b = elapsed["source_start_pts"], elapsed["source_end_pts"]
                assert f"Source PTS [{a}, {b}) at 1/48000." in words and elapsed["time_base"] == "1/48000"
                if origin is not None:
                    assert f"Video playback {a / 48000 - origin:.3f}–{b / 48000 - origin:.3f} s." in words
                asset = pair["artifacts"][f"{side}_recorded_clip"]
                clip = fetch(client, asset["url"])
                with wave.open(io.BytesIO(clip)) as original:
                    assert (original.getframerate(), original.getnchannels(), original.getsampwidth()) == (48000, 2, 2)
                    assert hashlib.sha256(original.readframes(original.getnframes())).hexdigest() == pair[side]["recorded_excerpt"]["pcm_s16le_sha256"]
                charts.append(pixels(fetch(client, pair["artifacts"][f"{side}_waveform"]["url"])))
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            images = {((pix.width, pix.height), hashlib.sha256(pix.samples).hexdigest())
                      for page in document for image in page.get_images()
                      for pix in [pymupdf.Pixmap(document, image[0])] if pix.n == 3}
        assert set(charts).issubset(images)
    (output / f"{name}.pdf").write_bytes(pdf)
    created = body(client.post(f"{route}/packages", json={"report_id": report_id, "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{created['id']}")
    target = output / f"{name}.zip"
    target.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    verified = verify_package(target, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                              source_audio=source_file)
    assert verified.outcome == "verified", verified.as_dict()
    if check:
        assert any("48 kHz stereo clock" in row.detail for row in verified.checks)
    with ZipFile(target) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return hashlib.sha256(pdf).hexdigest()


def run(panel, output, *, browser, aac=False, rate=False):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
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
    stub = PaymentStub()
    service._stripe = stub
    rows, browser_cases, decisions = [], [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-native-stereo-repeat", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = body(client.get("/api/v2/instance/public-key"))
        if rate:
            cases, description = rate_fixtures(panel, output), RATE_DESCRIPTION
        else:
            cases = aac_fixtures(panel, output) if aac else fixtures(panel, output)
            description = AAC_DESCRIPTION if aac else DESCRIPTION
        for name, media, kind, mime, enabled, expected in cases:
            settings.audio_stereo_aac_repeat_review = enabled and aac
            settings.audio_stereo_native_repeat_review = (enabled or rate) and not aac
            settings.audio_stereo_rate_repeat_review = enabled and rate
            aid = checkout(client, stub, media, kind=kind, media_type=mime, filename=media.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for public in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(public.decode())
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            decisions[name] = (result["decision"], result.get("customer_assessment"))
            audio = evidence if kind == "audio" else evidence["audio"]
            if expected is None:
                if rate and name == "off":
                    assert "rate_measurement" not in audio["stereo_repetition"]
                else:
                    assert "stereo_repetition" not in audio
                check = None
            else:
                packet = audio["stereo_repetition"]
                assert packet["affects_overall_confidence"] is False
                check = packet["rate_measurement" if rate else "aligned_measurement"]
                assert check["sample_rate"] == 48000
                if aac:
                    assert check["input_samples"] == audio["source"]["source_duration_pts"]
                    assert check["input_samples"] != audio["source"]["samples"] * 3
                bounds = check["pairs"]
                ranges = [[[pair[side]["start_sample"], pair[side]["end_sample"]] for side in ("first", "second")]
                          for pair in bounds]
                actual = ([{"first": first, "second": second, "faster": pair["faster_occurrence"],
                            "ratio": [pair["duration_numerator"], pair["duration_denominator"]]}
                           for (first, second), pair in zip(ranges, bounds, strict=True)] if rate else ranges)
                assert actual == expected
                if rate and name == "same-speed":
                    assert packet["aligned_measurement"]["pairs"] and not bounds
                if bounds:
                    browser_cases.append({"analysis_id": aid, "kind": kind, "name": name, "pairs": bounds,
                                          "description": description, "rate": rate})
            origin = float(evidence["source"]["timeline_origin_s"]) if kind == "video" else None
            if rate and origin is not None:
                for pair in check["pairs"]:
                    for side in ("first", "second"):
                        span = pair[side]
                        assert round(origin * 48000) + 19200 + span["start_sample"] == span["source_timestamp_ranges"][0]["source_start_pts"]
            report_sha = verify_exports(client, route, name, media, evidence, check, signer, output, origin,
                                        description=description)
            rows.append({"case": name, "source_sha256": digest(media), "measured": check is not None,
                         "ranges": expected, "source_times": [
                             [pair[s]["source_timestamp_ranges"] for s in ("first", "second")] for pair in check["pairs"]
                         ] if check else [], "pdf_sha256": report_sha})
            print(name, "verified", expected, flush=True)
        assert decisions["off"] == decisions["aac160" if aac else "wav"] and len(rows) == len(stub.sessions)
        browser_results = browser_check(output, app, key, browser_cases, scenario="native-stereo-repeat.mjs") if browser else None
    schema = "rate-stereo" if rate else "aac-stereo" if aac else "native-stereo"
    save(output / "receipt.json", {"schema": f"chipmunk.{schema}-local-checkout/1", "cases": rows,
                                   "same_assessment_with_option_off": True, "simulated_payment_usd_cents": 999,
                                   "jobs_before_payment": 0, "browser": browser_results})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--browser", action="store_true")
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--aac", action="store_true")
    modes.add_argument("--rate", action="store_true")
    run(**vars(cli.parse_args()))
