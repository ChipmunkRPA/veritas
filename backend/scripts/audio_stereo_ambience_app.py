# SPDX-License-Identifier: Apache-2.0
"""Replay stereo difference evidence through local checkout and verifiable exports."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_native_alignment_panel import digest, iou, read, save
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def _examples(panel, output):
    records = {(row["role"], row["carrier"]): row for row in read(panel / "result.json.gz")["rows"] if row["group"] == "bus04"}
    pcm = panel / "bus04-wide_reduction.wav"
    aac = panel / "bus04-wide_reduction.m4a"
    control = panel / "bus04-untouched_mix.m4a"
    for path, role, carrier in ((pcm, "wide_reduction", "pcm"), (aac, "wide_reduction", "aac160"),
                                (control, "untouched_mix", "aac160")):
        if digest(path) != records[role, carrier]["delivery_sha256"]:
            raise ValueError("Example input differs from the held panel")
    video = output / "offset-soundtrack.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=8",
                    "-itsoffset", "0.4", "-i", str(aac), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "copy", str(video)], check=True, timeout=30)
    yield "stereo-wav", pcm, "audio", "audio/wav", records["wide_reduction", "pcm"]
    yield "stereo-aac", aac, "audio", "audio/mp4", records["wide_reduction", "aac160"]
    yield "unaltered-aac", control, "audio", "audio/mp4", records["untouched_mix", "aac160"]
    yield "offset-video", video, "video", "video/mp4", records["wide_reduction", "aac160"]


def _report(client, route, signer, evidence, measurement, original, name, output):
    import hashlib

    import pymupdf

    from app.evidence.stereo_difference_binding import KEY
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report_id = wait(client, f"/api/v2/reports/{report['id']}")["id"]
    pdf = fetch(client, f"/api/v2/reports/{report_id}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    assert "Brief narrowing of stereo sound" in words
    audio = evidence["audio"] if name == "offset-video" else evidence
    if measurement:
        assert "Stereo differences narrowed" in words and "Channel difference" in words and "Shared sound" in words
        prefix = "/audio" if name == "offset-video" else ""
        assert f"{prefix}/stereo_context/{KEY}/selected/0" in words
        start, end = (measurement[key] / 16000 for key in ("start_sample", "end_sample"))
        assert f"{start:.3f}–{end:.3f} s" in words
        if name == "offset-video":
            source = measurement["source_timestamp_ranges"][0]
            origin = evidence["source"]["timeline_origin_s"]
            start = source["source_start_pts"] / 16000 - origin
            end = source["source_end_pts"] / 16000 - origin
            assert f"Video playback {start:.3f}–{end:.3f} s." in words
        waveform = audio["stereo_listening"][KEY][0]["artifacts"]["waveform"]
        shape = pixels(fetch(client, waveform["url"]))
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            embedded = {((pix.width, pix.height), hashlib.sha256(pix.samples).hexdigest())
                        for page in document for image in page.get_images()
                        for pix in [pymupdf.Pixmap(document, image[0])] if pix.n == 3}
        assert shape in embedded
    else:
        assert "No qualifying reduction in stereo difference was found" in words
    (output / f"{name}.pdf").write_bytes(pdf)
    draft = body(client.post(f"{route}/packages", json={"report_id": report_id, "kind": "draft", "include_original": False}))
    draft_id = wait(client, f"/api/v2/packages/{draft['id']}")["id"]
    path = output / f"{name}.zip"
    path.write_bytes(fetch(client, f"/api/v2/packages/{draft_id}/file"))
    replay = verify_package(path, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                            source_audio=original)
    assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
    assert any("Recomputed the stereo difference reduction" in row.detail for row in replay.checks), replay.as_dict()
    with ZipFile(path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return {"signed_source_replay": True, "pdf_waveform_verified": bool(measurement)}


def run(panel, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_PUBLIC_PDF_LAYOUT="standard", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_ENVIRONMENT="dev", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import numpy as np
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import decode_audio, wav_bytes
    from app.engine.registry import get_registry
    from app.evidence.stereo_difference_binding import FLANK, KEY
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    get_registry().enabled = lambda *_: []  # Verify the evidence independently of learned detectors.
    stub = PaymentStub()
    service._stripe = stub
    rows = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("stereo-difference-local-trial", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, path, kind, mime, expected in _examples(panel, output):
            aid = checkout(client, stub, path, kind=kind, media_type=mime, filename=path.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for public in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(public.decode())
            audio = evidence["audio"] if kind == "video" else evidence
            check = audio["stereo_context"][KEY]
            assert check["affects_overall_confidence"] is False
            measurement = check["selected"][0] if check["selected"] else None
            assert bool(measurement) == bool(expected["truth"])
            if measurement:
                start, end = measurement["start_sample"], measurement["end_sample"]
                assert iou([start, end], expected["truth"]) >= .40
                if kind == "audio":
                    assert audio["stereo_context"]["stereo_pcm_sha256"] == expected["pcm_sha256"]
                    assert {key: measurement[key] for key in expected["measurements"]["reduction"]} == expected["measurements"]["reduction"]
                else:
                    source, = measurement["source_timestamp_ranges"]
                    assert source["source_start_pts"] - start > .30 * 16000
                    assert source["source_end_pts"] - end == source["source_start_pts"] - start
                decoded = decode_audio(path, source_sha256=digest(path), inspect_stereo=True)
                excerpt = decoded.stereo_pcm[start - FLANK:end + FLANK].astype(np.float32) / 32768
                assets = audio["stereo_listening"][KEY][0]["artifacts"]
                for role, data in (("left", excerpt[:, 0]), ("right", excerpt[:, 1]),
                                   ("difference", (excerpt[:, 0] - excerpt[:, 1]) / 2),
                                   ("mono", decoded.pcm[start - FLANK:end + FLANK])):
                    assert fetch(client, assets[role]["url"]) == wav_bytes(data)
            else:
                assert check["status"] == "no_qualifying_change" and KEY not in audio.get("stereo_listening", {})
            proof = _report(client, route, signer, evidence, measurement, path, name, output)
            rows.append({"case": name, "status": check["status"], "selected": measurement, "source_sha256": digest(path),
                         "matched_listening_wavs": 4 if measurement else 0, **proof})
            print(name, check["status"], flush=True)
    save(output / "receipt.json", {"schema": "chipmunk.stereo-ambience-local-checkout/1", "cases": rows,
                                   "simulated_payment_usd_cents": 999, "jobs_before_payment": 0, "voice_model_calls": 0})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
