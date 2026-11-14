"""Verify local mouth-hold sound across a real audio timestamp gap."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "3e1a8ed6bb5ea52b81de03b8dd7a15ecf863a4e1ee22e5e48dd8fbcbb8d63dae"
RANGE = (52_800, 59_200)
START_DECODED = 41_600
EXPECTED_FRAMES = [98, 105, 111]
CASES = {"before": 40_960, "inside": 49_152, "after": 65_536}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def deliver(source, output, split):
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
        "-f", "lavfi", "-i", "sine=frequency=150:sample_rate=16000:duration=7",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-af", f"asetnsamples=n=4096:p=0,asetpts=PTS+8000+if(gte(N\\,{split})\\,3200\\,0)",
        "-c:a", "alac", "-video_track_timescale", "30000", str(output),
    ], timeout=30, check=True)


def independent_pcm(source):
    return subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "-",
    ], timeout=30)


def check_report(client, aid, evidence, source, packet, output, positive):
    import pymupdf

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    assets = packet["artifacts"]
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    roles = {frame["artifact_refs"][role] for frame in packet["frames"] for role in ("original", "crop")}
    if positive:
        roles.add("mouth_hold_waveform")
    images = []
    for role in roles:
        exhibit = fetch(client, assets[role]["url"])
        assert sha(exhibit) == assets[role]["sha256"]
        images.append(pixels(exhibit))
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        assert ("Mono soundtrack during these measured pictures (source time)" in words) == positive
        xrefs = {image[0] for page in document for image in page.get_images(full=True)}
        assert set(images) <= {pixels(document.extract_image(xref)["image"]) for xref in xrefs}
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                              json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    archive = output / f"{source.stem}.zip"
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    pem = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    report = verify_package(archive, public_key_pem=pem, source_video=source)
    assert report.outcome == "verified" and any(row.ok and row.name == "original video frames" for row in report.checks)
    with ZipFile(archive) as saved:
        assert json.loads(saved.read("evidence.json")) == evidence and saved.read("report.pdf") == pdf
    return len(images)


def run(source, output):
    assert sha(source.read_bytes()) == SOURCE
    configure(output, ROOT)

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import decode_audio, wav_bytes
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key == "vid.faces.yunet"]
    service._stripe = stub = PaymentStub()
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-mouth-clock-trial", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for name, split in CASES.items():
            path = output / f"{name}.mp4"
            deliver(source, path, split)
            digest = sha(path.read_bytes())
            decoded = decode_audio(path, source_sha256=digest, max_seconds=120, inspect_stereo=True, retain_stereo=True)
            assert decoded.mapping["timeline"] == [
                {"start_sample": 0, "end_sample": split, "source_pts": 8000},
                {"start_sample": split, "end_sample": 112_000, "source_pts": 11_200 + split},
            ]
            started = time.monotonic()
            aid = checkout(client, stub, path)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(exported.decode())
            assert json.loads(exported)["evidence"] == evidence
            [packet] = [item for item in evidence["packets"] if item.get("mouth_local_hold")]
            assert [frame["anchor"]["frame_index"] for frame in packet["frames"]] == EXPECTED_FRAMES
            positive = name != "inside"
            result = {"timeline": decoded.mapping["timeline"], "source_sha256": digest,
                      "source_frames": EXPECTED_FRAMES, "source_picture_range_s": [3.3, 3.7],
                      "sound_measured": positive, "weight_in_overall_confidence": 0}
            if positive:
                sound = packet["mouth_hold_sound"]
                start = START_DECODED if name == "before" else 44_800
                stop = start + 6_400
                assert sound["source_pts"] == list(RANGE) and sound["samples"] == [start, stop]
                assert sound["sustained_periodicity"]
                wav = fetch(client, packet["artifacts"]["mouth_hold_audio"]["url"])
                assert sha(wav) == packet["artifacts"]["mouth_hold_audio"]["sha256"]
                assert wav == wav_bytes(decoded.pcm[start:stop])
                assert wav[44:] == independent_pcm(path)[2 * start:2 * stop]
                assert packet["artifacts"]["mouth_hold_waveform"]["time_axis"] == "source_pts"
                result.update(source_pts=sound["source_pts"], decoded_samples=sound["samples"],
                              periodic_blocks=sound["periodic_blocks"], measured_blocks=sound["blocks"])
            else:
                assert "mouth_hold_sound" not in packet and "mouth_hold_audio" not in packet["artifacts"]
                assert "The soundtrack was not measured alongside these pictures." in packet["limitations"]
            images = check_report(client, aid, evidence, path, packet, output, positive)
            result["verified_pdf_images"] = images
            result["json_pdf_and_signed_replay"] = True
            results[name] = result
            print(name, "measured" if positive else "unassessed", round(time.monotonic() - started, 2), flush=True)
    receipt = {"schema": "chipmunk.video-mouth-hold-audio-gap/1", "cases": results, "simulated_paid_runs": len(results),
               "unpaid_jobs": 0, "duplicate_paid_webhooks_checked": True, "sample_time_base": "1/16000",
               "missing_samples": 3200, "scope": "Same archived test pictures and authored periodic tone in every delivery."}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="New private run directory")
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
