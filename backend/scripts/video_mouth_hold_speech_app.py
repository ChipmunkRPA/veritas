"""Verify speech activity in the app and signed report for a source-bound mouth hold."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_vocal_event_mixtures import save
from scripts.video_mouth_hold_clock_app import check_report, sha
from scripts.video_mouth_hold_speech_transfer import interval, sources
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]
PICTURE_SHA = "d611ea57436932647fdb3075943f60fc90c609409b4271bb937a7f0071dc2866"
CASES = {(0, "variant", "train"): True, (0, "background", "train"): False,
         (2, "background", "competing_speech"): True}
MOUTH_SAMPLES = (8320, 13440)


def run(output, picture, bank, sounds):
    if sha(picture.read_bytes()) != PICTURE_SHA:
        raise ValueError("The picture fixture differs from the known hold")
    cases = {}
    for pair, role, scene, pcm in sources(bank, sounds):
        key = pair["id"], role, scene
        if key in CASES:
            offset = interval(pair)[0] - MOUTH_SAMPLES[0]
            cases[key] = pcm[offset:offset + 32000]
    if cases.keys() != CASES.keys() or any(len(pcm) != 32000 for pcm in cases.values()):
        raise ValueError("The audio fixtures do not cover the complete two-second source")

    configure(output, ROOT)
    os.environ["VERITAS_VIDEO_MOUTH_HOLD_SPEECH_CONTEXT"] = "true"

    import cv2
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import wav_bytes
    from app.engine.registry import get_registry
    from app.evidence.mouth_hold_sound import ACTIVITY_SCHEMA
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
            principal = auth.new_principal("local-mouth-hold-speech", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for key, expected in CASES.items():
            _, role, scene = key
            name = f"{role}-{scene}"
            wave = output / f"{name}.wav"
            wave.write_bytes(wav_bytes(cases[key].astype("float32") / 32768))
            video = output / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(picture), "-i", str(wave),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "alac", str(video)],
                           check=True, timeout=20)
            aid = checkout(client, stub, video)
            order = next(reversed(stub.sessions.values()))
            assert (order["amount_total"], order["currency"]) == (999, "usd")
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert body(client.get(f"/api/analyses/{aid}/export.json"))["evidence"] == evidence
            packet, = (row for row in evidence["packets"] if row.get("mouth_local_hold"))
            sound = packet["mouth_hold_sound"]
            assert sound["schema"] == ACTIVITY_SCHEMA and sound["samples"] == list(MOUTH_SAMPLES)
            assert sound["source_pts"] == list(MOUTH_SAMPLES) and sound["speech_activity"]["context"] == [0, 32000]
            assert (sound["speech_activity"]["samples"] >= 3200) == expected
            assert packet["affects_overall_confidence"] is False
            assert [row["anchor"]["frame_index"] for row in packet["frames"]] == [12, 17, 21]
            scan(json.dumps(packet))
            exhibits = check_report(client, aid, evidence, video, packet, output, True)
            with ZipFile(output / f"{name}.zip") as archive:
                pdf = archive.read("report.pdf")
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                text = " ".join(" ".join(page.get_text().split()) for page in document)
                assert "Estimated speech activity during the hold" in text
                assert ("The sound cannot be assigned to the visible person" in text) == expected
            results[name] = {"source_sha256": sha(video.read_bytes()), "sound": sound,
                             "frame_indices": [12, 17, 21], "source_assets_verified": exhibits}
            print(name, sound["speech_activity"], flush=True)
    save(output / "receipt.json", {"cases": results, "paid_runs": len(results), "unpaid_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "picture", "bank", "sounds"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.output, args.picture, args.bank, args.sounds)
