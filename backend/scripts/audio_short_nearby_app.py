# SPDX-License-Identifier: MIT
"""Verify nearby speech-window evidence through prepaid analysis and signed exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from subprocess import run as run_command

from scripts.audio_short_edit_input_views import SOURCE, duration

ROOT = Path(__file__).resolve().parents[2]
CASES = (("diagnostic", 42, "varied", "wav"), ("fresh", 17, "repeated", "flac"), ("fresh", 16, None, "wav"))


def run(args):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(args.output, ROOT)
    os.environ.update(VERITAS_AUDIO_DETECTOR_MODE="candidate", VERITAS_AUDIO_SHORT_ADMISSION_REVIEW="true",
                      VERITAS_LOG_LEVEL="WARNING", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import numpy as np
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import wav_bytes
    from app.evidence.audio_short_review_binding import NEARBY_SCHEMA, NEARBY_SUMMARIES
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.verify_short_speech_coverage import verify_coverage

    study = duration.read_gz(args.study)
    stub = PaymentStub()
    service._stripe = stub
    observed = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-nearby-windows", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        public_key = args.output / "public-key.pem"
        public_key.write_text(signer["public_key_pem"])
        for panel, index, expected, carrier in CASES:
            case = study["plan"]["cases"][panel][index]
            witness = study["readings"][panel][index]["current"]
            folder = args.fresh if panel == "fresh" else args.diagnostic
            raw = (folder / "pcm" / (case["pcm_sha256"] + ".pcm")).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == case["pcm_sha256"]
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.
            wav = args.output / f"{panel}-{index}.wav"
            wav.write_bytes(wav_bytes(pcm))
            file = wav.with_suffix(f".{carrier}")
            if carrier == "flac":
                run_command(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "flac", "-sample_fmt", "s16", str(file)], check=True)
            aid = checkout(client, stub, file, kind="audio", media_type=f"audio/{carrier}", filename=file.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            export = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(export)["evidence"] == evidence
            scan(export.decode())
            result = evidence["short_voice_review"]
            nearby = result.get("nearby_window_review")
            if expected is not None:
                assert result["schema"] == NEARBY_SCHEMA and result["affects_overall_confidence"] is False
                assert result["status"] == "flagged" and nearby["status"] == expected
                assert [result["interval"][key] for key in ("start_sample", "end_sample")] == witness["range"]
                for row in nearby["windows"]:
                    a, b = row["start_sample"], row["end_sample"]
                    assert row["pcm_sha256"] == hashlib.sha256(pcm[a:b].astype("<f4").tobytes()).hexdigest()
            else:
                assert result["status"] == "no_flag" and nearby is None and witness["range"] is None
            report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            text = " ".join(pdf_text(pdf).split())
            scan(text)
            if nearby:
                assert NEARBY_SUMMARIES[expected] in text
                for row in nearby["windows"]:
                    a, b = row["start_sample"], row["end_sample"]
                    assert f"{a/16000:.3f}–{b/16000:.3f} s." in text and f"Source PTS [{a}, {b}) at 1/16000." in text
            package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft",
                                                                 "include_original": False}))
            package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
            archive = file.with_suffix(".zip")
            archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
            verified = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(),
                                      expect_key_id=signer["key_id"], source_audio=file)
            assert verified.outcome == "verified", verified.as_dict()
            if nearby:
                assert "Verified the samples in each nearby speech window." in str(verified.as_dict())
                coverage = verify_coverage(archive, file, public_key, ROOT / "models/weights/audio-upgrade/silero_vad.jit")
                assert coverage["nearby_windows"] == nearby["windows"]
            file.with_suffix(".pdf").write_bytes(pdf)
            observed.append({"panel": panel, "row": index, "pcm_sha256": case["pcm_sha256"], "format": carrier,
                             "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                             "short_status": result["status"], "interval": result["interval"],
                             "nearby": nearby, "verified": True})
            print(panel, index, result["status"], expected, "signed source verified", flush=True)
    assert len(stub.sessions) == len(CASES)
    SOURCE.write(args.output / "receipt.json", {"schema": "chipmunk.audio-short-nearby-app/1",
                                               "simulated_payment_usd_cents": 999, "jobs_before_payment": 0,
                                               "cases": observed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "diagnostic", "fresh", "output"):
        parser.add_argument(key, type=Path)
    run(parser.parse_args())
