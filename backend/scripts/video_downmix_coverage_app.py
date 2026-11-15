# SPDX-License-Identifier: MIT
"""Reproduce cancelled video soundtrack coverage through paid local analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.video-downmix-coverage/1"
RATE = 16000
CASES = {
    "alac-control": ("alac", "1-equal_channels", []),
    "alac-partial": ("alac", "1-cancellation", [1]),
    "aac-control": ("aac", "2-equal_channels", []),
    "aac-partial": ("aac", "2-cancellation", [1]),
    "alac-cancelled": ("alac", "3-cancellation", [1]),
    "alac-silent": ("alac", None, []),
}


def prepare(inputs, destination):
    import numpy as np

    receipt = json.loads((inputs / "receipt.json").read_bytes())
    known = {(row["file_sha256"], row["case"]) for row in receipt["rows"]}
    destination.mkdir(mode=0o700)
    for name, (codec, stem, _) in CASES.items():
        wav = inputs / f"{stem}.wav" if stem else None
        if wav is not None:
            assert (digest(wav.read_bytes()), stem.split("-", 1)[1]) in known
        if name in {"alac-cancelled", "alac-silent"}:
            values = np.zeros((3 * RATE, 2), dtype="<i2")
            if wav is not None:
                with wave.open(str(wav), "rb") as carrier:
                    assert (carrier.getnchannels(), carrier.getsampwidth(), carrier.getframerate()) == (2, 2, RATE)
                    stereo = np.frombuffer(carrier.readframes(3 * RATE), dtype="<i2").reshape(-1, 2).astype(np.int32)
                side = (stereo[:, 0] - stereo[:, 1]) // 2
                values[:, 0], values[:, 1] = side, -side
            wav = destination / f"{name}.wav"
            with wave.open(str(wav), "wb") as carrier:
                carrier.setparams((2, 2, RATE, 0, "NONE", ""))
                carrier.writeframes(values.tobytes())
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=3",
                        "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-threads:v", "1",
                        "-pix_fmt", "yuv420p", "-c:a", codec, str(destination / f"{name}.mp4")], check=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(fixtures, output, *, before):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(output, ROOT)
    os.environ.update(VERITAS_AUDIO_DETECTOR_MODE="candidate", VERITAS_DEVICE="auto", TOKENIZERS_PARALLELISM="false",
                      VERITAS_LOG_LEVEL="ERROR")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence import video_audio
    from app.evidence.audio_binding import MONO_VOICE_COVERAGE, MONO_VOICE_LIMIT
    from app.evidence.stereo_binding import destructive_downmix_windows
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if before:
        video_audio.destructive_downmix_windows = lambda _context: []
    stub = PaymentStub()
    service._stripe = stub
    rows = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-video-downmix-check", "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        for name, (_, _, expected) in CASES.items():
            source = fixtures / f"{name}.mp4"
            aid = checkout(client, stub, source, filename=source.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            audio = evidence["audio"]
            stereo = audio["stereo_context"]
            assert destructive_downmix_windows(stereo) == expected
            affected = [stereo["windows"][index] for index in expected]
            public = body(client.get(route))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(exported.decode())
            customer = public["customer_assessment"]
            json_data = json.loads(exported)
            assert json_data["evidence"] == evidence and json_data["customer_assessment"] == customer
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            soundtrack = customer.get("soundtrack")
            limit = audio["coverage"].get(MONO_VOICE_COVERAGE)
            guarded = bool(expected) and not before
            assert bool(limit) == guarded
            if guarded:
                assert limit["stereo_window_indices"] == expected and limit["reason"] == MONO_VOICE_LIMIT
                assert soundtrack["direction"] == "inconclusive" and soundtrack["confidence_pct"] is None
                assert "voice" in customer["excluded_channels"]
            verified = False
            if guarded:
                for index in expected:
                    listening, = [row for row in audio["stereo_listening"]["windows"] if row["stereo_window_index"] == index]
                    for asset in listening["artifacts"].values():
                        assert digest(fetch(client, asset["url"])) == asset["sha256"]
                report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                words = " ".join(pdf_text(pdf).split())
                scan(words)
                assert MONO_VOICE_LIMIT in words and "/audio/coverage/mono_voice_coverage_limit" in words
                assert "1.000–2.000 s" in words
                (output / f"{name}.pdf").write_bytes(pdf)
                package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft"}))
                package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
                archive = output / f"{name}.zip"
                archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                signer = body(client.get("/api/v2/instance/public-key"))
                proof = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                                       source_audio=source, source_video=source)
                assert proof.ok, proof.as_dict()
                verified = True
            rows[name] = {"source_sha256": digest(source.read_bytes()), "mono_sha256": audio["source"]["pcm_sha256"],
                          "voice_window_count": len(audio["coverage"]["windows"]), "affected": affected,
                          "decode_status": result["manifest"]["sampling"]["video_audio_evidence"]["decode_status"],
                          "overall_code": result["decision"]["code"], "soundtrack": soundtrack, "guarded": guarded,
                          "signed_report_verified": verified}
            print(name, rows[name]["decode_status"], soundtrack, flush=True)
    assert len(rows) == len(stub.sessions) == len(CASES)
    (output / "receipt.json").write_text(json.dumps({"schema": SCHEMA, "price_usd_cents": 999, "jobs_before_payment": 0,
                                                    "rows": rows}, indent=2, sort_keys=True) + "\n")


def review(summary):
    prior, current = summary["before"], summary["after"]
    for receipt in (prior, current):
        assert receipt["schema"] == SCHEMA and receipt["price_usd_cents"] == 999 and receipt["jobs_before_payment"] == 0
        assert set(receipt["rows"]) == CASES.keys()
    original_audio, delivered = prior["rows"], current["rows"]
    assert delivered["alac-partial"]["mono_sha256"] == delivered["alac-control"]["mono_sha256"]
    assert delivered["alac-cancelled"]["mono_sha256"] == delivered["alac-silent"]["mono_sha256"]
    for name, (_, _, expected) in CASES.items():
        before, after = original_audio[name], delivered[name]
        for key in ("source_sha256", "mono_sha256", "voice_window_count", "affected", "overall_code"):
            assert before[key] == after[key], (name, key)
        assert not before["guarded"] and not before["signed_report_verified"]
        if expected:
            row, = after["affected"]
            assert row["start_sample"] == RATE and row["end_sample"] == 2 * RATE and row["correlation"] <= -.85
            assert row["source_timestamp_ranges"] == [{"start_sample": RATE, "end_sample": 2 * RATE,
                "source_start_pts": RATE, "source_end_pts": 2 * RATE, "time_base": "1/16000"}]
            assert row["downmix"]["strong_attenuation"] and after["guarded"] and after["signed_report_verified"]
            assert after["soundtrack"]["direction"] == "inconclusive" and after["soundtrack"]["confidence_pct"] is None
            if name.endswith("partial"):
                assert before["soundtrack"]["direction"] == "original" and before["voice_window_count"] > 0
        else:
            assert before == after and not after["affected"]
    assert original_audio["alac-cancelled"]["decode_status"] == "silent" and delivered["alac-cancelled"]["decode_status"] == "cancelled"
    print("Verified three corrected soundtrack assessments, three unchanged controls and three signed original-video replays.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("prepare", "before", "after", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("output", nargs="?", type=Path)
    args = cli.parse_args()
    if args.mode == "review":
        review(json.loads(args.path.read_bytes()))
    elif args.output is None:
        cli.error("An output directory is required")
    elif args.mode == "prepare":
        prepare(args.path, args.output)
    else:
        run(args.path, args.output, before=args.mode == "before")
