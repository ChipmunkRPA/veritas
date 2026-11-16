# SPDX-License-Identifier: MIT
"""Replay short speech evidence on identical-channel lossless uploads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path
from subprocess import run

from scripts.audio_short_edit_input_views import SOURCE, duration

ROOT = Path(__file__).resolve().parents[2]


def prepared(inputs, old, output):
    import numpy as np

    case = old["plan"]["cases"]["fresh"][17]
    witness = old["readings"]["fresh"][17]["current"]
    raw = (inputs / "pcm" / (case["pcm_sha256"] + ".pcm")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["pcm_sha256"] and witness["range"] is not None
    signal = np.frombuffer(raw, "<i2")
    assert signal.min() > -32768 and signal.max() < 32767
    for name, data in (
        ("mono", signal[:, None]),
        ("identical-wav", np.column_stack((signal, signal))),
        ("identical-flac", np.column_stack((signal, signal))),
        ("different", np.column_stack((signal - 1, signal + 1))),
    ):
        file = output / f"{name}.wav"
        with wave.open(str(file), "wb") as stream:
            stream.setparams((data.shape[1], 2, 16000, len(signal), "NONE", "not compressed"))
            stream.writeframes(data.astype("<i2").tobytes())
        if name == "identical-flac":
            encoded = file.with_suffix(".flac")
            run(["ffmpeg", "-v", "error", "-i", str(file), str(encoded)], check=True)
            file = encoded
        yield name, file, case["pcm_sha256"], witness["range"]


def measure(args):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(args.output, ROOT)
    os.environ.update(
        VERITAS_AUDIO_DETECTOR_MODE="candidate",
        VERITAS_AUDIO_SHORT_ADMISSION_REVIEW="true",
        VERITAS_LOG_LEVEL="WARNING",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence.audio_short_review_binding import IDENTICAL_CHANNEL_DESCRIPTION, IDENTICAL_CHANNEL_PCM
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    old = duration.read_gz(args.prior)
    stub = PaymentStub()
    service._stripe = stub
    observed = []
    baseline = None
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-identical-stereo", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, file, channel_pin, range_ in prepared(args.inputs, old, args.output):
            ident = name.startswith("identical-")
            aid = checkout(
                client, stub, file, kind="audio", media_type=f"audio/{file.suffix[1:]}", filename=file.name
            )
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            scan(exported.decode())
            source, review = (evidence[key] for key in ("source", "short_voice_review"))
            assert source.get(IDENTICAL_CHANNEL_PCM) == (channel_pin if ident else None)
            assert review["affects_overall_confidence"] is False
            if name == "different":
                assert review["status"] == "skipped" and not review["checked_windows"]
            else:
                assert review["status"] == "flagged"
                assert [review["interval"][key] for key in ("start_sample", "end_sample")] == range_
            hearing = {
                key: review.get(key) for key in ("interval", "window_coverage", "nearby_window_review")
            }
            hearing["assets"] = {key: item["sha256"] for key, item in review["artifacts"].items()}
            if baseline is None:
                baseline = (source["pcm_sha256"], hearing)
            else:
                assert source["pcm_sha256"] == baseline[0]
            if ident:
                assert hearing == baseline[1]
            report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            written = " ".join(pdf_text(pdf).split())
            scan(written)
            assert (IDENTICAL_CHANNEL_DESCRIPTION in written) == ident
            if ident:
                assert f"Source PTS [{range_[0]}, {range_[1]}) at 1/16000." in written
            pack = body(
                client.post(
                    f"{route}/packages",
                    json={"report_id": report["id"], "kind": "draft", "include_original": False},
                )
            )
            pack = wait(client, f"/api/v2/packages/{pack['id']}", timeout=120)
            archive = file.with_suffix(".zip")
            archive.write_bytes(fetch(client, f"/api/v2/packages/{pack['id']}/file"))
            verified = verify_package(
                archive,
                public_key_pem=signer["public_key_pem"].encode(),
                expect_key_id=signer["key_id"],
                source_audio=file,
            )
            assert verified.outcome == "verified", verified.as_dict()
            assert (
                "Verified that both recorded channels match the analyzed audio." in str(verified.as_dict())
            ) == ident
            file.with_suffix(".pdf").write_bytes(pdf)
            observed.append(
                {
                    "case": name,
                    "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "pcm_sha256": source["pcm_sha256"],
                    "channel_proof": source.get(IDENTICAL_CHANNEL_PCM),
                    "status": review["status"],
                    "evidence": hearing,
                    "signed_source_verified": True,
                }
            )
            print(name, review["status"], review["interval"], "signed source verified", flush=True)
    assert len(stub.sessions) == len(observed) == 4
    SOURCE.write(
        args.output / "receipt.json",
        {
            "schema": "chipmunk.audio-short-identical-app/1",
            "cases": observed,
            "simulated_payment_usd_cents": 999,
            "jobs_before_payment": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "inputs", "output"):
        parser.add_argument(key, type=Path)
    measure(parser.parse_args())
