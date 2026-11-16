# SPDX-License-Identifier: MIT
"""Verify 48 kHz short speech analysis and signed exports through the paid flow."""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

from app.evidence import audio_short_review_binding as binding
from scripts.audio_short_lossless_48k_transfer import (
    PARENT,
    deliver,
    ffmpeg,
    read_record,
    require,
    sha,
    write_wav,
)

ROOT = Path(__file__).resolve().parents[2]


def fixtures(folder, fresh):
    parent = read_record(PARENT)
    for index, purpose in ((17, "edit"), (16, "ordinary")):
        case = parent["plan"]["cases"]["fresh"][index]
        pin = case["pcm_sha256"]
        raw = (fresh / "pcm" / f"{pin}.pcm").read_bytes()
        require(sha(raw) == pin, "Fixture differs from the held recording")
        range_ = parent["readings"]["fresh"][index]["current"]["range"]
        interval = dict(zip(("start_sample", "end_sample"), range_, strict=True)) if range_ else None
        target = folder / purpose
        target.mkdir()
        _, files = deliver(raw, target)
        for carrier, path in files.items():
            if purpose == "edit" or carrier == "flac":
                yield f"{purpose}-{carrier}", path, interval, True, 288000
        if purpose == "edit":
            yield "option-off", files["wav"], None, False, 288000
            with wave.open(str(files["wav"]), "rb") as stream:
                native = stream.readframes(stream.getnframes())
            outside = target / "one-extra-sample.wav"
            write_wav(outside, native + b"\x00\x00")
            yield "one-extra-sample", outside, None, True, 288001
            inconsistent = target / "inconsistent-clock.flac"
            ffmpeg("-i", str(outside), "-c:a", "flac", str(inconsistent))
            raw_flac = inconsistent.read_bytes()
            fields = int.from_bytes(raw_flac[18:26], "big")
            mask = (1 << 36) - 1
            require(fields & mask == 288001, "Expected the off-grid control extent")
            inconsistent.write_bytes(raw_flac[:18] + ((fields & ~mask) | 288000).to_bytes(8, "big") + raw_flac[26:])
            yield "inconsistent-flac", inconsistent, None, True, 288000


def verify_flow(args):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate", VERITAS_AUDIO_SHORT_ADMISSION_REVIEW="true",
                      VERITAS_LOG_LEVEL="WARNING", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results = []
    assessments = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-lossless-audio-48k", "examiner")
            db.add(principal)
            db.flush()
            token, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, file, interval, enabled, native_samples in fixtures(args.output, args.fresh):
            settings.audio_short_lossless_48k_review = enabled
            aid = checkout(client, stub, file, kind="audio", media_type=f"audio/{file.suffix[1:]}", filename=file.name)
            with get_session() as db:
                snapshot = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
                assessments[name] = (snapshot["decision"], snapshot.get("customer_assessment"))
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            require(json.loads(exported)["evidence"] == evidence, "JSON differs from the evidence API")
            scan(exported.decode())
            source, review = evidence["source"], evidence["short_voice_review"]
            require(source["source_sample_rate"] == 48000 and source["source_time_base"] == "1/48000"
                    and source["source_duration_pts"] == native_samples and source["samples"] == binding.SAMPLES,
                    "The app must preserve the exact source extent separately from its six-second decoded input")
            native_verified = name not in {"one-extra-sample", "inconsistent-flac"}
            require(bool(source.get(binding.LOSSLESS_NATIVE_CLOCK)) == native_verified, "Native decode coverage differs")
            checked = native_verified and enabled
            status = "flagged" if interval else "no_flag" if checked else "skipped"
            require(review["status"] == status and review["interval"] == interval and review["affects_overall_confidence"] is False,
                    "The app review differs from the held scan or selected trial setting")
            if checked:
                require(len(review["window_coverage"]) == 5, "The app does not cover the five short windows")
            if name.startswith("edit-"):
                require(interval is not None and "nearby_window_review" in review, "The fixture needs a source-linked positive")
            report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
            report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
            pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
            written = " ".join(pdf_text(pdf).split())
            scan(written)
            require((binding.RESAMPLED_DESCRIPTION in written) == checked, "The PDF omits or misstates the input clock")
            if interval:
                require(f"Source PTS [{interval['start_sample']}, {interval['end_sample']}) at 1/16000." in written,
                        "The PDF interval differs from the API")
            pack = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft", "include_original": False}))
            pack = wait(client, f"/api/v2/packages/{pack['id']}", timeout=120)
            archive = args.output / f"{name}.zip"
            archive.write_bytes(fetch(client, f"/api/v2/packages/{pack['id']}/file"))
            signed = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(), expect_key_id=signer["key_id"],
                                    source_audio=file)
            require(signed.outcome == "verified", str(signed.as_dict()))
            (args.output / f"{name}.pdf").write_bytes(pdf)
            results.append({"case": name, "enabled": enabled, "source_sha256": sha(file.read_bytes()),
                            "source_duration_pts": source["source_duration_pts"], "pcm_sha256": source["pcm_sha256"],
                            "native_clock_verified": native_verified,
                            "status": status, "interval": interval, "source_timestamp_ranges": review["source_timestamp_ranges"],
                            "assets": {key: item["sha256"] for key, item in review["artifacts"].items()},
                            "signed_source_verified": True})
            print(name, status, interval, "signed source verified", flush=True)
    require(len(stub.sessions) == len(results) == 6, "Expected six simulated paid analyses")
    require(assessments["edit-wav"] == assessments["option-off"], "Optional short speech changed the overall assessment")
    (args.output / "receipt.json").write_text(json.dumps({"schema": "chipmunk.audio-short-lossless-48k-app/1", "cases": results,
                                                         "simulated_payment_usd_cents": 999, "jobs_before_payment": 0,
                                                         "overall_assessment_preserved": True},
                                                        indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("fresh", "output"):
        parser.add_argument(item, type=Path)
    verify_flow(parser.parse_args())
