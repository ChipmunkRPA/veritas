"""Verify source-clocked 48 kHz speech evidence through local prepaid analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from app.evidence import audio_two_second_binding as binding
from scripts import audio_two_second_carrier_transfer as study
from scripts.audio_two_second_app import span, verify_exports, without_asset_urls
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

ROOT = Path(__file__).resolve().parents[2]
MEDIA_TYPE = {"wav": "audio/wav", "flac": "audio/flac", "m4a": "audio/mp4"}


def representatives(path):
    result = study.previous.prior.read(path)
    assert study.recount(result["rows"]) == result["counts"]
    match = study.previous.prior.matches
    edited, ordinary = [], []
    for row in result["rows"]:
        base, union, additions = study.ranges(row)
        if row["kind"] == "edited" and match(union, row["truth"]) - match(base, row["truth"]):
            edited.append(row)
        if row["kind"] == "ordinary" and not additions and any(excerpt["responses"] for excerpt in row["excerpts"]):
            ordinary.append(row)
    if not edited:
        raise ValueError("No added useful range for the app replay in this carrier receipt")
    candidate = edited[0]
    yield result["carrier"], candidate, False, "option-off"
    yield result["carrier"], candidate, True, "edited"
    if result["carrier"] == "wav":
        yield "flac", candidate, True, "flac"
    yield result["carrier"], ordinary[0], True, "ordinary"


def run(receipts, roots, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_FOCUSED_LONG_REVIEW="true", VERITAS_AUDIO_FOCUSED_LONG_48K_REVIEW="true",
                      VERITAS_AUDIO_TWO_SECOND_LONG_REVIEW="true", VERITAS_AUDIO_TWO_SECOND_LONG_CONFIRMATION="true",
                      VERITAS_LOG_LEVEL="WARNING")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.audio.decode import decode_audio
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    cases, browser_cases, earlier = {}, [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-48k-speech-review", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for receipt in receipts:
            for carrier, row, enabled, purpose in representatives(receipt):
                name = f"{carrier}-{purpose}"
                media = output / f"{name}.{carrier}"
                pcm = roots[row["role"]] / "pcm" / f"{row['pcm_sha256']}.pcm"
                assert study.digest(pcm.read_bytes()) == row["pcm_sha256"]
                if carrier == "flac":
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(output / "wav-edited.wav"),
                                    "-c:a", "flac", str(media)], check=True, timeout=30)
                    pin = study.digest(media.read_bytes())
                else:
                    pin = study.encode(pcm, media)
                    assert pin == row["file_sha256"]
                decoded = decode_audio(media, source_sha256=pin)
                assert binding.presented_samples(decoded.mapping) == row["samples"]
                assert decoded.mapping["pcm_sha256"] == row["decoded_pcm_sha256"]
                settings.audio_two_second_long_48k_review = enabled and carrier in {"wav", "flac"}
                settings.audio_two_second_long_aac48_review = enabled and carrier == "m4a"
                aid = checkout(client, stub, media, kind="audio", media_type=MEDIA_TYPE[carrier], filename=media.name)
                route = f"/api/v2/analyses/{aid}"
                evidence = body(client.get(f"{route}/evidence-packets"))
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                for value in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                    scan(value.decode())
                    assert binding.DETECTOR_ID.encode() not in value and b'"logit"' not in value
                with get_session() as db:
                    result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
                assessment = (result["decision"], result.get("customer_assessment"),
                              without_asset_urls(evidence.get("long_voice_review")))
                if not enabled:
                    assert binding.KEY not in evidence
                    earlier[carrier] = assessment
                    continue
                if purpose == "edited":
                    assert assessment == earlier[carrier]
                review = evidence[binding.KEY]
                binding.validate(review, evidence["source"])
                expected = binding.regions(binding.summarize_readings(row["excerpts"], row["samples"],
                                           schema=binding.PRESENTED_SCHEMA, codec=row["codec"]), schema=binding.PRESENTED_SCHEMA)
                assert [span(area) for area in review["regions"]] == [span(area) for area in expected]
                assert [span(area["confirming_window"]) for area in review["regions"]] == [
                    span(area["confirming_window"]) for area in expected]
                assert review["status"] == ("no_flag" if purpose == "ordinary" else "flagged"), review["reason"]
                decoded_pcm = (decoded.pcm * 32768).astype("<i2").tobytes()
                sha = verify_exports(client, route, media, evidence, decoded_pcm, signer)
                browser_cases.append({"analysis_id": aid, "name": name, "review": review})
                cases[name] = {"status": review["status"], "pdf_sha256": sha, "file_sha256": pin,
                               "known_edit_ranges": row["truth"], "regions": [span(area) for area in expected],
                               "supporting_regions": [span(area["confirming_window"]) for area in expected],
                               "source_ticks": decoded.mapping["source_duration_pts"], "decoded_samples": len(decoded.pcm)}
                print(name, cases[name], flush=True)
        browser = browser_check(output, app, token, browser_cases, scenario="long-two-second-speech-replay.mjs")
    study.previous.prior.save(output / "receipt.json", {"cases": cases, "browser": browser,
        "paid_analyses": len(stub.sessions), "unpaid_jobs": 0, "same_assessment_and_baseline": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    run(args.receipts, dict(zip(study.ROLE, (args.first, args.second), strict=True)), args.output)
