# SPDX-License-Identifier: Apache-2.0
"""Replay the AAC or Opus waveform evidence in a local app with simulated payment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]


def run(output: Path, inputs: Path, *, browser: bool = False, gapless: bool = False, opus: bool = False):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true", VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_AAC="true", VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_MATCHING_WAVEFORM="true", VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"))
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import read_wav
    from app.evidence.audio_reference import (
        SCHEMA_WITH_AAC_MATCHING_WAVEFORM,
        SCHEMA_WITH_GAPLESS_AAC_MATCHING_WAVEFORM,
        SCHEMA_WITH_MATCHING_WAVEFORM,
        SCHEMA_WITH_OPUS_MATCHING_WAVEFORM,
        _identity,
        read_track,
    )
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as replay
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = inputs / "earlier.flac"
    schema = SCHEMA_WITH_GAPLESS_AAC_MATCHING_WAVEFORM if gapless else SCHEMA_WITH_AAC_MATCHING_WAVEFORM
    codec, proof, extension, media_type = "AAC", "aac_presentation", "m4a", "audio/mp4"
    if opus:
        schema = SCHEMA_WITH_OPUS_MATCHING_WAVEFORM
        codec, proof, extension, media_type = "Opus", "opus_presentation", "opus", "audio/opus"
    stub = service._stripe = PaymentStub()
    cases, results = [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-aac-waveform-review", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        scenarios = [("positive", "localized_difference"), ("ordinary", "no_local_difference")]
        if opus:
            scenarios.append(("rejected", "unassessed"))
        for role, status in scenarios:
            path = inputs / f"{role}.{extension}"
            aid = checkout(client, stub, path, kind="audio", media_type=media_type, filename=path.name)
            with get_session() as db:
                result_hash = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/audio-reference-comparison"
            assert body(client.get(endpoint))["matching_waveform_aac_available"]
            files = {"earlier": (source.name, source.read_bytes(), "audio/flac")}
            assert "matching_waveform" not in body(client.post(endpoint, files=files))["comparison"]
            payload = body(client.post(endpoint, files=files, data={"include_pdf": "true", "earlier_has_clean_speech": "true"}))
            record = payload["comparison"]
            waveform = record["matching_waveform"]
            expected_schema = SCHEMA_WITH_MATCHING_WAVEFORM if role == "rejected" else schema
            assert record["schema"] == expected_schema and waveform["status"] == status
            assert (proof in waveform) == (role != "rejected")
            assert not waveform["affects_overall_confidence"] and not record["affects_overall_confidence"]
            scan(json.dumps(record))
            evidence = output / f"{role}.json"
            evidence.write_text(json.dumps({part: payload[part] for part in ("comparison", "assets")}))
            assert replay(source, path, evidence, verify=True) == record
            pdf = base64.b64decode(payload["pdf"], validate=True)
            (output / f"{role}.pdf").write_bytes(pdf)
            with pymupdf.open(stream=pdf, filetype="pdf") as report:
                [page] = [page for page in report if "Matching speech waveform" in page.get_text()]
                text = " ".join(page.get_text().split())
                scan(text)
                note = "Playback begins after 2,112 priming samples" if gapless else "48 kHz AAC presentation clock"
                if opus:
                    note = "Verified 48 kHz Opus playback."
                assert waveform["summary"] in text and (note in text) == (role != "rejected") and "zero weight" in text
                if span := waveform.get("range"):
                    assert sum(len(drawing["items"]) > 20 for drawing in page.get_drawings()) == 2
                    for side, original in (("earlier", source), ("submitted", path)):
                        clip = span[side]
                        start, stop = clip["samples"]
                        audio = read_wav(base64.b64decode(payload["assets"][clip["audio"]["file"]], validate=True))
                        _, pcm, _ = read_track(original, _identity(original, original.name), max_seconds=7)
                        assert (audio == pcm[start:stop]).all()
                        assert f"{clip['start_s']:.3f} to {clip['end_s']:.3f} s" in text
                    first, last = span["submitted"]["native_source_pts"]
                    assert f"{codec} presentation [{first}, {last}) on the 48 kHz clock." in text
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == result_hash
            cases.append({"analysis_id": aid, "earlier": str(source.resolve()), "expected": str(evidence), "status": status})
            results[role] = {"status": status, "sha256": record["submitted"]["sha256"],
                             "pdf_sha256": hashlib.sha256(pdf).hexdigest(), proof: waveform.get(proof),
                             "ranges": {side: {key: span[side][key] for key in ("samples", "start_s", "end_s", "pcm_sha256")}
                                        for side in ("earlier", "submitted")} if span else {}}
            if span:
                results[role]["native_source_pts"] = span["submitted"]["native_source_pts"]
        receipt = {"results": results, "simulated_paid_jobs": len(stub.sessions), "unpaid_jobs": 0,
                   "original_assessments_preserved": True}
        if browser:
            receipt["browser"] = browser_check(output, app, key, cases, scenario="audio-waveform-aac-replay.mjs")
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--browser", action="store_true")
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--gapless", action="store_true")
    formats.add_argument("--opus", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, arguments.inputs, browser=arguments.browser, gapless=arguments.gapless, opus=arguments.opus)
