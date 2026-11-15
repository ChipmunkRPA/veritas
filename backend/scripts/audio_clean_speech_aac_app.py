# SPDX-License-Identifier: Apache-2.0
"""Check the AAC earlier-copy path through paid analysis and both downloads."""

import argparse
import base64
import json
import os
from pathlib import Path

from scripts.audio_clean_speech_aac_transfer import (
    CARRIERS48,
    ami_cases,
    encode,
    encode48,
    icsi_cases,
    read,
    review,
    sha,
)
from scripts.audio_focused_window_app import write_wav
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, counts

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_AUDIO_REFERENCE_LOCAL_TIMING="true", VERITAS_AUDIO_REFERENCE_AAC_TIMING="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true", VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_MATCHING_WAVEFORM="true")
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.audio_reference import SCHEMA_WITH_AAC_CLEAN_SPEECH, SCHEMA_WITH_MATCHING_WAVEFORM
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as compare_replay
    from scripts.verify_audio_clean_speech_export import verify_export
    from scripts.verify_multimedia_exports import scan

    receipt = read(args.receipt)
    review(receipt)
    carrier = args.carrier
    if receipt["bank"] == "ami":
        group_id, correction, source = 0, "piper", ami_cases
    else:
        source = icsi_cases
        selected = next(row for row in receipt["rows"] if row["kind"].startswith("edited/")
                        and row["outcomes"][carrier]["range"])
        group_id, correction = selected["group"], selected["kind"]
    pattern = {"piper" if receipt["bank"] == "ami" else "edited": correction, "untouched": "untouched", "unrelated": "unrelated"}
    expected = {row["kind"]: row for row in receipt["rows"] if row["group"] == group_id}
    fixtures = {kind: (dry, noisy) for group, kind, _, dry, noisy in source(args.sources) if group == group_id}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results, analyses = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-aac-speech-compare", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role, case in pattern.items():
            dry, noisy = fixtures[case]
            pinned = expected[case]
            assert sha(dry) == pinned["earlier_pcm_sha256"] and sha(noisy) == pinned["submitted_pcm_sha256"]
            earlier = args.output / f"{role}-earlier.wav"
            write_wav(earlier, dry)
            upload = "untouched" if role == "unrelated" else role
            submitted = args.output / f"{upload}-submitted.m4a"
            if upload not in analyses:
                if carrier in CARRIERS48:
                    encode48(noisy, submitted, carrier)
                else:
                    encode(noisy, submitted, 32)
                analyses[upload] = checkout(client, stub, submitted, kind="audio", media_type="audio/mp4", filename=submitted.name)
            assert sha(submitted.read_bytes()) == pinned["encoded_sha256"][carrier]
            aid = analyses[upload]
            with get_session() as db:
                frozen = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/audio-reference-comparison"
            files = {"earlier": (earlier.name, earlier.read_bytes(), "audio/wav")}
            form = {"include_pdf": "true", "earlier_has_clean_speech": "true"}
            before = counts()
            settings.audio_reference_clean_speech_aac = False
            assert "clean_speech_aac_available" not in body(client.get(endpoint))
            baseline = body(client.post(endpoint, data=form, files=files))["comparison"]["clean_speech"]
            assert baseline["status"] == "unassessed" and "range" not in baseline
            settings.audio_reference_clean_speech_aac = True
            assert body(client.get(endpoint))["clean_speech_aac_available"] is True
            result = body(client.post(endpoint, data=form, files=files))
            comparison = result["comparison"]
            check = comparison["clean_speech"]
            assert comparison["schema"] == SCHEMA_WITH_MATCHING_WAVEFORM
            assert comparison["comparison_schema"] == SCHEMA_WITH_AAC_CLEAN_SPEECH
            measured = pinned["outcomes"][carrier]
            span = measured["range"]
            assert check["status"] == measured["status"] and check["aac_presentation"] == measured["presentation"]
            assert check.get("range", {}).get("samples") == span and check["affects_overall_confidence"] is False
            exported = args.output / f"{role}.json"
            exported.write_text(json.dumps({name: result[name] for name in ("comparison", "assets")}))
            scan(exported.read_text())
            assert compare_replay(earlier, submitted, exported, verify=True) == comparison
            proof = verify_export(earlier, submitted, exported)
            assert proof["excerpts_verified"] == (2 if span else 0)
            pdf = base64.b64decode(result["pdf"], validate=True)
            (args.output / f"{role}.pdf").write_bytes(pdf)
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                page = next(page for page in document if "Local speech comparison" in page.get_text())
                words = " ".join(page.get_text().split())
                scan(words)
                assert "AAC duration 6.000 s" in words and check["summary"] in words
                if carrier in CARRIERS48:
                    channels = CARRIERS48[carrier][1]
                    assert f"48 kHz {'stereo' if channels == 2 else 'mono'} source" in words
                if span:
                    assert f"Source ticks [{span[0]}, {span[1]})" in words and len(page.get_drawings()) >= 4
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == frozen
            assert counts() == before
            results[role] = {"earlier_pcm_sha256": sha(dry), "submitted_m4a_sha256": sha(submitted.read_bytes()),
                             "baseline_status": baseline["status"], "presentation": check["aac_presentation"],
                             "pdf_sha256": sha(pdf), "source_replay": proof}
    (args.output / "receipt.json").write_text(json.dumps({"cases": results, "paid_analyses": len(analyses),
        "usd_cents": 999, "jobs_before_payment": 0, "saved_assessments_preserved": True}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--carrier", choices=("aac32", *CARRIERS48), default="aac32")
    run(parser.parse_args())
