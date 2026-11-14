"""Verify aligned speech evidence through an isolated prepaid audio analysis."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import wave
from pathlib import Path

from scripts.audio_aligned_speech_transfer import cases, sources
from scripts.audio_focused_window_app import checkout, write_wav
from scripts.audio_reference_variable_window import read_noise
from scripts.audio_vocal_event_mixtures import digest, read, save
from scripts.video_short_return_encoded_app import PaymentStub, body, configure

ROOT = Path(__file__).resolve().parents[2]
CASES = {("clean", "cut_paste"): "localized_difference", ("clean", "unchanged"): "no_local_difference",
         ("train", "cut_paste"): "unassessed"}


def run(output, bank, sounds):
    fixtures = {(scene, role): (truth, first, second)
                for item, scene, role, truth, first, second in cases(read(sources()[0]), bank, read_noise(sounds))
                if item["id"] == 13 and (scene, role) in CASES}
    if fixtures.keys() != CASES.keys():
        raise ValueError("Missing expected public test pairs")
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true", VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_LONG_CLEAN_SPEECH="true", VERITAS_AUDIO_REFERENCE_DENSE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_EQ_GUARD="true", VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH="true")

    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence.audio_reference import SCHEMA_WITH_ALIGNED_SPEECH
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as replay
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-aligned-speech-review", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()

        for (scene, role), status in CASES.items():
            truth, before, after = fixtures[scene, role]
            name = f"{scene}-{role}"
            earlier, submitted = (output / f"{name}-{side}.wav" for side in ("earlier", "submitted"))
            for path, pcm in ((earlier, before), (submitted, after)):
                write_wav(path, pcm.tobytes())
            aid, _ = checkout(client, stub, submitted)
            with get_session() as db:
                frozen = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/audio-reference-comparison"
            assert body(client.get(endpoint))["aligned_speech_available"] is True
            files = {"earlier": (earlier.name, earlier.read_bytes(), "audio/wav")}
            plain = body(client.post(endpoint, files=files))["comparison"]
            assert "aligned_speech" not in plain
            data = {"include_pdf": "true", "earlier_has_clean_speech": "true"}
            payload = body(client.post(endpoint, files=files, data=data))
            comparison = payload["comparison"]
            check = comparison["aligned_speech"]
            assert comparison["schema"] == SCHEMA_WITH_ALIGNED_SPEECH
            assert check["status"] == status and check["affects_overall_confidence"] is False
            scan(json.dumps(comparison))
            exported = output / f"{name}.json"
            save(exported, {key: payload[key] for key in ("comparison", "assets")})
            assert replay(earlier, submitted, exported, verify=True) == comparison
            pdf = base64.b64decode(payload["pdf"], validate=True)
            (output / f"{name}.pdf").write_bytes(pdf)

            excerpts = {}
            with pymupdf.open(stream=pdf, filetype="pdf") as report:
                pages = [page for page in report if "Aligned speech comparison" in page.get_text()]
                assert len(pages) == 1
                page = pages[0]
                text = " ".join(page.get_text().split())
                scan(text)
                assert "zero weight" in text and check["summary"] in text
                if status == "localized_difference":
                    assert sum(len(path["items"]) > 20 for path in page.get_drawings()) == 2
                    assert "Each surrounding section lasts 400 ms" in text
                    span = check["range"]
                    for side, pcm in (("earlier", before), ("submitted", after)):
                        item = span[side]
                        start, end = item["samples"]
                        encoded = base64.b64decode(payload["assets"][item["audio"]["file"]], validate=True)
                        with wave.open(io.BytesIO(encoded)) as clip:
                            snippet = clip.readframes(clip.getnframes())
                        assert snippet == pcm[start:end].tobytes() and digest(snippet) == item["pcm_sha256"]
                        assert f"{start / 16000:.3f} to {end / 16000:.3f} s" in text
                        assert f"Source samples [{start}, {end})" in text
                        excerpts[side] = {"samples": [start, end], "pcm_sha256": item["pcm_sha256"]}
                    assert excerpts["earlier"]["samples"] != excerpts["submitted"]["samples"]
                else:
                    assert "range" not in check and "Source samples" not in text
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == frozen
            results[name] = {"truth_samples": truth, "status": status, "excerpts": excerpts,
                             "source_sha256": [digest(pcm.tobytes()) for pcm in (before, after)]}
    save(output / "receipt.json", {"cases": results, "paid_runs": len(results), "unpaid_jobs": 0,
                                    "overall_assessments_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sounds", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.bank, args.sounds)
