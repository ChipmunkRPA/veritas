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
WAVEFORM_CASES = {("train", "cut_paste"): "localized_difference", ("train", "unchanged"): "no_local_difference",
                  ("clean", "resampled"): "unassessed"}


def run(output, bank, sounds, *, waveform=False):
    expected = WAVEFORM_CASES if waveform else CASES
    case_id = 23 if waveform else 13
    fixtures = {(scene, role): (truth, first, second)
                for item, scene, role, truth, first, second in cases(read(sources()[0]), bank, read_noise(sounds))
                if item["id"] == case_id and (scene, role) in expected}
    if fixtures.keys() != expected.keys():
        raise ValueError("Missing expected public test pairs")
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true", VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_LONG_CLEAN_SPEECH="true", VERITAS_AUDIO_REFERENCE_DENSE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_EQ_GUARD="true", VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_MATCHING_WAVEFORM="true" if waveform else "false")

    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence.audio_reference import SCHEMA_WITH_ALIGNED_SPEECH, SCHEMA_WITH_MATCHING_WAVEFORM
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as replay
    from scripts.verify_multimedia_exports import scan

    check_key = "matching_waveform" if waveform else "aligned_speech"
    page_title = "Matching speech waveform" if waveform else "Aligned speech comparison"
    schema = SCHEMA_WITH_MATCHING_WAVEFORM if waveform else SCHEMA_WITH_ALIGNED_SPEECH
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

        for (scene, role), status in expected.items():
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
            assert check_key not in plain
            data = {"include_pdf": "true", "earlier_has_clean_speech": "true"}
            payload = body(client.post(endpoint, files=files, data=data))
            comparison = payload["comparison"]
            check = comparison[check_key]
            assert comparison["schema"] == schema and comparison.get("aligned_speech")
            assert check["status"] == status and check["affects_overall_confidence"] is False
            scan(json.dumps(comparison))
            exported = output / f"{name}.json"
            save(exported, {key: payload[key] for key in ("comparison", "assets")})
            assert replay(earlier, submitted, exported, verify=True) == comparison
            pdf = base64.b64decode(payload["pdf"], validate=True)
            (output / f"{name}.pdf").write_bytes(pdf)

            excerpts = {}
            with pymupdf.open(stream=pdf, filetype="pdf") as report:
                pages = [page for page in report if page_title in page.get_text()]
                assert len(pages) == 1
                page = pages[0]
                text = " ".join(page.get_text().split())
                scan(text)
                assert "zero weight" in text and check["summary"] in text
                if status == "localized_difference":
                    assert sum(len(path["items"]) > 20 for path in page.get_drawings()) == 2
                    span = check["range"]
                    if waveform:
                        from scripts.audio_source_isolation_diagnostic import useful

                        assert any(useful(span["submitted"]["samples"], interval) for interval in truth)
                        assert f"{span['matching_share']:.0%} of the weaker surrounding match." in text
                        for role, offset in span["flank_offsets_samples"].items():
                            assert f"{offset / 16:+.1f} ms {role}" in text
                    else:
                        assert "Each surrounding section lasts 400 ms" in text
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
            results[name] = {"truth_samples": truth, "status": status, "reason": check["reason"], "excerpts": excerpts,
                             "source_sha256": [digest(pcm.tobytes()) for pcm in (before, after)]}
    save(output / "receipt.json", {"cases": results, "paid_runs": len(results), "unpaid_jobs": 0,
                                    "overall_assessments_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sounds", type=Path, required=True)
    parser.add_argument("--waveform", action="store_true")
    args = parser.parse_args()
    run(args.output, args.bank, args.sounds, waveform=args.waveform)
