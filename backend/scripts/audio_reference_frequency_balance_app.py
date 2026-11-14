# SPDX-License-Identifier: Apache-2.0
"""Verify earlier-copy speech frequency evidence through the simulated prepaid app."""

import argparse
import base64
import json
import os
from pathlib import Path

from scripts.audio_focused_window_app import checkout, write_wav
from scripts.audio_reference_variable_window import pairs
from scripts.audio_vocal_event_mixtures import digest, save
from scripts.video_short_return_encoded_app import PaymentStub, body, configure

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    (2, "train", "eq"): ("localized_difference", "broad_processing_difference"),
    (2, "train", "piper"): ("localized_difference", "localized_difference"),
    (2, "train", "untouched"): ("no_local_difference", "no_local_difference"),
}


def run(output, bank, sounds):
    fixtures = {row[:3]: row[3:] for row in pairs("assessment", bank, sounds) if row[:3] in CASES}
    assert fixtures.keys() == CASES.keys()
    run_cases(output, fixtures)


def run_cases(output, fixtures):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true", VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_LONG_CLEAN_SPEECH="true", VERITAS_AUDIO_REFERENCE_DENSE_CLEAN_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_EQ_GUARD="false")

    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.audio_reference import SCHEMA_WITH_CLEAN_SPEECH_SUPPORT
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as replay
    from scripts.verify_audio_clean_speech_export import verify_export
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-reference-frequency-balance", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for key, expected in CASES.items():
            _, _, kind = key
            truth, original, noisy = fixtures[key]
            earlier, submitted = (output / f"{kind}-{side}.wav" for side in ("earlier", "submitted"))
            write_wav(earlier, original.tobytes())
            write_wav(submitted, noisy.tobytes())
            aid, _ = checkout(client, stub, submitted)
            with get_session() as db:
                frozen = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/audio-reference-comparison"
            files = {"earlier": (earlier.name, earlier.read_bytes(), "audio/wav")}
            form = {"include_pdf": "true", "earlier_has_clean_speech": "true"}
            observed = []
            for active, status in zip((False, True), expected, strict=True):
                settings.audio_reference_clean_speech_eq_guard = active
                payload = body(client.post(endpoint, files=files, data=form))
                comparison = payload["comparison"]
                check = comparison["clean_speech"]
                assert check["status"] == status and check["affects_overall_confidence"] is False, check
                observed.append(check)
                if not active:
                    continue
                assert comparison["schema"] == SCHEMA_WITH_CLEAN_SPEECH_SUPPORT
                assert check["match_support"]["admitted"] is True
                scan(json.dumps(comparison))
                exported = output / f"{kind}.json"
                save(exported, {name: payload[name] for name in ("comparison", "assets")})
                assert replay(earlier, submitted, exported, verify=True) == comparison
                verification = verify_export(earlier, submitted, exported)
                pdf = base64.b64decode(payload["pdf"], validate=True)
                (output / f"{kind}.pdf").write_bytes(pdf)
                with pymupdf.open(stream=pdf, filetype="pdf") as report:
                    page = " ".join(report[1].get_text().split())
                    scan(page)
                    assert "Local speech comparison" in page and "zero weight" in page
                    support = check["match_support"]
                    assert f"Matching sound in {support['matching_windows']} of {support['active_windows']} active windows" in page
                    if status == "broad_processing_difference":
                        assert "Frequency balance across the recording" in page and "4 to 6 s" in page
                        assert "Selected interval" not in page and "range" not in check
                        assert check["frequency_balance"]["section_start_samples"] == [0, 32000, 64000]
                    elif status == "localized_difference":
                        assert observed[0]["range"] == check["range"]
                        assert "Selected interval" in page and verification["excerpts_verified"] == 2
                with get_session() as db:
                    assert db.get(Analysis, aid).result_hash == frozen
                results[kind] = {"truth_samples": truth, "before": expected[0], "after": status, "match_support": check["match_support"],
                                 "source_replay": verification, "frequency_balance": check.get("frequency_balance"),
                                 "earlier_sha256": digest(original.tobytes()), "submitted_sha256": digest(noisy.tobytes())}
    save(output / "receipt.json", {"cases": results, "paid_runs": len(results), "unpaid_jobs": 0,
                                    "overall_assessments_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sounds", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.bank, args.sounds)
