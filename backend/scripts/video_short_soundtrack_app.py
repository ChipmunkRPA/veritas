# SPDX-License-Identifier: Apache-2.0
"""Verify a held short soundtrack shift through prepaid comparison and native export replay."""

import argparse
import base64
import json
import os
from pathlib import Path

from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure
from scripts.video_short_soundtrack_transfer import AAC, OFFSETS, ROOT, qualified, sha


def run(bank, receipt, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_VIDEO_REFERENCE_SOUNDTRACK_REENCODE="true",
                      VERITAS_VIDEO_REFERENCE_SHORT_SOUNDTRACK="true",
                      VERITAS_VIDEO_REFERENCE_PICTURE_INSERTION="false",
                      VERITAS_VIDEO_REFERENCE_LOCAL_AREA="false")

    import pymupdf
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.engine.registry import get_registry
    from app.evidence.video_reference_soundtrack import SHORT_BASIS
    from app.main import app
    from app.storage.db import get_session
    from scripts.compare_video_references import run as replay
    from scripts.verify_multimedia_exports import scan

    result = json.loads(receipt.read_bytes())
    candidates = [(r["group"], case) for r in result["rows"] if r["offset_s"] == OFFSETS[0]
                  for case in r["cases"] if case["carrier"] == AAC and case["role"] == "earlier"
                  and qualified(case["picture"]) and case["old"] is None and case["short"] is not None]
    group, case = candidates[0]
    earlier, submitted = (bank / f"g{group}-{name}.mp4" for name in ("earlier", "submitted"))
    assert sha(submitted) == case["delivery_sha256"]
    registry = get_registry()
    registry.enabled = lambda *_: []
    stub = PaymentStub()
    service._stripe = stub
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-short-soundtrack-trial", "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        aid = checkout(client, stub, submitted, filename=submitted.name)
        endpoint = f"/api/v2/analyses/{aid}/video-reference-comparison"
        assert body(client.get(endpoint))["available"]
        payload = body(client.post(endpoint, data={"include_pdf": "true"},
                                   files={"earlier": (earlier.name, earlier.read_bytes(), "video/mp4")}))
        comparison = payload["comparison"]
        timing = comparison["soundtrack_alignment"]
        assert timing["status"] == "local_shift" and timing["waveform_basis"] == SHORT_BASIS
        assert timing["affects_overall_confidence"] is False
        span = timing["range"]
        assert all(span[k] == value for k, value in case["short"].items())
        before, during, after = [span["paired_pictures"][role]["submitted"]["frame_index"]
                                 for role in ("before", "during", "after")]
        pdf = base64.b64decode(payload["pdf"], validate=True)
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            words = " ".join(" ".join(page.get_text().split()) for page in document)
            scan(words)
            page = next(page for page in document if "Local soundtrack timing" in page.get_text())
            assert "80 ms windows" in page.get_text() and len(page.get_images()) == 6
            assert "earlier between the clips" in page.get_text()
        exported = output / "comparison.json"
        exported.write_text(json.dumps({"comparison": comparison, "assets": payload["assets"]}))
        scan(json.dumps(comparison))
        assert replay(earlier, submitted, exported, verify=True) == comparison
    summary = {"group": group, "case_sha256": case["delivery_sha256"], "offset_ms": timing["offset_ms"],
               "earlier_samples": span["earlier_samples"], "submitted_samples": span["submitted_samples"],
               "before_during_after_frames": [before, during, after], "native_pictures_replayed": 6,
               "native_audio_excerpts_replayed": 2, "simulated_paid_runs": 1, "unpaid_jobs": 0,
               "model_calls": 0, "affects_overall_confidence": False}
    (output / "comparison.pdf").write_bytes(pdf)
    (output / "receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.bank, args.receipt, args.output)
