# SPDX-License-Identifier: Apache-2.0
"""Verify paid local stereo Opus evidence, including an unsupported delivery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.audio_native_alignment_app import AAC_DESCRIPTION, verify_exports
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE_HEADING = "Stereo Opus repeat check unavailable"


def run(fixtures, output, *, browser):
    metadata = read(fixtures / "metadata.json")
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_PUBLIC_PDF_LAYOUT="standard",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.stereo_repetition_binding import OPUS_ALIGNED_SCHEMA, OPUS_UNAVAILABLE
    from app.main import app
    from app.reports.engine import pdf_text
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    stub = PaymentStub()
    service._stripe = stub
    records, browser_cases, decisions = [], [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-opus-stereo-repeat", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        signer = body(client.get("/api/v2/instance/public-key"))
        assert settings.audio_stereo_opus_repeat_review is False
        for case, fixture in (("off", "gain"), ("on", "gain"), ("ordinary", "ordinary"), ("unsupported", "variable")):
            settings.audio_stereo_opus_repeat_review = case != "off"
            source = fixtures / f"{fixture}.ogg"
            assert digest(source) == metadata[fixture]["sha256"]
            aid = checkout(client, stub, source, kind="audio", media_type="audio/ogg", filename=source.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for public in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(public.decode())
            unsupported = case == "unsupported"
            assert evidence.get(OPUS_UNAVAILABLE) is (True if unsupported else None)
            repeat = evidence.get("stereo_repetition")
            assert bool(repeat) == (case in {"on", "ordinary"})
            check = repeat["aligned_measurement"] if repeat else None
            ranges, times = [], []
            if repeat:
                assert repeat["schema"] == OPUS_ALIGNED_SCHEMA and repeat["affects_overall_confidence"] is False
                assert check["sample_rate"] == 48_000 and check["input_samples"] == evidence["source"]["samples"] * 3
                ranges = [[[pair[side]["start_sample"], pair[side]["end_sample"]] for side in ("first", "second")]
                          for pair in check["pairs"]]
                assert ranges == metadata[fixture]["ranges"]
                times = [[pair[side]["source_timestamp_ranges"] for side in ("first", "second")] for pair in check["pairs"]]
                if case == "on":
                    browser_cases.append({"analysis_id": aid, "name": case, "pairs": check["pairs"],
                                          "description": AAC_DESCRIPTION})
            if unsupported:
                browser_cases.append({"analysis_id": aid, "name": case, "unavailable": True})
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            decisions[case] = (result["decision"], result.get("customer_assessment"))
            report_sha = verify_exports(client, route, case, source, evidence, check, signer, output, None,
                                        description=AAC_DESCRIPTION)
            words = " ".join(pdf_text((output / f"{case}.pdf").read_bytes()).split())
            assert (UNAVAILABLE_HEADING in words) == unsupported
            records.append({"case": case, "source_sha256": digest(source), "measured": check is not None,
                            "unavailable": unsupported, "ranges": ranges, "source_times": times,
                            "signed_pdf_sha256": report_sha})
            print(case, "verified", ranges, flush=True)
        assert decisions["off"] == decisions["on"] and len(stub.sessions) == len(records)
        browser_results = browser_check(output, app, key, browser_cases, scenario="native-stereo-repeat.mjs") if browser else None
    save(output / "receipt.json", {"schema": "chipmunk.opus-stereo-local-checkout/1", "cases": records,
                                   "same_assessment_with_option_off": True, "simulated_payment_usd_cents": 999,
                                   "jobs_before_payment": 0, "browser": browser_results})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixtures", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--browser", action="store_true")
    run(**vars(cli.parse_args()))
