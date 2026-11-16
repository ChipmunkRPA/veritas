"""Verify the middle-six-second speech check through prepaid local analyses."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import wave
from pathlib import Path
from zipfile import ZipFile

from app.evidence import audio_focused_edit_binding as binding
from scripts.audio_focused_context_transfer import digest, matches, read, save
from scripts.audio_long_context_review import common_truth, predictions
from scripts.audio_two_second_app import render, without_asset_urls
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def trials(panel):
    points = panel["calibration"]["points"]
    rows = [row for row in panel["rows"] if row["condition"] == "overlap_-5dB"]

    def select(row):
        return predictions(row, points, "wider")[0]

    added = []
    for row in rows:
        if row["kind"] != "edited":
            continue
        selected = select(row)
        known = common_truth(row)
        if matches(selected["current_plus_six"], known) - matches(selected["current"], known):
            added.append((row, selected["six"][0]))
    assert len(added) == 3
    first, second = added[:2]
    yield "before", first[0], False, "wav", None
    yield "edited-wav", first[0], True, "wav", first[1]
    yield "edited-flac", second[0], True, "flac", second[1]
    for kind in ("ordinary", "background"):
        row = next(row for row in rows if row["kind"] == kind and any(row["readings"]["short"]) and not select(row)["six"])
        yield kind, row, True, "wav", None


def verify_exports(client, route, path, evidence, raw, signer):
    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    review = evidence[binding.KEY]
    scope = review["checked_scope"]
    caption = f"Search scope {scope['start_sample'] / binding.RATE:.3f}–{scope['end_sample'] / binding.RATE:.3f} s."
    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    assert caption in words and review["reason"] in words
    if review["status"] == "flagged":
        a, b = review["interval"]["start_sample"], review["interval"]["end_sample"]
        selected = fetch(client, review["artifacts"]["selected_clip"]["url"])
        with wave.open(io.BytesIO(selected)) as stream:
            assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, binding.RATE)
            assert stream.readframes(stream.getnframes()) == raw[2 * a:2 * b]
        chart = pixels(fetch(client, review["artifacts"]["waveform"]["url"]))
        pts = f"Source PTS [{a}, {b}) at 1/16000."
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            assert any(caption in " ".join(page.get_text().split()) and pts in " ".join(page.get_text().split())
                       and chart in {((p.width, p.height), digest(p.samples)) for image in page.get_images()
                                     if (p := pymupdf.Pixmap(document, image[0])).n == 3} for page in document)
    package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = path.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    path.with_suffix(".pdf").write_bytes(pdf)
    verified = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(),
                              expect_key_id=signer["key_id"], source_audio=path)
    assert verified.outcome == "verified", verified.as_dict()
    if review["status"] == "flagged":
        assert any("Recomputed the marked focused speech waveform" in item.detail for item in verified.checks)
    with ZipFile(archive) as exported:
        assert json.loads(exported.read("evidence.json")) == evidence
        assert exported.read("report.pdf") == pdf
    return hashlib.sha256(pdf).hexdigest()


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_AUDIO_FOCUSED_LONG_REVIEW="true", VERITAS_AUDIO_TWO_SECOND_LONG_REVIEW="true",
                      VERITAS_AUDIO_TWO_SECOND_LONG_CONFIRMATION="true")
    import torch
    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from fastapi.testclient import TestClient
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    outcomes, baseline, browser_cases = {}, {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-centered-speech", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, row, enabled, carrier, expected in trials(read(args.receipt)):
            settings.audio_centered_speech_edit_review = enabled
            raw = (args.media / "pcm" / f"{row['pcm_sha256']}.pcm").read_bytes()
            assert digest(raw) == row["pcm_sha256"] and len(raw) == 2 * row["samples"]
            media = render(args.output, name, raw, carrier)
            analysis_id = checkout(client, stub, media, kind="audio", media_type=f"audio/{carrier}", filename=media.name)
            route = f"/api/v2/analyses/{analysis_id}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{analysis_id}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for response in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(response.decode())
                assert b"local.centered_speech_edit_v1" not in response and b"centered-speech-head" not in response
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, analysis_id).result_hash))
            baseline[name] = (result["decision"], result.get("customer_assessment"),
                              without_asset_urls(evidence["long_voice_review"]))
            review = evidence.get(binding.KEY)
            assert (review is not None) == enabled
            if not enabled:
                print(f"{name}: previous speech checks exercised after payment", flush=True)
                continue
            binding.validate(review, evidence["source"])
            assert review["schema"] == binding.CENTERED_SCHEMA
            assert review["checked_scope"] == binding.interval_dict(*row["scope"])
            assert review["status"] == ("flagged" if expected else "no_flag"), review
            assert review["interval"] == (binding.interval_dict(*expected) if expected else None)
            pdf_sha256 = verify_exports(client, route, media, evidence, raw, signer)
            outcomes[name] = {"format": carrier, "status": review["status"], "scope_samples": row["scope"],
                              "selected_samples": expected, "pcm_sha256": row["pcm_sha256"], "pdf_sha256": pdf_sha256}
            browser_cases.append({"analysis_id": analysis_id, "name": name, "review": review})
            print(f"{name}: {review['status']} with matching API, PDF and signed audio", flush=True)
        assert baseline["before"] == baseline["edited-wav"]
        browser = browser_check(args.output, app, token, browser_cases, scenario="centered-speech-replay.mjs") if args.browser else None
    save(args.output / "receipt.json", {"cases": outcomes, "browser": browser, "paid_analyses": len(stub.sessions),
                                        "jobs_before_payment": 0, "unchanged_baseline_and_assessment": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("receipt", "media", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--browser", action="store_true")
    run(parser.parse_args())
