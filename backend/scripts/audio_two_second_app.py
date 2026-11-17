# SPDX-License-Identifier: MIT
"""Replay the longer-file two-second review through checkout and signed evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import wave
from pathlib import Path
from zipfile import ZipFile

from app.evidence import audio_two_second_binding as binding
from scripts import audio_focused_context_transfer as prior
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]


def cases(receipt):
    def selected(row):
        return sorted({(reading["short"][0] + reading["start_sample"], reading["short"][1] + reading["start_sample"])
                       for reading in row["excerpts"] if reading["short"]})

    def target(condition):
        return next(row for row in receipt["rows"] if row["treatment"] == condition and row["kind"] == "edited"
                    and prior.matches(prior.proposed(row)["current_plus_short"], row["truth"])
                    - prior.matches(prior.proposed(row)["current"], row["truth"]))

    clean = target("clean")
    fixtures = [("clean-before", clean, False, "wav"), ("clean-edited", clean, True, "wav"),
                ("noisy-edited", target("overlap_-10dB"), True, "flac")]
    fixtures += [(kind, next(row for row in receipt["rows"] if row["kind"] == kind
                            and any(e["short_assessed"] for e in row["excerpts"]) and not selected(row)), True, "wav")
                 for kind in ("ordinary", "background")]
    for name, row, enabled, carrier in fixtures:
        yield name, row, carrier, binding.SCHEMA if enabled else None, selected(row), []


def confirmation_cases(receipt):
    from scripts import audio_two_second_transfer as study

    rows = receipt["rows"]
    assert study.summarize(receipt["freeze"], rows) == receipt["summary"]

    def plan(row, schema):
        readings = binding.summarize_readings(row["excerpts"], row["samples"], schema=schema)
        return binding.regions(readings, schema=schema)

    def trial(name, row, carrier, schema):
        regions = plan(row, schema)
        support = [span(region["confirming_window"]) for region in regions] if schema == binding.CONFIRMED_SCHEMA else []
        return name, row, carrier, schema, [span(region) for region in regions], support

    def target(condition):
        for row in rows:
            if row["condition"] == condition and row["kind"] == "edited":
                arms = study.locations(row)
                if prior.matches(arms["adjacent_support"], row["truth"]) - prior.matches(arms["current"], row["truth"]):
                    return row
        raise ValueError(f"No additionally located edit under {condition}")

    confirmed = binding.CONFIRMED_SCHEMA
    strong = target("overlap_-10dB")
    yield trial("strong-single-window", strong, "flac", binding.SCHEMA)
    yield trial("strong-confirmed", strong, "flac", confirmed)
    yield trial("moderate-confirmed", target("overlap_-5dB"), "flac", confirmed)
    yield trial("clean-confirmed", target("clean"), "wav", confirmed)
    for kind in ("ordinary", "background"):
        control = next(row for row in rows if row["kind"] == kind and row["condition"] == "overlap_-5dB"
                       and plan(row, binding.SCHEMA) and not plan(row, confirmed))
        yield trial(f"{kind}-single-window", control, "wav", binding.SCHEMA)
        yield trial(f"{kind}-confirmed", control, "wav", confirmed)


def span(window):
    return window["start_sample"], window["end_sample"]


def render(folder, name, raw, carrier):
    wav = folder / f"{name}.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, binding.RATE, len(raw) // 2, "NONE", "not compressed"))
        stream.writeframes(raw)
    if carrier == "wav":
        return wav
    result = wav.with_suffix(".flac")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "flac", str(result)], check=True)
    return result


def without_asset_urls(value):
    if isinstance(value, dict):
        return {key: without_asset_urls(item) for key, item in value.items() if key != "url"}
    if isinstance(value, list):
        return [without_asset_urls(item) for item in value]
    return value


def verify_exports(client, route, media, evidence, pcm, signer):
    import pymupdf

    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan

    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    words = " ".join(pdf_text(pdf).split())
    scan(words)
    presented = binding.KEY in evidence
    review = evidence[binding.KEY] if presented else evidence["long_voice_review"][binding.KEY]
    assert binding.TITLE in words and review["reason"] in words
    assert words.count("Confirming response") == sum("confirming_window" in region and "same_pair_as" not in region
                                                     for region in review["regions"])
    with pymupdf.open(stream=pdf, filetype="pdf") as view:
        pages = [
            (" ".join(page.get_text().split()), {((pix.width, pix.height), hashlib.sha256(pix.samples).hexdigest())
              for image in page.get_images() if (pix := pymupdf.Pixmap(view, image[0])).n == 3})
            for page in view
        ]
        for region in review["regions"]:
            witnesses = []
            for window in binding.media_windows({"regions": [region]}):
                start, end = span(window)
                clip = fetch(client, window["artifacts"]["selected_clip"]["url"])
                with wave.open(io.BytesIO(clip), "rb") as stream:
                    assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (binding.RATE, 1, 2)
                    assert stream.readframes(stream.getnframes()) == pcm[2 * start:2 * end]
                chart = pixels(fetch(client, window["artifacts"]["waveform"]["url"]))
                caption = (f"Original source ticks [{start * 3}, {end * 3}) at 1/48000." if presented
                           else f"Source PTS [{start}, {end}) at 1/16000.")
                witnesses.append((caption, chart))
            assert any(all(caption in text and chart in charts for caption, chart in witnesses)
                       and (len(witnesses) == 1 or "Confirming response" in text) for text, charts in pages)
    bundle = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{bundle['id']}", timeout=120)
    media.with_suffix(".pdf").write_bytes(pdf)
    archive = media.with_suffix(".zip")
    archive.write_bytes(fetch(client, f"/api/v2/packages/{bundle['id']}/file"))
    checked = verify_package(archive, public_key_pem=signer["public_key_pem"].encode(),
                             expect_key_id=signer["key_id"], source_audio=media)
    assert checked.outcome == "verified", checked.as_dict()
    check_text = ("original 48 kHz presentation clock" if presented
                  else f"{len(review['regions'])} marked two-second windows")
    assert any(check_text in check.detail for check in checked.checks)
    if review["schema"] == binding.CONFIRMED_SCHEMA and review["regions"]:
        assert any(f"{len(review['regions'])} confirming two-second windows" in check.detail for check in checked.checks)
    with ZipFile(archive) as exported:
        assert json.loads(exported.read("evidence.json")) == evidence
        assert exported.read("report.pdf") == pdf
    return hashlib.sha256(pdf).hexdigest()


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_FOCUSED_LONG_REVIEW="true", VERITAS_LOG_LEVEL="WARNING")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = prior.read(args.receipt)
    if not args.confirmation:
        prior.validate(frozen)
    simulation = PaymentStub()
    service._stripe = simulation
    observed, before, browser_cases = {}, {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-longer-two-second-review", "examiner")
            db.add(owner)
            db.flush()
            token, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = token
        signer = body(client.get("/api/v2/instance/public-key"))
        for name, row, carrier, schema, expected, expected_support in (confirmation_cases(frozen) if args.confirmation else cases(frozen)):
            raw = (args.media / "pcm" / f"{row['pcm_sha256']}.pcm").read_bytes()
            assert prior.digest(raw) == row["pcm_sha256"] and len(raw) == 2 * row["samples"]
            settings.audio_two_second_long_review = schema is not None
            settings.audio_two_second_long_confirmation = schema == binding.CONFIRMED_SCHEMA
            media = render(args.output, name, raw, carrier)
            aid = checkout(client, simulation, media, kind="audio", media_type=f"audio/{carrier}", filename=media.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            assert json.loads(exported)["evidence"] == evidence
            for response in (exported, fetch(client, route), fetch(client, f"{route}/detector-runs")):
                scan(response.decode())
                assert binding.DETECTOR_ID.encode() not in response and b'"logit"' not in response
            with get_session() as db:
                result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
            long = evidence["long_voice_review"]
            baseline = without_asset_urls({k: v for k, v in long.items() if k != binding.KEY})
            assessment = result["decision"], result.get("customer_assessment"), baseline
            before[name] = assessment
            review = long.get(binding.KEY)
            assert (review is None) == (schema is None)
            if review is None:
                continue
            binding.validate(review, evidence["source"])
            assert review["schema"] == schema
            assert review["status"] == ("flagged" if expected else "no_flag")
            assert [span(region) for region in review["regions"]] == expected
            assert [span(region["confirming_window"]) for region in review["regions"] if "confirming_window" in region] == expected_support
            sha_pdf = verify_exports(client, route, media, evidence, raw, signer)
            browser_cases.append({"analysis_id": aid, "name": name, "review": review})
            observed[name] = {"container": carrier, "status": review["status"], "ranges": expected, "pdf_sha256": sha_pdf}
            if args.confirmation:
                observed[name].update(schema=schema, confirming_ranges=expected_support, source_sha256=row["pcm_sha256"],
                                      condition=row["condition"], known_edit_ranges=row["truth"])
            print(f"{name}: {review['status']} with matching API, PDF and signed source playback", flush=True)
        if args.confirmation:
            for case in ("strong", "ordinary", "background"):
                assert before[f"{case}-single-window"] == before[f"{case}-confirmed"]
        else:
            assert before["clean-before"] == before["clean-edited"]
        browser = browser_check(args.output, app, token, browser_cases, scenario="long-two-second-speech-replay.mjs") if args.browser else None
    prior.save(args.output / "receipt.json", {"cases": observed, "browser": browser, "paid_analyses": len(simulation.sessions),
                                             "jobs_before_payment": 0, "same_assessment_and_baseline": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("receipt", "media", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--confirmation", action="store_true", help="Replay the two-window rule on the fresh numeric comparison")
    run(parser.parse_args())
