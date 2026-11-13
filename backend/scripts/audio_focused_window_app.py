# SPDX-License-Identifier: Apache-2.0
"""Exercise focused long speech evidence through the isolated paid audio worker."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from scripts.video_short_return_encoded_app import PaymentStub, body, configure, counts, event, fetch, wait

RATE = 16_000
FOCUS = (48_000, 72_000)
SCHEMA = "chipmunk.long-voice-review/2"
OLD_SCHEMA = "chipmunk.long-voice-review/1"
CASES = {"rain_edit_wav": (19, "wav"), "rain_edit_flac": (19, "flac"), "genuine_rain": (55, "wav"),
         "rain_only": (90, "wav"), "sparse_opt_out": (19, "wav"), "sparse_only_retained": (45, "wav")}
SPARSE_RANGES = [(0, 48_000), (118_709, 166_709)]
ROOT = Path(__file__).resolve().parents[2]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def fixture_plan(study):
    freeze = read(study / "freeze.json")
    roster = read(study / "roster.json")
    receipt = read(study / "receipt.json")
    assert receipt["freeze"] == freeze and receipt["roster"] == roster and receipt["result"]["passed"] is True
    assert digest((study / "roster.json").read_bytes()) == freeze["roster_sha256"]
    plan = {}
    for index, _ in CASES.values():
        row, response = roster[index], receipt["responses"][index]
        raw = (study / "pcm" / f"{index:03}.pcm").read_bytes()
        assert row["case"] == response["case"] == index
        assert row["scene"] == ("gain_control" if index == 45 else "rain")
        assert len(raw) == 2 * row["samples"] and digest(raw) == row["pcm_sha256"]
        readings = {}
        for window in response["windows"]:
            a, b = window["span"]
            assert digest(raw[2*a:2*b]) == window["pcm_sha256"]
            readings[a, b] = window["direction"]
        plan[index] = (row, raw, readings)
    assert plan[19][0]["source"]["edits"] and all(value != "ai" for value in plan[55][2].values())
    assert plan[55][0]["source"]["label"] == "genuine" and plan[90][0]["source"] is None
    assert set(plan[90][2].values()) == {"unassessed"}
    assert all(plan[45][2][window] == "ai" for window in SPARSE_RANGES)
    return plan


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def write_wav(path, raw):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(raw)


def fixture(folder, case, container, pcm):
    path = folder / f"rain-study-{case:03}.{container}"
    if path.exists():
        return path
    if container == "wav":
        write_wav(path, pcm)
    else:
        wav = fixture(folder, case, "wav", pcm)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(wav),
                        "-map", "0:a:0", "-c:a", "flac", "-sample_fmt", "s16", str(path)], check=True)
    return path


def checkout(client, stub, source):
    from app.billing import purchase_notice

    before = counts()
    order = body(client.post("/api/v2/billing/checkout", headers={"Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "audio", "profile": "full", "risk_accepted": "true",
              "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256},
        files={"file": (source.name, source.read_bytes(), f"audio/{source.suffix[1:]}")}))
    assert counts() == before
    sid = next(reversed(stub.sessions))
    assert (stub.sessions[sid]["amount_total"], stub.sessions[sid]["currency"]) == (999, "usd")
    nonce = uuid.uuid4().hex
    assert event(client, sid, nonce + "unpaid")["received"] and counts() == before
    stub.sessions[sid].update(payment_status="paid", status="complete")
    assert event(client, sid, nonce + "paid")["received"] and event(client, sid, nonce + "paid")["received"]
    order = body(client.get(f"/api/v2/billing/orders/{order['order_id']}"))
    assert order["status"] == "fulfilled" and counts() == tuple(value + 1 for value in before)
    aid = order["analysis_id"]
    result = wait(client, f"/api/analyses/{aid}")
    return aid, result


def span(item):
    return item["start_sample"], item["end_sample"]


def timestamps(a, b):
    return [{"start_sample": a, "end_sample": b, "source_start_pts": a, "source_end_pts": b, "time_base": "1/16000"}]


def verify_windows(review, readings, pcm, focused):
    from app.evidence.audio_long_review_binding import intervals

    windows = review["windows"]
    ranges = [span(row) for row in windows]
    expected = sorted(readings) if focused else intervals(len(pcm))
    assert ranges == expected, (ranges, expected)
    for row in windows:
        a, b = span(row)
        assert row["direction"] == readings[a, b]
        assert row["source_timestamp_ranges"] == timestamps(a, b)
        assert row["pcm_sha256"] == digest(pcm[a:b].astype("<f4").tobytes())
    return windows


def signature(raw):
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        return rgb.size, digest(rgb.tobytes())


def focused_asset_proof(client, review, region, basis, target, supporting, pcm, raw):
    from app.evidence.audio_packets import _chart

    a, b = span(region)
    assert region["basis"] == basis and (a, b) == target
    assert region["source_timestamp_ranges"] == timestamps(a, b)
    assert region["pcm_sha256"] == digest(pcm[a:b].astype("<f4").tobytes())
    assert [span(review["windows"][index]) for index in region["supporting_windows"]] == supporting
    assert all(review["windows"][index]["direction"] == "ai" for index in region["supporting_windows"])
    assert set(region["artifacts"]) == {"selected_clip", "waveform"}
    for key, asset in region["artifacts"].items():
        assert asset["sha256"] == digest(fetch(client, asset["url"])), key
    wav = fetch(client, region["artifacts"]["selected_clip"]["url"])
    with wave.open(io.BytesIO(wav), "rb") as payload:
        assert (payload.getframerate(), payload.getsampwidth(), payload.getnchannels(), payload.getnframes()) == (RATE, 2, 1, b-a)
        assert payload.readframes(payload.getnframes()) == raw[2*a:2*b]
    png = fetch(client, region["artifacts"]["waveform"]["url"])
    assert png == _chart(pcm, [], start=0, end=len(pcm), selected=(a, b), scores=False)
    assert signature(png)[0] == (960, 280)
    return wav, png


def verify_exports(client, aid, source, evidence, assets, output):
    import pymupdf

    from app.reports import engine
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    output.mkdir(mode=0o700)
    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    scan(exported.decode())
    assert json.loads(exported)["evidence"] == evidence
    (output / "export.json").write_bytes(exported)
    for route in ("signals", "detector-runs", "findings"):
        scan(fetch(client, f"/api/v2/analyses/{aid}/{route}").decode())
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    (output / "report.pdf").write_bytes(pdf)
    wav, png = assets
    (output / "interval.wav").write_bytes(wav)
    (output / "waveform.png").write_bytes(png)
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = " ".join(" ".join(page.get_text().split()) for page in doc)
        scan(text)
        assert ("Overlap of two AI-leaning windows. 3.000–4.500 s. Samples [48000, 72000) at 16000 Hz. "
                "Source PTS [48000, 72000) at 1/16000." in text)
        images = {item[0] for page in doc for item in page.get_images(full=True)}
        assert signature(png) in {signature(doc.extract_image(xref)["image"]) for xref in images}
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
        json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    path = output / "signed-package.zip"
    path.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    signer = body(client.get("/api/v2/instance/public-key"))
    key = signer["public_key_pem"].encode()
    proof = verify_package(path, public_key_pem=key, expect_key_id=signer["key_id"], source_audio=source)
    assert proof.ok and proof.signer_authenticated, proof.as_dict()
    assert any(item.ok and item.name == "original audio samples" and "against the same original samples" in item.detail
               for item in proof.checks)
    with ZipFile(path) as archive:
        bundle = json.loads(archive.read("bundle.json"))
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read(bundle["reports"][0]["file"]) == pdf
        for role, asset in evidence["long_voice_review"]["regions"][0]["artifacts"].items():
            [entry] = [name for name in archive.namelist() if Path(name).name.startswith(asset["sha256"] + ".")]
            assert archive.read(entry) == (wav if role == "selected_clip" else png)
    wrong = output / "wrong-original.wav"
    wrong.write_bytes(source.read_bytes() + b"\n")
    assert not verify_package(path, public_key_pem=key, source_audio=wrong).ok
    return {"pdf_api_json_agree": True, "template_set": engine.TEMPLATE_SET, "signed_source_replay": True,
            "wrong_original_rejected": True, "selected_clip_sha256": digest(wav), "waveform_sha256": digest(png)}


def assess(client, aid, media, expected, enabled):
    import numpy as np

    from app.detectors.audio.decode import decode_audio

    row, raw, readings = expected
    sha = digest(media.read_bytes())
    decoded = decode_audio(media, source_sha256=sha)
    np.testing.assert_array_equal(decoded.pcm, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
    evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
    assert evidence["source"]["sha256"] == sha and evidence["source"]["samples"] == row["samples"]
    review = evidence["long_voice_review"]
    assert review["schema"] == (SCHEMA if enabled else OLD_SCHEMA) and review["affects_overall_confidence"] is False
    windows = verify_windows(review, readings, decoded.pcm, enabled)
    positive = enabled and row["case"] in (19, 45)
    assert review["status"] == ("corroborated" if positive else "no_corroboration")
    assert bool(review.get("regions")) == positive
    if row["case"] == 90:
        assert {reading["direction"] for reading in windows} == {"unassessed"}
    if row["case"] == 55:
        assert any(reading["direction"] != "unassessed" for reading in windows)
    assets = None
    if positive and row["case"] == 19:
        [region] = review["regions"]
        assets = focused_asset_proof(client, review, region, "overlap", FOCUS, [(24_000, 72_000), (48_000, 96_000)], decoded.pcm, raw)
    if positive and row["case"] == 45:
        assert [span(region) for region in review["regions"]] == SPARSE_RANGES
        for region, target in zip(review["regions"], SPARSE_RANGES, strict=True):
            focused_asset_proof(client, review, region, "sparse", target, SPARSE_RANGES, decoded.pcm, raw)
    summary = {"case": row["case"], "container": media.suffix[1:], "file_sha256": sha, "pcm_sha256": row["pcm_sha256"],
               "status": review["status"], "schema": review["schema"], "windows": len(windows),
               "regions": [{"samples": list(span(region)), "basis": region["basis"]} for region in review.get("regions", [])]}
    return evidence, summary, assets


def run(study, output):
    plan = fixture_plan(study)
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_FOCUSED_LONG_REVIEW="true", VERITAS_JOB_TIMEOUT_S="240",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store

    assert settings.data_dir == (output / "app").resolve() and settings.audio_focused_long_review
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    rendered, decisions, cases = {}, {}, {}
    inputs = output / "fixtures"
    inputs.mkdir(mode=0o700)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("focused-audio-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for label, (index, container) in CASES.items():
            enabled = label != "sparse_opt_out"
            settings.audio_focused_long_review = enabled
            source = fixture(inputs, index, container, plan[index][1])
            aid, _ = checkout(client, stub, source)
            with get_session() as db:
                analysis = db.get(Analysis, aid)
                result = json.loads(default_store().read_bytes(analysis.result_hash))
                assert analysis.profile == "full"
                decisions[label] = result["decision"]
            evidence, summary, assets = assess(client, aid, source, plan[index], enabled)
            cases[label] = summary
            if assets:
                rendered[label] = tuple(digest(asset) for asset in assets)
            if label == "rain_edit_wav":
                summary.update(verify_exports(client, aid, source, evidence, assets, output / label))
                write_json(output / "browser-input.json", {"analysis_id": aid, "title": evidence["long_voice_review"]["title"]})
            print(label, summary, flush=True)
    assert rendered["rain_edit_wav"] == rendered["rain_edit_flac"]
    assert decisions["rain_edit_wav"] == decisions["sparse_opt_out"]
    write_json(output / "receipt.json", {"cases": cases, "simulated_paid_jobs": len(cases), "unpaid_jobs": 0,
        "exact_wav_flac_parity": True, "overall_decision_identical_flag_on_off": True})


def browser(output, frontend_port):
    target = read(output / "browser-input.json")
    receipt = read(output / "receipt.json")
    assert receipt["simulated_paid_jobs"] == len(CASES)
    data = output / "app"
    database = data / "veritas.db"
    assert database.is_file()
    os.environ.update(VERITAS_DATA_DIR=str(data), VERITAS_DATABASE_URL=f"sqlite:///{database}",
        VERITAS_MODELS_DIR=str(ROOT / "models"), VERITAS_HF_HOME=str(ROOT / "models/hf"),
        VERITAS_DEVICE="cpu", VERITAS_OFFLINE="true", VERITAS_RUN_WORKER="false", VERITAS_SINGLE_USER="false",
        VERITAS_API_KEY=secrets.token_hex(32), VERITAS_API_KEY_PEPPER=secrets.token_hex(32),
        VERITAS_SIGNING_KEY=secrets.token_hex(32), VERITAS_BILLING_MODE="disabled",
        MPLCONFIGDIR=str(output / "matplotlib"))

    import uvicorn
    from sqlalchemy import select

    from app.api import auth
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, PrincipalRow, get_session

    assert settings.data_dir == data and not settings.run_worker
    with get_session() as db:
        analysis = db.get(Analysis, target["analysis_id"])
        [owner] = db.scalars(select(PrincipalRow).where(PrincipalRow.username.like("focused-audio-test-%"))).all()
        assert analysis.submitted_by == owner.id
        api_key, _ = auth.issue_api_key(db, owner)
        db.commit()
    frontend = ROOT / "frontend"
    with socket.socket() as api, (output / "vite.log").open("w") as log:
        api.bind(("127.0.0.1", 0))
        api.listen()
        api_port = api.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [api]}, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{frontend_port}"
        with subprocess.Popen(["node", "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--strictPort",
                                "--port", str(frontend_port)], cwd=frontend, stdout=log, stderr=log,
                               env={**os.environ, "VERITAS_API_TARGET": f"http://127.0.0.1:{api_port}"}) as process:
            try:
                for _ in range(80):
                    if process.poll() is not None:
                        raise RuntimeError("The isolated frontend exited before becoming ready")
                    if server.started:
                        try:
                            with urlopen(url, timeout=.5):
                                break
                        except OSError:
                            pass
                    time.sleep(.25)
                else:
                    raise TimeoutError("The isolated frontend did not start")
                payload = {**target, "api_key": api_key, "root": str(ROOT), "url": url}
                test = subprocess.run(["node", str(Path(__file__).with_name("audio_focused_window_browser.cjs"))],
                                      input=json.dumps(payload), text=True, capture_output=True, timeout=60, check=False)
                if test.returncode:
                    raise RuntimeError(test.stderr + test.stdout)
                result = json.loads(test.stdout)
                write_json(output / "browser-receipt.json", result)
                print(json.dumps(result), flush=True)
            finally:
                process.terminate()
                process.wait(timeout=10)
                server.should_exit = True
                worker.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    paid = subparsers.add_parser("paid", help="Create a new isolated paid replay")
    paid.add_argument("study", type=Path, help="Retained PCM and frozen evaluation receipt")
    paid.add_argument("output", type=Path, help="New isolated run directory")
    chrome = subparsers.add_parser("browser", help="Open a completed paid replay on localhost")
    chrome.add_argument("output", type=Path, help="Completed isolated run directory")
    chrome.add_argument("--port", type=int, default=5268)
    args = parser.parse_args()
    if args.phase == "paid":
        run(args.study.resolve(), args.output.resolve())
    else:
        browser(args.output.resolve(), args.port)


if __name__ == "__main__":
    main()
