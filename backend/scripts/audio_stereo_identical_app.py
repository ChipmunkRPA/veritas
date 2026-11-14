# SPDX-License-Identifier: Apache-2.0
"""Replay a held stereo observation through a simulated prepaid analysis."""

import argparse
import json
import wave
from pathlib import Path
from zipfile import ZipFile

from scripts.audio_stereo_identical_transfer import ROOT, sha, source_paths
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait


def run(bank, receipt, output):
    configure(output, ROOT)

    import numpy as np
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.audio.decode import read_wav
    from app.engine.registry import get_registry
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    panel = json.loads(receipt.read_bytes())
    assert panel["results"]["advance"] and panel["freeze"]["protocol"] == "revision"
    row = next(row for row in panel["rows"] if row["meeting"] == "ES2010a"
               and row["condition"] == "local_left_copy" and row["truth"] in row["measured_ranges"])
    start, end = row["truth"]
    tracks = []
    for name in source_paths(row["meeting"]):
        path = bank / name
        assert sha(path) == panel["freeze"]["sources"][name]["sha256"]
        with wave.open(str(path), "rb") as source:
            source.setpos(row["excerpt_start_s"] * 16000)
            tracks.append(np.frombuffer(source.readframes(48000), dtype="<i2"))
    pcm = np.column_stack(tracks)
    pcm[start:end, 1] = pcm[start:end, 0]
    path = output / "controlled-channel-edit.wav"
    with wave.open(str(path), "wb") as target:
        target.setparams((2, 2, 16000, 0, "NONE", ""))
        target.writeframes(pcm.astype("<i2", copy=False).tobytes())
    assert sha(path) == row["source_sha256"]
    registry = get_registry()
    registry.enabled = lambda *_: []  # Exercise signal evidence without voice origin inference.
    stub = PaymentStub()
    service._stripe = stub
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-identical-stereo-trial", "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        aid = checkout(client, stub, path, kind="audio", media_type="audio/wav", filename=path.name)
        route = f"/api/v2/analyses/{aid}"
        evidence = body(client.get(f"{route}/evidence-packets"))
        key = "identical_channel_interval"
        stereo = evidence["stereo_context"]
        assert stereo["stereo_pcm_sha256"] == row["decoded_stereo_sha256"]
        observed = stereo[key]
        assert observed["affects_overall_confidence"] is False
        (selected,) = observed["selected"]
        assert [selected["start_sample"], selected["end_sample"]] == row["truth"]
        assert selected["source_timestamp_ranges"] == [
            {"start_sample": start, "end_sample": end, "source_start_pts": start, "source_end_pts": end, "time_base": "1/16000"}
        ]
        (listening,) = evidence["stereo_listening"][key]
        for index, label in enumerate(("left", "right")):
            asset = listening["artifacts"][label]
            track = read_wav(fetch(client, asset["url"]))
            np.testing.assert_array_equal(track, pcm[start - 1280:end + 1280, index] / 32768)
        exported = fetch(client, f"/api/analyses/{aid}/export.json")
        assert json.loads(exported)["evidence"] == evidence
        scan(exported.decode())
        report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
        report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
        pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
        words = " ".join(pdf_text(pdf).split())
        scan(words)
        assert f"Channels match, decoded audio {start / 16000:.3f}–{end / 16000:.3f} s" in words
        assert f"/stereo_context/{key}/selected/0" in words
        package = body(client.post(f"{route}/packages",
            json={"report_id": report["id"], "kind": "draft", "include_original": False}))
        package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
        signed = output / "signed-package.zip"
        signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
        signer = body(client.get("/api/v2/instance/public-key"))
        replay = verify_package(signed, public_key_pem=signer["public_key_pem"].encode(),
                                expect_key_id=signer["key_id"], source_audio=path)
        assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
        assert any(item.ok and "Recomputed identical-channel audio and validated 1 selected intervals" in item.detail
                   for item in replay.checks), replay.as_dict()
        with ZipFile(signed) as archive:
            assert json.loads(archive.read("evidence.json")) == evidence
        (output / "report.pdf").write_bytes(pdf)
        (output / "export.json").write_bytes(exported)
    summary = {"simulated_paid_runs": 1, "unpaid_jobs": 0, "source_decoded_samples": 48000,
               "source_interval_samples": row["truth"], "matched_channel_wavs": 2,
               "source_replay_verified": True, "affects_overall_confidence": False, "voice_model_calls": 0}
    (output / "receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.bank, args.receipt, args.output)
