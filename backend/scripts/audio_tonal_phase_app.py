"""Replay the source-linked tone phase observation through a simulated prepaid analysis."""

from __future__ import annotations

import argparse
import io
import json
import os
import wave
from pathlib import Path

from app.evidence import audio_tonal_phase_binding as tone
from scripts import audio_persistent_tone_transfer as transfer
from scripts.audio_focused_window_app import digest, read, write_json, write_wav
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch

ROOT = Path(__file__).resolve().parents[2]


def recordings(receipt, shards, hums, sounds):
    rows = receipt["rows"]
    edit = next(row for row in rows if row["case"]["role"] == "mixed_cut"
                and (match := row["ranges"]["harmonic"]) is not None and match["score"] > tone.POINT
                and transfer.first.useful(match["tones"][0]["samples"], row["case"]["truth"]))
    selected = {"measured_change": edit}
    for state, role in (("no_supported_change", "untouched"), ("unassessed", "ordinary_clean")):
        selected[state] = next(row for row in rows if row["case"]["slot"] == edit["case"]["slot"]
                               and row["case"]["scene"] == edit["case"]["scene"] and row["case"]["role"] == role)

    inputs = {}
    for case, data in transfer.samples(receipt["freeze"], shards, hums, sounds):
        for state, row in selected.items():
            if case == row["case"]:
                raw = data.tobytes()
                assert digest(raw) == row["sha256"]
                inputs[state] = (row, raw)
    assert inputs.keys() == selected.keys()
    return inputs


def verify_outputs(client, aid, source, folder, key, expected, row, raw):
    from app.reports.verify import verify_package
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import run

    checks = run(client, aid, folder, default_store(), key)
    source_check = verify_package(folder / "evidence.zip", public_key_pem=key, source_audio=source)
    assert source_check.ok, source_check.as_dict()
    assert any(item.ok and item.name == "original audio samples" and "tone phase" in item.detail
               for item in source_check.checks)
    evidence = read(folder / "evidence.json")["acoustic_context"]
    assert evidence["schema"] == tone.SCHEMA and evidence[tone.KEY]["status"] == expected
    assert evidence[tone.KEY]["comparisons"] == row["comparisons"]["harmonic"]
    events = [item for item in evidence["events"] if item["category"] == tone.KEY]
    words = " ".join((folder / "report-text.txt").read_text().split())
    event_summary = None
    if expected == "measured_change":
        event, = events
        match = evidence[tone.KEY]["measurement"]
        assert match == row["ranges"]["harmonic"] and tone.TITLE in words
        assert "zero AI-confidence weight" in words
        start, end = match["tones"][0]["samples"]
        assert [event["interval"][key] for key in ("start_sample", "end_sample")] == [start, end]
        ranges = event["source_timestamp_ranges"]
        assert [(value["source_start_pts"], value["source_end_pts"]) for value in ranges] == [(start, end)]
        clip = event["artifacts"]["selected_clip"]
        with wave.open(io.BytesIO(fetch(client, clip["url"]))) as excerpt:
            assert excerpt.readframes(excerpt.getnframes()) == raw[2*start:2*end]
        for band in match["tones"]:
            assert f"{band['frequency_hz']} Hz phase offset near the start" in words
        event_summary = {"source_interval_samples": [start, end], "tones": match["tones"], "clip_sha256": clip["sha256"]}
    else:
        assert events == [] and tone.TITLE not in words
        status_copy = {"unassessed": "lacked sufficiently clear paired tones",
                       "no_supported_change": "No phase change met the measurement rules"}
        assert status_copy[expected] in words
    return {"status": expected, "truth_samples": row["case"]["truth"], "scene": row["case"]["scene"],
            "source_pcm_sha256": row["sha256"], "event": event_summary, "signed_source_replay": True,
            "evidence_assets_verified": checks["assets"], "exports_match": checks["api_json_snapshot_equal"]}


def run(args):
    receipt = read(args.receipt)
    transfer.review(receipt, args.parent)
    inputs = recordings(receipt, args.shards, args.hums, args.sounds)
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_DETECTOR_MODE="candidate",
                      VERITAS_AUDIO_TONAL_PHASE_REVIEW="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store

    def decision(aid):
        with get_session() as db:
            stored = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return stored["decision"], stored.get("customer_assessment")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-tone-phase-review", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
        for state, (row, raw) in inputs.items():
            source = args.output / f"{state}.wav"
            write_wav(source, raw)
            settings.audio_tonal_phase_review = False
            baseline = checkout(client, stub, source, kind="audio", media_type="audio/wav", filename=source.name)
            original = body(client.get(f"/api/v2/analyses/{baseline}/evidence-packets"))["acoustic_context"]
            assert tone.KEY not in original
            settings.audio_tonal_phase_review = True
            aid = checkout(client, stub, source, kind="audio", media_type="audio/wav", filename=source.name)
            assert decision(aid) == decision(baseline)
            results[state] = verify_outputs(client, aid, source, args.output / state, key, state, row, raw)
    write_json(args.output / "receipt.json", {"cases": results, "simulated_paid_analyses": len(stub.sessions),
                                             "unpaid_jobs": 0, "overall_assessments_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "receipt", "parent", "shards", "hums", "sounds"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
