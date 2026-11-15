# SPDX-License-Identifier: MIT
"""Replay paired stereo cancellation and level controls through prepaid audio analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path

from scripts.audio_short_edit_room_training import SOURCE, base, plan_hash, validate
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
TYPES = ("mono", "equal_channels", "single_channel", "cancellation")
PANEL = ROOT / "docs/evaluation-panels/audio-short-edit-room-training-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-downmix-coverage/1"


def fixtures(inputs, output):
    import numpy as np

    plan_path = inputs / "freeze.json"
    plan = SOURCE.read(plan_path)
    validate(plan)
    groups = [group for group in plan["groups"] if group["role"] == "held" and group["panel"] == "phoneme"]
    selected = []
    for group in sorted(groups, key=lambda row: row["id"]):
        if group["speaker"] not in {row["speaker"] for row in selected}:
            selected.append(group)
    selected.append(min((row for row in groups if row not in selected), key=lambda row: row["id"]))
    assert len(selected) == 4
    for index, group in enumerate(selected):
        unedited = group["observations"]["dry"]["untouched"]
        carrier = base.pcm(inputs, {"pcm_sha256": unedited})[:3 * RATE]
        replacement = selected[(index + 1) % len(selected)]
        family = base.sources.FAMILIES[index % 2]
        edited = replacement["observations"]["dry"][f"edited/{family}"]
        start, stop = replacement["span"]
        side_start = min(5 * RATE, max(0, (start + stop - RATE) // 2))
        side = base.pcm(inputs, {"pcm_sha256": edited})[side_start:side_start + RATE]
        side = np.rint(side * (14500 / np.max(np.abs(side)))).astype(np.int32)
        # Keep the exact same PCM in the mono and averaged dual-channel conditions.
        mid = np.rint(carrier * (1800 / np.max(np.abs(carrier)))).astype(np.int32)
        left, right = mid.copy(), mid.copy()
        left[RATE:2 * RATE] += side
        right[RATE:2 * RATE] -= side
        for case, values in zip(TYPES, (mid, np.column_stack((mid, mid)),
                                       np.column_stack((mid * 2, np.zeros_like(mid))),
                                       np.column_stack((left, right))), strict=True):
            path = output / f"{index + 1}-{case}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1 if case == "mono" else 2, 2, RATE, 0, "NONE", ""))
                wav.writeframes(values.astype("<i2").tobytes())
            yield case, path, {"group": group["id"], "carrier_pcm_sha256": unedited,
                               "replacement_pcm_sha256": edited, "replacement_sample_start": side_start}


def run(inputs, output, *, before):
    configure(output, ROOT)
    os.environ.update(VERITAS_AUDIO_DETECTOR_MODE="candidate", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.engine.stages import audio
    from app.evidence.audio_binding import MONO_VOICE_COVERAGE, MONO_VOICE_LIMIT
    from app.main import app
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if before:
        audio.destructive_downmix_windows = lambda _context: []
    stub = PaymentStub()
    service._stripe = stub
    rows = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-stereo-coverage-check", "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        for case, path, join in fixtures(inputs, output):
            aid = checkout(client, stub, path, kind="audio", media_type="audio/wav", filename=path.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            scan(exported.decode())
            assert json.loads(exported)["evidence"] == evidence
            with get_session() as db:
                row = db.get(Analysis, aid)
                result = json.loads(default_store().read_bytes(row.result_hash))
            decision = result["decision"]
            context = evidence["stereo_context"]
            selected = [window for window in context["windows"] if window["downmix"]["strong_attenuation"]]
            assert len(selected) == int(case == "cancellation"), (path.name, context)
            assert case != "cancellation" or selected[0]["start_sample"] == RATE
            limit = evidence["coverage"].get(MONO_VOICE_COVERAGE)
            guarded = case == "cancellation" and not before
            assert bool(limit) == guarded
            if guarded:
                assert limit["stereo_window_indices"] == [1] and decision["reason"] == MONO_VOICE_LIMIT
                assert decision["code"] == "inconclusive" and decision["probability_ai_calibrated"] is None
                assert not decision["contributions"]
                # The sound outside the mono input remains available as original-channel evidence.
                (listening,) = evidence["stereo_listening"]["windows"]
                assert listening["stereo_window_index"] == 1
                for asset in listening["artifacts"].values():
                    assert hashlib.sha256(fetch(client, asset["url"])).hexdigest() == asset["sha256"]
            report_checked = False
            if case == "cancellation" and not before and not any(r["report_checked"] for r in rows):
                report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                words = " ".join(pdf_text(pdf).split())
                scan(words)
                assert MONO_VOICE_LIMIT in words and "/coverage/mono_voice_coverage_limit" in words
                (output / "report.pdf").write_bytes(pdf)
                (output / "export.json").write_bytes(exported)
                package = body(client.post(f"{route}/packages", json={"report_id": report["id"], "kind": "draft"}))
                package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
                path_zip = output / "signed-package.zip"
                path_zip.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                signer = body(client.get("/api/v2/instance/public-key"))
                result = verify_package(path_zip, public_key_pem=signer["public_key_pem"].encode(),
                                        expect_key_id=signer["key_id"], source_audio=path)
                assert result.outcome == "verified" and result.signer_authenticated, result.as_dict()
                report_checked = True
            observation = {**join, "case": case, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "decoded_pcm_sha256": evidence["source"]["pcm_sha256"],
                           "decision": {key: decision[key] for key in ("code", "probability_ai_calibrated", "reason")},
                           "attenuated_windows": selected, "report_checked": report_checked}
            rows.append(observation)
            print(path.name, decision["code"], decision["probability_ai_calibrated"], flush=True)
    assert len(rows) == 16 and len(stub.sessions) == 16
    SOURCE.write(output / "receipt.json", {"schema": SCHEMA, "price_usd_cents": 999, "jobs_before_payment": 0, "before": before,
                                           "plan_sha256": SOURCE.path_sha(inputs / "freeze.json"), "rows": rows})


def review(comparison):
    import gzip

    from app.evidence.audio_binding import MONO_VOICE_LIMIT

    SOURCE.check(comparison["schema"] == SCHEMA, "Unsupported downmix coverage evaluation")
    with gzip.open(PANEL, "rt", encoding="utf-8") as source:
        plan = json.load(source)["plan"]
    validate(plan)
    archived = {row["id"]: set(row["observations"]["dry"].values()) for row in plan["groups"]
                if row["panel"] == "phoneme" and row["role"] == "held"}
    sessions = [comparison["before"], comparison["after"]]
    for before, session in zip((True, False), sessions, strict=True):
        SOURCE.check(session["jobs_before_payment"] == 0 and session["price_usd_cents"] == 999
                     and session["before"] is before and session["plan_sha256"] == plan_hash(plan),
                     "Evaluation does not match the sealed source and payment scope")
    cases = [{(row["group"], row["case"]): row for row in session["rows"]} for session in sessions]
    ordinary = cancelled = 0
    for identity in sorted(cases[0]):
        prior, current = (case[identity] for case in cases)
        for key in ("carrier_pcm_sha256", "replacement_pcm_sha256", "replacement_sample_start", "file_sha256",
                    "decoded_pcm_sha256", "attenuated_windows"):
            SOURCE.check(prior[key] == current[key], "Matched analysis source or window differs")
        group, station = identity
        SOURCE.check(current["carrier_pcm_sha256"] in archived[group]
                     and any(current["replacement_pcm_sha256"] in source for source in archived.values()),
                     "Fixture differs from the released source commitments")
        controls = [cases[0][group, kind] for kind in TYPES]
        SOURCE.check(len({row["decoded_pcm_sha256"] for row in controls}) == 1,
                     "Control and cancellation produced different mono input")
        if station == "cancellation":
            window, = current["attenuated_windows"]
            SOURCE.check(window["start_sample"] == RATE and window["end_sample"] == 2 * RATE
                         and window["source_timestamp_ranges"] == [{"start_sample": RATE, "end_sample": 2 * RATE,
                            "source_start_pts": RATE, "source_end_pts": 2 * RATE, "time_base": "1/16000"}]
                         and window["correlation"] <= -.85 and window["downmix"]["strong_attenuation"]
                         and prior["decision"]["code"] == "no_ai_indicators"
                         and current["decision"] == {"code": "inconclusive", "probability_ai_calibrated": None,
                                                     "reason": MONO_VOICE_LIMIT}, "The cancelled input lacks its scope decision")
            cancelled += 1
        else:
            SOURCE.check(current["decision"] == prior["decision"] and not current["attenuated_windows"], "A control changed")
            ordinary += 1
    SOURCE.check(len(cases[0]) == len(cases[1]) == 16 and all(len(session["rows"]) == 16 for session in sessions)
                 and cancelled == 4 and ordinary == 12 and sum(row["report_checked"] for row in sessions[1]["rows"]) == 1,
                 "Incomplete cancellation, control or signed report evidence")
    print(f"Verified {cancelled} corrected coverage decisions and {ordinary} identical control decisions across 32 paid simulations.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", type=Path)
    cli.add_argument("output", type=Path, nargs="?")
    cli.add_argument("--before", action="store_true")
    cli.add_argument("--review", action="store_true")
    args = cli.parse_args()
    if args.review:
        review(SOURCE.read(args.inputs))
    else:
        if args.output is None:
            cli.error("An output directory is required for an app replay")
        run(args.inputs, args.output, before=args.before)
