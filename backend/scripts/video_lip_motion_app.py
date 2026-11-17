# SPDX-License-Identifier: Apache-2.0
"""Verify the motion and soundtrack comparison through prepaid analysis and exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.bench import video_av_local_agreement as bench
from app.evidence import video_lip_agreement as agreement
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_lip_articulation_app import verify
from scripts.video_short_return_encoded_app import PaymentStub, checkout, configure
from scripts.video_silent_mouth_app import nested

ROOT = Path(__file__).resolve().parents[2]


def fixtures(panel, native):
    outcomes = bench.read(panel)
    verified = bench.read(native / "review.json.gz")
    if bench.digest(panel) != verified["held_sha256"]:
        raise ValueError("Native cases differ from the measured confirmation")
    clips = {(row["source_sha256"], row["carrier"], row["role"]): row for row in verified["deliveries"]}
    for row in outcomes["rows"]:
        selected = []
        for carrier, points in agreement.POINTS.items():
            scenes = row["cases"][carrier]
            edit, control = (scenes[role] for role in ("different_words", "original"))
            candidate, normal = (case["proposals"]["shared"]["candidate"] for case in (edit, control))
            if (
                edit["baseline"]
                or control["baseline"]
                or not candidate
                or round(candidate["score"] * agreement.UNITS) <= points[0]
                or not bench.useful(candidate["span"], row["truth"])
                or (normal and round(normal["score"] * agreement.UNITS) > points[0])
            ):
                break
            selected.append((carrier, candidate))
        if len(selected) == len(bench.CARRIERS):
            for carrier, candidate in selected:
                for role in ("different_words", "original"):
                    delivery = clips[row["source_sha256"], carrier, role]
                    path = native / "clips" / f"{delivery['id']}.mov"
                    if bench.digest(path) != delivery["input_sha256"]:
                        raise ValueError("The prepaid fixture differs from its native video receipt")
                    yield carrier, role, path, candidate if role == "different_words" else None
            return
    raise ValueError("No edited source qualified in both encodings with a clean corresponding original")


def run(panel, native, assets, models, output, *, browser):
    configure(output, ROOT)
    os.environ.update(
        VERITAS_ENVIRONMENT="dev",
        VERITAS_DEVICE="auto",
        VERITAS_LOG_LEVEL="ERROR",
        VERITAS_JOB_TIMEOUT_S="360",
        VERITAS_VIDEO_LIP_TIMING_REVIEW="true",
        VERITAS_VIDEO_LIP_TIMING_RANKED="true",
        VERITAS_VIDEO_LIP_ARTICULATION_REVIEW="true",
        VERITAS_VIDEO_LIP_AAC48_ARTICULATION_REVIEW="true",
        VERITAS_VIDEO_LIP_SILENT_MOUTH_REVIEW="true",
        VERITAS_VIDEO_LIP_TIMING_MODEL_DIR=str((assets / "checkpoint").resolve()),
        VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str((assets / "face_landmarker.task").resolve()),
        VERITAS_VIDEO_LIP_TIMING_RUNTIME=str((assets / "runtime").resolve()),
    )

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import video_viseme_binding as binding
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(analysis):
        with get_session() as db:
            value = json.loads(default_store().read_bytes(db.get(Analysis, analysis).result_hash))
        return value["decision"], value.get("customer_assessment")

    stub = PaymentStub()
    service._stripe = stub
    results, examples = [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-motion-sound-smoke", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for carrier, role, path, candidate in fixtures(panel, native):
            initial = None
            for enabled in (False, True) if carrier == "pcm" and candidate else (True,):
                settings.video_lip_motion_sound_models = models if enabled else None
                analysis = checkout(client, stub, path, media_type="video/quicktime", filename=path.name)
                evidence, exported = nested(client, analysis)
                scan(json.dumps(exported))
                review = evidence[binding.KEY]
                case = {
                    "carrier": carrier,
                    "role": role,
                    "enabled": enabled,
                    "status": review["status"],
                    "input_sha256": bench.digest(path),
                }
                print(carrier, role, enabled, review["status"], review.get("measurement_basis"), flush=True)
                if not enabled:
                    assert review["witness"] is None
                    initial = decision(analysis)
                elif candidate:
                    if initial is not None:
                        assert decision(analysis) == initial
                    proof = output / carrier
                    proof.mkdir()
                    case.update(
                        verify(
                            client,
                            analysis,
                            path,
                            proof,
                            evidence,
                            {"span": candidate["span"], "frames": candidate["bracket_frames"]},
                            motion=True,
                        )
                    )
                    (proof / "mismatch.json").write_text(json.dumps(exported, allow_nan=False))
                    examples[carrier] = {
                        "analysis_id": analysis,
                        "title": agreement.TITLE,
                        "witness": case["witness"],
                        "sound_sha256": case["sound_sha256"],
                    }
                else:
                    assert review["witness"] is None
                results.append(case)
        tested = (
            {
                carrier: browser_check(output / carrier, app, key, [example], scenario="lip-articulation-replay.mjs")
                for carrier, example in examples.items()
            }
            if browser
            else None
        )
    assert len(stub.sessions) == len(results) == 5
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "cases": results,
                "browser": tested,
                "paid_analyses": len(results),
                "unpaid_jobs": 0,
                "price_usd": 9.99,
                "assessment_same_with_option_off": True,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "native", "assets", "models", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--browser", action="store_true")
    run(**vars(cli.parse_args()))
