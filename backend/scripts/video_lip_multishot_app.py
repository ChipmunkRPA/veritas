# SPDX-License-Identifier: Apache-2.0
"""Replay a late soundtrack dropout through payment, browser and signed exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_lip_multishot_panel import CARRIERS, digest, overlap, read
from scripts.video_short_return_encoded_app import PaymentStub, checkout, configure
from scripts.video_silent_mouth_app import nested, proof_files

ROOT = Path(__file__).resolve().parents[2]


def examples(panel):
    rows = read(panel / "result.json.gz")["rows"]
    for carrier in CARRIERS:
        by_case = {(row["group"], row["role"]): row for row in rows if row["carrier"] == carrier}
        for (group, role), row in sorted(by_case.items()):
            if (role in {"muted", "muted_faded"} and len(row["matches"]) == 1
                    and overlap(row["matches"][0], row["truth"]) >= .4 and not by_case[group, "original"]["matches"]):
                control = by_case[group, "original"]
                yield row, control
                break


def run(panel, assets, output, *, browser):
    configure(output, ROOT)
    timing_assets = assets
    os.environ.update(
        VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="auto", VERITAS_LOG_LEVEL="ERROR", VERITAS_JOB_TIMEOUT_S="360",
        VERITAS_VIDEO_LIP_TIMING_REVIEW="true",
        VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str((timing_assets / "face_landmarker.task").resolve()),
        VERITAS_VIDEO_LIP_TIMING_RUNTIME=str((timing_assets / "runtime").resolve()),
        VERITAS_VIDEO_LIP_TIMING_MODEL_DIR=str((timing_assets / "checkpoint").resolve()),
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

    def decision(aid):
        with get_session() as db:
            result = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return result["decision"], result.get("customer_assessment")

    selected = list(examples(panel))
    if not selected:
        raise ValueError("The confirmation panel has no correct late near-silent interval for app replay")
    stub = PaymentStub()
    service._stripe = stub
    receipts = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-shot-motion-smoke", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for index, (row, control) in enumerate(selected):
            carrier = row["carrier"]
            outcomes = {}
            runs = [("edit", row, True), ("normal", control, True)]
            if index == 0:
                runs.insert(0, ("off", row, False))
            for name, item, enabled in runs:
                settings.video_lip_silent_mouth_review = enabled
                media = panel / item["delivery"]
                if digest(media) != item["input_sha256"]:
                    raise ValueError("A local replay input differs from the measured upload")
                aid = checkout(client, stub, media, media_type="video/quicktime", filename=media.name)
                evidence, exported = nested(client, aid)
                scan(json.dumps(exported))
                review = evidence[binding.KEY]
                result = {"carrier": carrier, "case": name, "status": review["status"], "input_sha256": digest(media)}
                if enabled:
                    assert review["schema"] == binding.SCHEMA_WITH_SHOTS and review["shot_coverage"] == item["coverage"]
                    assert review["affects_overall_confidence"] is False
                    outcomes[name] = decision(aid)
                else:
                    assert review["status"] == "unsupported" and not list(binding.exhibits(review))
                    outcomes[name] = decision(aid)
                if name == "edit":
                    evidence_dir = output / carrier
                    evidence_dir.mkdir()
                    [(_, measured)] = list(binding.exhibits(review))
                    assert [measured["witness"]["audio_source_pts"]] == item["matches"]
                    result.update(proof_files(client, aid, media, evidence_dir, evidence, {
                        "measurements": measured["measurement"], "frames": measured["witness"]["picture_frames"],
                    }, shots=True))
                    if browser:
                        result["browser"] = browser_check(evidence_dir, app, key, [{
                            "analysis_id": aid, "frames": result["frames"], "shot_count": len(review["shot_coverage"]),
                        }], scenario="video-silent-mouth-local.mjs")
                    (evidence_dir / "near-silent.json").write_text(json.dumps(exported, allow_nan=False))
                elif name == "normal":
                    assert review["status"] == "no_match" and not list(binding.exhibits(review))
                assert "off" not in outcomes or "edit" not in outcomes or outcomes["off"] == outcomes["edit"]
                receipts.append(result)
                print(carrier, name, review["status"], flush=True)
    assert len(receipts) == len(stub.sessions)
    (output / "receipt.json").write_text(json.dumps({
        "cases": receipts, "price_usd": 9.99, "paid_analyses": len(receipts), "unpaid_jobs": 0,
        "assessment_same_with_option_off": True,
    }, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "assets", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--browser", action="store_true")
    run(**vars(cli.parse_args()))
