# SPDX-License-Identifier: Apache-2.0
"""Verify the held second-half portrait edit through local app evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_common_sources_v89 import sha, write
from scripts.video_face_portrait_eight_seconds import POINT, SCHEMA
from scripts.video_face_portrait_hevc_transfer import choose, located
from scripts.video_stereo_covariance_panel import read
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def run(receipt, media, output):
    comparison = read(receipt)
    if comparison["schema"] != SCHEMA or not comparison["summary"]["approved_offline"]:
        raise ValueError("The held native-portrait test must pass before the local app replay")
    point = float.fromhex(POINT)
    available = [row for row in comparison["rows"] if row["truth"] and located(row["truth"], choose(row, point))]
    ordinary = {(row["group"], row["kind"]): row for row in comparison["rows"] if not row["truth"]}
    positives = sorted(available, key=lambda row: (row["truth"][0] < 120, choose(row, point) != row["truth"], row["file"]))
    paid.configure(output, ROOT)
    models = local_model_view(output)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_PORTRAIT_RETURN_REVIEW="true",
                      VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"), VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_MODELS_DIR=str(models), VERITAS_HF_HOME=str(models / "hf"))

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.face_portrait_return_binding import EIGHT_SECOND_ENCODING, EIGHT_SECOND_METHOD
    from app.evidence.video_packet_versions import PORTRAIT_EIGHT_SECOND_RETURN_VERSION
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    active = registry.enabled
    registry.enabled = lambda kind, profile: [check for check in active(kind, profile) if check in paid.ALLOW]
    payment = paid.PaymentStub()
    service._stripe = payment
    browser, result, attempted = [], [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("eight-second-portrait-local", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for candidate in positives:
            source = media / candidate["file"]
            if sha(source) != candidate["input_sha256"]:
                raise ValueError("Local app input differs from the exact held upload")
            positive_aid = paid.checkout(client, payment, source, filename="known-edit.mp4")
            _, check = paid.read_result(positive_aid, candidate["input_sha256"], require_signal=False)
            details = check["details"]
            route = f"/api/v2/analyses/{positive_aid}/evidence-packets"
            evidence = paid.body(client.get(route))
            fresh = any(entry.get("semantics") == EIGHT_SECOND_METHOD for entry in evidence["packets"])
            existing = [[details[name]["start_pts"], details[name]["end_pts"]] for name in ("face_sequence", "face_boundary") if name in details]
            attempted.append({"file": candidate["file"], "input_sha256": candidate["input_sha256"], "existing_range_ticks": existing,
                              "retained_new_range": fresh})
            if fresh:
                break
            print(f"The integrated worker withheld the standalone range for {candidate['file']}", flush=True)
        else:
            write(output / "withheld.json", attempted)
            raise ValueError("No held candidate adds a distinct face range with the legacy checks active")
        paired = [ordinary[candidate["group"], kind] for kind in ("original", "different_shot")]
        assert all(choose(row, point) is None for row in paired)
        samples = [("known-edit", candidate, positive_aid), ("unchanged", paired[0], None), ("ordinary-cut", paired[1], None)]
        for name, row, known_aid in samples:
            source = media / row["file"]
            if sha(source) != row["input_sha256"]:
                raise ValueError("Local app input differs from the exact held upload")
            expected = choose(row, point)
            aid = known_aid or paid.checkout(client, payment, source, filename=f"{name}.mp4")
            _, check = paid.read_result(aid, row["input_sha256"], require_signal=False)
            route = f"/api/v2/analyses/{aid}"
            evidence = paid.body(client.get(f"{route}/evidence-packets"))
            packets = [item for item in evidence["packets"] if item.get("semantics") == EIGHT_SECOND_METHOD]
            assert len(packets) == int(expected is not None), check["details"].get("face_return_check")
            row_proof = {"name": name, "file": row["file"], "input_sha256": row["input_sha256"],
                         "known_range_frames": row["truth"], "selected_frames": expected}
            if packets:
                packet, = packets
                record = packet["face_return"]
                assert evidence["version"] == PORTRAIT_EIGHT_SECOND_RETURN_VERSION
                assert tuple(record["source_encoding"]) == EIGHT_SECOND_ENCODING
                assert [record["start_frame_index"], record["end_frame_index"]] == expected
                for item in packet["frames"]:
                    anchor = item["anchor"]
                    assert [anchor[key] for key in ("frame_index", "pts", "decoded_rgb_sha256")] == row["proposal"]["anchors"][item["interval_role"]]
                json_export = paid.fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(json_export)["evidence"] == evidence
                for blob in (json_export, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
                    scan(blob.decode())
                folder = output / name
                pictures = paid.proof_files(client, packet, folder)
                row_proof.update(paid.encode_report(client, aid, source, evidence, pictures, folder,
                                                     expected_text="no abrupt shot change"))
                row_proof.update(anchors=row["proposal"]["anchors"], time_base=record["time_base"])
                browser.append({"id": aid, "name": name, "packet": packet["id"], "target": packet["frames"][1]["anchor"],
                                "frames": packet["window"]["frame_indices"]})
            settings.video_face_portrait_return_review = False
            rollback = paid.checkout(client, payment, source, filename=f"{name}.mp4")
            older = paid.body(client.get(f"/api/v2/analyses/{rollback}/evidence-packets"))
            assert all(item.get("semantics") != EIGHT_SECOND_METHOD for item in older["packets"])
            assert tokens_assessed(aid) == tokens_assessed(rollback)
            settings.video_face_portrait_return_review = True
            result.append(row_proof | {"confidence_identical": True})
            print(f"Exact eight-second replay passed for {name}", flush=True)
        ui = browser_check(output, app, key, browser, scenario="video-face-shot-guard.mjs")
    write(output / "receipt.json", {"schema": "chipmunk.portrait-eight-second-app/1", "held_sha256": sha(receipt),
                                    "cases": result, "attempted_edits": attempted, "browser": ui,
                                    "simulated_paid_jobs": len(payment.sessions), "unpaid_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ("receipt", "media", "output"):
        parser.add_argument(attr, type=Path)
    run(**vars(parser.parse_args()))
