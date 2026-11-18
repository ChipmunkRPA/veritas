# SPDX-License-Identifier: Apache-2.0
"""Replay a held square edit and its controls through local checkout and reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_common_sources_v89 import sha, write
from scripts.video_face_portrait_hevc_transfer import choose, located
from scripts.video_face_square_transfer import POINT, SCHEMA
from scripts.video_stereo_covariance_panel import read
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def run(receipt, media, output):
    comparison = read(receipt)
    if comparison["schema"] != SCHEMA or not comparison["summary"]["accepted_offline"]:
        raise ValueError("The frozen square video comparison must qualify before the app replay")
    point = float.fromhex(POINT)
    for group in comparison["cohorts"]["fresh"]:
        items = {row["kind"]: row for row in group["rows"]}
        candidates = [row for row in group["rows"] if row["truth"] and row["truth"] == choose(row, point)]
        if candidates and all(choose(items[kind], point) is None for kind in ("original", "different_shot")):
            inputs = [("known-edit", candidates[0]), ("unchanged", items["original"]), ("ordinary-cut", items["different_shot"])]
            break
    else:
        raise ValueError("No fresh source supplies an exact edit and both ordinary controls")
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
    from app.evidence.face_portrait_return_binding import SQUARE_ENCODING, SQUARE_METHOD
    from app.evidence.video_packet_versions import SQUARE_FACE_RETURN_VERSION
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    current = registry.enabled
    registry.enabled = lambda modality, profile: [item for item in current(modality, profile) if item in paid.ALLOW]
    payment = paid.PaymentStub()
    service._stripe = payment
    cases, browser = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("square-face-local-review", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for label, row in inputs:
            source = media / "fresh" / row["file"]
            if sha(source) != row["input_sha256"]:
                raise ValueError("The local worker must use the exact evaluated square video")
            expected = choose(row, point)
            assert bool(expected) == located(row["truth"], expected)
            aid = paid.checkout(client, payment, source, filename=f"{label}.mp4")
            _, detector = paid.read_result(aid, row["input_sha256"], require_signal=False)
            route = f"/api/v2/analyses/{aid}"
            evidence = paid.body(client.get(f"{route}/evidence-packets"))
            packets = [packet for packet in evidence["packets"] if packet.get("semantics") == SQUARE_METHOD]
            assert len(packets) == int(expected is not None), detector["details"].get("face_return_check")
            checked = {"case": label, "file": row["file"], "input_sha256": row["input_sha256"],
                       "labeled_frames": row["truth"], "selected_frames": expected}
            if packets:
                packet, = packets
                record = packet["face_return"]
                assert evidence["version"] == SQUARE_FACE_RETURN_VERSION
                assert tuple(record["source_encoding"]) == SQUARE_ENCODING
                assert [record["start_frame_index"], record["end_frame_index"]] == expected
                for witness in packet["frames"]:
                    anchor = witness["anchor"]
                    exact = [anchor[name] for name in ("frame_index", "pts", "decoded_rgb_sha256")]
                    assert exact == row["proposal"]["anchors"][witness["interval_role"]]
                exported = paid.fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                for blob in (exported, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
                    scan(blob.decode())
                folder = output / label
                pictures = paid.proof_files(client, packet, folder)
                checked.update(paid.encode_report(client, aid, source, evidence, pictures, folder,
                                                  expected_text="no abrupt shot change"))
                checked.update(anchors=row["proposal"]["anchors"], time_base=record["time_base"])
                browser.append({"id": aid, "name": label, "packet": packet["id"], "target": packet["frames"][1]["anchor"],
                                "frames": packet["window"]["frame_indices"]})
            settings.video_face_portrait_return_review = False
            rollback = paid.checkout(client, payment, source, filename=f"{label}.mp4")
            baseline = paid.body(client.get(f"/api/v2/analyses/{rollback}/evidence-packets"))
            assert all(packet.get("semantics") != SQUARE_METHOD for packet in baseline["packets"])
            assert tokens_assessed(aid) == tokens_assessed(rollback)
            settings.video_face_portrait_return_review = True
            cases.append(checked | {"confidence_identical": True})
            print(f"Verified paid square video and rollback for {label}", flush=True)
        shown = browser_check(output, app, key, browser, scenario="video-face-shot-guard.mjs")
    write(output / "receipt.json", {"schema": "chipmunk.square-face-checkout/1", "held_sha256": sha(receipt),
                                      "cases": cases, "browser": shown, "simulated_paid_jobs": len(payment.sessions),
                                      "unpaid_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("receipt", "media", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
