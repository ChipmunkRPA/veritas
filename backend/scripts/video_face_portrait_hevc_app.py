# SPDX-License-Identifier: Apache-2.0
"""Check a held HEVC portrait edit and its matched controls through the local app."""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_common_sources_v89 import sha, write
from scripts.video_face_portrait_hevc_transfer import SCHEMA, choose, located
from scripts.video_stereo_covariance_panel import read
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def run(held, media, output):
    document = read(held)
    if document["schema"] != SCHEMA or not document["summary"]["accepted_offline"]:
        raise ValueError("The held HEVC transport must qualify before the worker replay")
    point = float.fromhex(document["source"]["point"])
    by_name = {case["hevc_file"]: case for case in document["rows"]}
    pairs = [(label, by_name[f"10-{kind}-hevc.mp4"]) for label, kind in
             (("known-edit", "inswapper"), ("unchanged", "original"), ("ordinary-cut", "different_shot"))]
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
    from app.evidence.face_portrait_return_binding import HEVC_ENCODING, HEVC_METHOD
    from app.evidence.video_packet_versions import PORTRAIT_HEVC_FACE_RETURN_VERSION
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    current = registry.enabled
    registry.enabled = lambda modality, profile: [item for item in current(modality, profile) if item in paid.ALLOW]
    payment = paid.PaymentStub()
    service._stripe = payment
    results, browser = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("portrait-hevc-local-review", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for label, row in pairs:
            source = media / row["hevc_file"]
            if sha(source) != row["hevc_sha256"]:
                raise ValueError("The stage must receive the exact tested HEVC file")
            expected = choose(row["hevc"], point)
            if bool(expected) != located(row["hevc"]["truth"], expected):
                raise ValueError("The selected replay must either be a useful edit location or have no proposal")
            identifier = paid.checkout(client, payment, source, filename=f"{label}.mp4")
            _, detector = paid.read_result(identifier, row["hevc_sha256"], require_signal=False)
            route = f"/api/v2/analyses/{identifier}"
            evidence = paid.body(client.get(f"{route}/evidence-packets"))
            packets = [packet for packet in evidence["packets"] if packet.get("semantics") == HEVC_METHOD]
            assert len(packets) == int(expected is not None), detector["details"].get("face_return_check")
            checked = {"case": label, "file": row["hevc_file"], "input_sha256": row["hevc_sha256"],
                       "labeled_frames": row["hevc"]["truth"], "selected_frames": expected}
            if packets:
                [packet] = packets
                report = packet["face_return"]
                assert evidence["version"] == PORTRAIT_HEVC_FACE_RETURN_VERSION
                assert tuple(report["source_encoding"]) == HEVC_ENCODING
                assert [report["start_frame_index"], report["end_frame_index"]] == expected
                for picture in packet["frames"]:
                    anchor = picture["anchor"]
                    expected_anchor = row["hevc"]["proposal"]["anchors"][picture["interval_role"]]
                    assert [anchor[k] for k in ("frame_index", "pts", "decoded_rgb_sha256")] == expected_anchor
                exported = paid.fetch(client, f"/api/analyses/{identifier}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                for blob in (exported, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
                    scan(blob.decode())
                folder = output / label
                pictures = paid.proof_files(client, packet, folder)
                checked.update(paid.encode_report(client, identifier, source, evidence, pictures, folder,
                                                  expected_text="no abrupt shot change"))
                checked["anchors"] = row["hevc"]["proposal"]["anchors"]
                checked["time_base"] = report["time_base"]
                clip = packet["clip"]
                assert clip["status"] == "ready"
                focus = packet["frames"][1]["anchor"]
                index = focus["frame_index"] - clip["start_frame_index"]
                assert clip["source_pts"][index] == focus["pts"]
                playback = float(clip["clip_pts"][index] * Fraction(clip["clip_time_base"]))
                browser.append({"id": identifier, "name": label, "packet": packet["id"],
                                "target": focus, "frames": packet["window"]["frame_indices"], "context_seek_s": playback})
            settings.video_face_portrait_return_review = False
            rollback = paid.checkout(client, payment, source, filename=f"{label}.mp4")
            earlier = paid.body(client.get(f"/api/v2/analyses/{rollback}/evidence-packets"))
            assert all(packet.get("semantics") != HEVC_METHOD for packet in earlier["packets"])
            assert tokens_assessed(identifier) == tokens_assessed(rollback)
            settings.video_face_portrait_return_review = True
            results.append(checked | {"confidence_identical": True})
            print(f"Verified paid local HEVC replay and rollback for {label}", flush=True)
        browser_receipt = browser_check(output, app, key, browser, scenario="video-face-shot-guard.mjs")
    write(output / "receipt.json", {
        "schema": "chipmunk.portrait-hevc-checkout/1", "held_sha256": sha(held), "cases": results,
        "browser": browser_receipt, "simulated_paid_jobs": len(payment.sessions), "unpaid_jobs": 0,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("held", "media", "output"):
        cli.add_argument(field, type=Path)
    run(**vars(cli.parse_args()))
