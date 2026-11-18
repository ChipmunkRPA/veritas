# SPDX-License-Identifier: Apache-2.0
"""Verify portrait video findings through local checkout, browser and signed reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.review_video_face_portrait import read, selected, verify
from scripts.video_common_sources_v89 import sha, write
from scripts.video_face_local_fit import overlap
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def shortlist(rows, point):
    for delivery in ("direct", "tandem"):
        workable = [row for row in rows if row["delivery"] == delivery and overlap(row["truth"], selected(row, point)) >= .5]
        yield delivery, sorted(workable, key=lambda row: (-overlap(row["truth"], selected(row, point)), row["file"]))


def run(receipt, freeze, media, output, *, heights, all_checks=False):
    comparison = read(receipt)
    verify(comparison, freeze)
    paid.configure(output, ROOT)
    if all_checks:
        os.environ.pop("VERITAS_VIDEO_WHOLE_FACE_REVIEW")
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
    from app.evidence.face_portrait_return_binding import METHOD, POINTS
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    if not all_checks:
        current = registry.enabled
        registry.enabled = lambda modality, profile: [item for item in current(modality, profile) if item in paid.ALLOW]
    payment = paid.PaymentStub()
    service._stripe = payment
    receipts, browser = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("portrait-face-local-review", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for height in heights:
            panel = comparison["banks"][str(height)]
            point = comparison["points"][str(height)][comparison["primary"][str(height)]]
            if POINTS[tuple(panel["header"]["size"])] != point:
                raise ValueError("Serving point differs from the frozen portrait test")
            for delivery, candidates in shortlist(panel["rows"], point):
                report_case = None
                for row in candidates[:4]:
                    source = media / f"held-{height}" / row["file"]
                    if sha(source) != row["input_sha256"]:
                        raise ValueError("Local app did not receive the exact evaluated video")
                    identifier = paid.checkout(client, payment, source, filename=source.name)
                    _, entry = paid.read_result(identifier, row["input_sha256"], require_signal=False)
                    details = entry["details"]
                    hit = details.get("face_return") or {}
                    expected = selected(row, point)
                    if hit.get("method") != METHOD:
                        print("Available app coverage", height, row["file"], details.get("face_return_check"), flush=True)
                        continue
                    if [hit["start_frame_index"], hit["end_frame_index"]] != expected:
                        raise ValueError("The worker moved the frozen native portrait location")
                    route = f"/api/v2/analyses/{identifier}"
                    evidence = paid.body(client.get(f"{route}/evidence-packets"))
                    [packet] = [entry for entry in evidence["packets"] if entry.get("semantics") == METHOD]
                    exported = paid.fetch(client, f"/api/analyses/{identifier}/export.json")
                    assert json.loads(exported)["evidence"] == evidence and packet["face_return"] == hit
                    for blob in (exported, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
                        scan(blob.decode())
                    for witness in packet["frames"]:
                        anchor = witness["anchor"]
                        expected_anchor = row["proposal"]["anchors"][witness["interval_role"]]
                        assert [anchor["frame_index"], anchor["pts"], anchor["decoded_rgb_sha256"]] == expected_anchor
                    folder = output / f"{height}-{source.stem}"
                    pictures = paid.proof_files(client, packet, folder)
                    proof = paid.encode_report(client, identifier, source, evidence, pictures, folder,
                                               expected_text="no abrupt shot change")
                    settings.video_face_portrait_return_review = False
                    old = paid.checkout(client, payment, source, filename=source.name)
                    baseline = paid.body(client.get(f"/api/v2/analyses/{old}/evidence-packets"))
                    assert all(item.get("semantics") != METHOD for item in baseline["packets"])
                    assert tokens_assessed(identifier) == tokens_assessed(old)
                    baseline_ranges = [[item["window"]["start_stream_s"], item["window"]["end_stream_s"]]
                        for item in baseline["packets"] if any(key in item for key in ("face_sequence", "face_boundary", "face_return"))]
                    settings.video_face_portrait_return_review = True
                    browser.append({"id": identifier, "name": folder.name, "packet": packet["id"],
                        "target": packet["frames"][1]["anchor"], "frames": packet["window"]["frame_indices"]})
                    report_case = {"file": row["file"], "input_sha256": row["input_sha256"], "dimensions": panel["header"]["size"],
                        "labeled_frames": row["truth"], "selected_frames": expected, "previous_ranges_seconds": baseline_ranges,
                        "anchors": row["proposal"]["anchors"], "confidence_identical": True, **proof}
                    print("Verified app, exports and rollback:", folder.name, expected, flush=True)
                    break
                if report_case is None:
                    raise ValueError(f"No {height} pixel {delivery} edit retained a distinct location in the app")
                receipts.append(report_case)
        shown = browser_check(output, app, key, browser, scenario="video-face-shot-guard.mjs")
    write(output / "receipt.json", {"schema": "chipmunk.portrait-face-checkout/1", "held_sha256": sha(receipt),
        "cases": receipts, "browser": shown, "all_checks": all_checks,
        "simulated_paid_jobs": len(payment.sessions), "unpaid_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "freeze", "media", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--heights", nargs="+", choices=(640, 1280), type=int, required=True)
    parser.add_argument("--all-checks", action="store_true")
    run(**vars(parser.parse_args()))
