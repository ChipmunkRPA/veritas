# SPDX-License-Identifier: Apache-2.0
"""Replay the actual local Full face-review lanes on the frozen cut panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from scripts import video_short_return_encoded_app as paid

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "backend/app"
BINDINGS = (
    "detectors/video/shots.py", "detectors/video/tracking.py", "detectors/video/face_appearance.py",
    "detectors/video/face_return.py", "detectors/video/face_boundary.py", "detectors/video/adapters/face_appearance.py",
    "detectors/video/adapters/face_sequence.py", "detectors/video/adapters/tracks.py", "engine/stages/video.py",
    "engine/pipelines.py", "models.yaml",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(folder, roster):
    lock = {
        "schema": "chipmunk.video-native-background-panel-incumbent/1",
        "freeze_sha256": digest(folder / "freeze.json"),
        "media_sha256": digest(folder / "media.json"),
        "checkout_transport_sha256": digest(Path(paid.__file__)),
        "worker_sha256": digest(Path(__file__)),
        "app_files": {name: digest(APP / name) for name in BINDINGS},
        "source_hashes": [row["sha256"] for row in roster],
        "profile": "full", "device": "cpu", "isolated_detectors": sorted(paid.ALLOW),
    }
    return lock


def compact(selected):
    if not selected:
        return None
    return {**{key: selected[key] for key in ("method", "time_base", "track_id", "start_pts", "end_pts", "start_stream_s", "end_stream_s", "response")},
            "frame_inputs": [{key: row[key] for key in ("frame_index", "pts", "decoded_rgb_sha256", "bbox")}
                             for row in selected["frame_inputs"]]}


def scope_range(record):
    if not record:
        return None
    return {key: record[key] for key in ("method", "time_base", "start_pts", "end_pts", "start_stream_s", "end_stream_s")}


def run(folder, archive):
    isolated = folder / "isolated-incumbent"
    paid.configure(isolated, ROOT)

    from app.core.config import settings

    if (settings.data_dir.resolve() != (isolated / "app").resolve()
            or str(settings.database_url) != os.environ["VERITAS_DATABASE_URL"]
            or settings.stripe_webhook_secret is None
            or settings.stripe_webhook_secret.get_secret_value() != paid.WEBHOOK_SECRET):
        raise RuntimeError("The isolated test settings must be loaded before application startup")

    from scripts import video_native_background_panel as panel

    common, require = panel.common, panel.require
    panel.frozen(folder, archive or panel.ARCHIVE)
    manifest = common.read(folder / "media.json")
    require(manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Frozen source hash differs")
    roster = manifest["media"]
    require(len(roster) == 64 and len({row["sha256"] for row in roster}) == 64, "Full prospective roster required")
    lock = commit(folder, roster)
    common.write(folder / "incumbent-lock.json", lock)
    import cv2
    import torch
    from fastapi.testclient import TestClient
    from sqlalchemy import select

    from app.api import auth
    from app.billing import service
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import Analysis, AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    require(paid.ALLOW.issubset(enabled("video", "full")), "Actual Full CPU detector path unavailable")
    registry.enabled = lambda modality, profile: [name for name in enabled(modality, profile) if name in paid.ALLOW]
    payments = paid.PaymentStub()
    service._stripe = payments

    def observe(aid, expected, client):
        with get_session() as db:
            analysis = db.get(Analysis, aid)
            require(analysis.sha256 == expected["sha256"] and analysis.profile == "full", "Wrong app source or profile")
            outputs = {}
            for name in ("shots", "tracks", "detect"):
                stage = db.scalar(select(AnalysisStage).where(AnalysisStage.analysis_id == aid, AnalysisStage.stage == name))
                require(stage.status == "done" and stage.output_artefact_id is not None, "Incomplete worker stage")
                artefact = db.get(Artefact, stage.output_artefact_id)
                outputs[name] = json.loads(default_store().read_bytes(artefact.sha256))
        [appearance] = [row for row in outputs["detect"]["detectors"] if row["detector_id"] == "vid.review.face_appearance"]
        [tracking] = [row for row in outputs["tracks"]["detectors"] if row["detector_id"] == "vid.tracks.yunet_dinov2"]
        details = appearance.get("details") or {}
        result = {"shot_boundaries": [item["frame_index"] for item in outputs["shots"]["boundaries"]],
                  "frames_scanned_for_shots": outputs["shots"]["frames_scanned"], "track_status": tracking["status"],
                  "tracks": len(outputs["tracks"]["tracks"]), "coarse_status": details.get("appearance_check_status"),
                  "appearance_status": appearance["status"], "coarse": compact(details.get("face_sequence")),
                  "boundary_refinement": scope_range(details.get("face_boundary")),
                  "short_range": scope_range(details.get("face_return")), "short_coverage": details.get("face_return_check")}
        require(result["frames_scanned_for_shots"] == panel.N, "Incomplete native shot scan")
        if result["coarse"]:
            for item in result["coarse"]["frame_inputs"]:
                i = item["frame_index"]
                require(item["pts"] == expected["clock"]["pts"][i] and item["decoded_rgb_sha256"] == expected["rgb_sha256"][i],
                        "Incumbent crop is not bound to the exact delivered picture")
        packets = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))["packets"]
        result["displayed"] = [
            {key: scope_range(packet[key]) for key in ("face_sequence", "face_boundary", "face_return") if packet.get(key)}
            for packet in packets if any(packet.get(key) for key in ("face_sequence", "face_boundary", "face_return"))
        ]
        return result

    started = time.monotonic()
    journal = folder / "private-incumbent.jsonl"
    with journal.open("x") as stream, TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("native-background-panel-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for number, row in enumerate(roster, 1):
            path = folder / "media" / row["file"]
            require(common.digest(path) == row["sha256"], "Submission bytes differ")
            before = time.monotonic()
            aid = paid.checkout(client, payments, path)
            result = observe(aid, row, client)
            entry = {**{name: row[name] for name in ("file", "sha256", "pair", "direction", "condition", "truth", "interval")},
                     **result, "paid_before_analysis": True, "paid_event_idempotent": True, "elapsed_s": time.monotonic()-before}
            stream.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            if number % 8 == 0:
                print("finished paid Full worker", number, "of", len(roster), flush=True)
    require(commit(folder, roster) == lock, "The incumbent or source bindings changed while running")
    common.write(folder / "incumbent-seal.json", {"status":"verified","cases": len(roster),
        "journal_sha256": common.digest(journal), "lock_sha256": common.digest(folder / "incumbent-lock.json"),
        "media_sha256": common.digest(folder / "media.json"), "elapsed_s": time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    run(args.folder, args.archive)


if __name__ == "__main__":
    main()
