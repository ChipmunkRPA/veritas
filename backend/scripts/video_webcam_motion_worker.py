# SPDX-License-Identifier: Apache-2.0
"""Run one isolated production motion-transfer case."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def environment(root, work):
    os.environ.update(VERITAS_DATA_DIR=str(work), VERITAS_DATABASE_URL=f"sqlite:///{work}/case.db",
        VERITAS_MODELS_DIR=str(root / "models"), VERITAS_HF_HOME=str(root / "models/hf"),
        VERITAS_DEVICE="cpu", VERITAS_OFFLINE="true", VERITAS_RUN_WORKER="false",
        VERITAS_BILLING_MODE="disabled", VERITAS_VIDEO_FACE_REVIEW_MODE="appearance",
        VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="false", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="false",
        VERITAS_VIDEO_WHOLE_FACE_REVIEW="false", MPLCONFIGDIR=str(work / "matplotlib"))


def native_transfer(folder, ordinal, study):
    from scripts import video_webcam_pose_native_worker as prior_worker

    lock = study.frozen(folder)
    pin = lock["primary_positive_pins"][ordinal - 1]
    role = lock["positive_roles"][ordinal - 1]
    doc = study.common.read(study.common.DATA / f"group-{pin['index']:03d}-complete.json")
    case = next(row for row in doc["cases"] if row["condition"] == role)
    path = study.pose.raw.ARCHIVE / "video-native-inserts-v95" / f"group-{pin['index']:03d}" / case["file"]

    def compare(record, table, expected, truth=None, status=None):
        if not record:
            return {"family": "native_swap", "truth": truth, "incumbent_extent": None, "status": status}
        return {"family": "native_swap", "truth": truth, **study.inspect(path, record, table, expected)}
    study.pose.case_result = compare
    return prior_worker.run(study.POSITIVE_RUN, ordinal, role)


def source_delivery(source, work, study):
    """Use a bounded 720p lossless delivery, preserving each source-relative picture time."""
    common, require = study.common, study.require
    table = common.probe_frames(source, timeout_s=35)
    require(table.width * 9 == table.height * 16 and not (table.rotation or table.truncated or table.interpolated),
            "Upright complete 16:9 controlled source expected")
    require(150 <= len(table) <= 950, "Controlled study duration outside the bounded source lane")
    output = work / "source-delivery.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "2",
                    "-i", str(source), "-map", "0:v:0", "-an", "-vf", "scale=1280:720:flags=area,format=rgb24",
                    "-fps_mode", "passthrough", "-c:v", "libx264rgb", "-preset", "ultrafast",
                    "-qp", "0", "-bf", "0", "-threads", "2", "-enc_time_base", f"{table.tb}",
                    "-video_track_timescale", str(table.tb.denominator), "-n", str(output)], check=True, timeout=100)
    delivery = common.probe_frames(output, timeout_s=30)
    require(not (delivery.rotation or delivery.truncated or delivery.interpolated)
            and (delivery.width, delivery.height) == (1280, 720)
            and [(pts - delivery.pts[0]) * delivery.tb for pts in delivery.pts]
                == [(pts - table.pts[0]) * table.tb for pts in table.pts],
            "Lossless delivery changed the source picture clock")
    return output, delivery


def genuine_app(folder, ordinal, work, study):
    import cv2
    import torch
    from sqlalchemy import select

    from app.core.config import settings
    from app.detectors.video.adapters.face_appearance import DETECTOR
    from app.engine.boundary import Deadline
    from app.engine.models import get_model_manager
    from app.engine.pipelines import run_analysis
    from app.engine.registry import get_registry
    from app.storage.db import Analysis, Artefact, DetectorRun, get_session, init_db, shutdown
    from app.storage.files import stream_upload, validate_kind
    from app.storage.objects import default_store

    start = time.monotonic()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    lock = study.frozen(folder)
    receipt = lock["gros"][ordinal - 1]
    source = study.GROS_SOURCE / receipt["file"]
    require, common = study.require, study.common
    require(common.digest(source) == receipt["sha256"] and source.stat().st_size == receipt["bytes"],
            "Licensed genuine original differs from its receipt")
    delivery, exact = source_delivery(source, work, study)
    registry = get_registry()
    enabled = registry.enabled
    allow = {DETECTOR, "vid.tracks.yunet_dinov2", "vid.faces.yunet"}
    require(allow <= set(enabled("video", "full")) and settings.offline and settings.device == "cpu",
            "Exact commercially scoped application detectors required")
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in allow]
    init_db()
    try:
        store = default_store()
        aid = str(uuid.uuid4())
        filename = "licensed-motion-control.mp4"
        digest = common.digest(delivery)
        with delivery.open("rb") as payload:
            _, head, actual, size = stream_upload(aid, filename, payload, settings.max_video_bytes)
        mime, suffix = validate_kind("video", head, filename, size)
        require(actual == digest and size == delivery.stat().st_size and (mime, suffix) == ("video/mp4", ".mp4"),
                "Local application did not receive the exact controlled source")
        with get_session() as db:
            db.add(Analysis(id=aid, kind="video", filename=filename, content_type=mime, size_bytes=size, sha256=digest, status="queued"))
            db.commit()
        outcome = run_analysis(aid, deadline=Deadline(135), profile="full", registry=registry,
                               models=get_model_manager(), store=store)
        require(outcome.status == "done" and outcome.outputs["probe"]["frame_table"] == exact.to_json(),
                "Controlled video app run did not finish on the exact clock")
        with get_session() as db:
            detectors = {row.detector_id: row for row in db.scalars(select(DetectorRun).where(DetectorRun.analysis_id == aid))}
            require(set(detectors) == allow and all(detectors[key].status == "ok" for key in allow - {DETECTOR}),
                    "Current production tracking was unavailable")
            detector = detectors[DETECTOR]
            details = {}
            if detector.raw_output_artefact_id:
                asset = db.get(Artefact, detector.raw_output_artefact_id)
                store.verify(asset.sha256)
                details = json.loads(store.read_bytes(asset.sha256))["details"]
        status = "incumbent_measured_no_flag" if details.get("appearance_check_status") == "ok" else "incumbent_unassessed"
        if selected := details.get("face_sequence"):
            if selected["method"] == "face-appearance-change/1":
                result = study.inspect(delivery, selected, exact.to_json())
            else:
                result = {"status": "incumbent_unassessed", "incumbent_extent": None}
        else:
            result = {"status": status, "incumbent_extent": None}
        return {"family": "gros_clean_motion", "group": ordinal, "exact_source_pictures": len(exact),
                "delivery_sha256": digest, **result, "elapsed_s": round(time.monotonic() - start, 3)}
    finally:
        shutdown()


def main():
    folder, family, ordinal = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="motion-worker-", dir=folder) as work:
        work = Path(work)
        environment(root, work)
        from scripts import video_webcam_motion_transfer as study

        if family == "positive":
            result = native_transfer(folder, ordinal, study)
        else:
            result = genuine_app(folder, ordinal, work, study)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
