# SPDX-License-Identifier: Apache-2.0
"""Run production tracking and face appearance on pinned native swap/control clips."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path


def main():
    output, ordinal, role = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with tempfile.TemporaryDirectory(prefix="native-worker-", dir=output) as folder:
        work = Path(folder)
        root = Path(__file__).resolve().parents[2]
        os.environ.update(VERITAS_DATA_DIR=str(work), VERITAS_DATABASE_URL=f"sqlite:///{work}/case.db",
            VERITAS_MODELS_DIR=str(root / "models"), VERITAS_HF_HOME=str(root / "models/hf"),
            VERITAS_DEVICE="cpu", VERITAS_OFFLINE="true", VERITAS_RUN_WORKER="false",
            VERITAS_BILLING_MODE="disabled", VERITAS_VIDEO_FACE_REVIEW_MODE="appearance",
            VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="false", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="false",
            VERITAS_VIDEO_WHOLE_FACE_REVIEW="false", MPLCONFIGDIR=str(work / "matplotlib"))
        result = run(output, ordinal, role)
    print(json.dumps(result, allow_nan=False), flush=True)


def run(output, ordinal, role):
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
    from scripts import video_webcam_pose_transfer as evaluation

    start = time.monotonic()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    lock = evaluation.frozen(output)
    pin = lock["swaps"][ordinal - 1]
    index, archive = pin["index"], evaluation.raw.ARCHIVE
    require, common = evaluation.require, evaluation.common
    require(settings.offline and settings.device == "cpu" and settings.video_face_review_mode == "appearance"
            and evaluation.native.source_pin(index, archive) == pin, "Wrong app settings or native source")
    path = common.DATA / f"group-{index:03d}-complete.json"
    doc = common.read(path)
    cases = {case["condition"]: case for case in doc["cases"]}
    case = cases[role]
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    source = folder / case["file"]
    digest = common.digest(source)
    require(digest == case["sha256"], "Native video upload bytes changed")
    original, times, spans = evaluation.trial.prior.comparison.match_cases(folder, cases, camera=False)
    lo, hi = spans["native_long"]
    expected = case["expected_rgb_sha256"]
    truth = [str(times[lo]), str(times[hi])] if role in common.SWAP_CONDITIONS else None
    if truth:
        require([i for i, (a, b) in enumerate(zip(original["expected_rgb_sha256"], expected, strict=True)) if a != b]
                == list(range(lo, hi)), "The exact edited pictures differ from the authored half-open range")

    registry = get_registry()
    enabled = registry.enabled
    allowed = {DETECTOR, "vid.tracks.yunet_dinov2", "vid.faces.yunet"}
    require(allowed <= set(enabled("video", "full")), "Commercially scoped production video adapters required")
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in allowed]
    init_db()
    try:
        store = default_store()
        aid = str(uuid.uuid4())
        filename = "native-case.mp4"
        with source.open("rb") as payload:
            _, head, actual_digest, size = stream_upload(aid, filename, payload, settings.max_video_bytes)
        mime, suffix = validate_kind("video", head, filename, size)
        require(actual_digest == digest and size == case["bytes"] and (mime, suffix) == ("video/mp4", ".mp4"),
                "The native MP4 differs from the exact app upload")
        with get_session() as db:
            analysis = Analysis(id=aid, kind="video", filename=filename, content_type=mime,
                                size_bytes=size, sha256=digest, status="queued")
            db.add(analysis)
            db.commit()
        outcome = run_analysis(aid, deadline=Deadline(180), profile="full", registry=registry,
                               models=get_model_manager(), store=store)
        with get_session() as db:
            analysis = db.get(Analysis, aid)
            require(analysis.result_hash and analysis.sha256 == digest and analysis.profile == "full",
                    "The production pipeline did not complete its selected route")
            detectors = {row.detector_id: row for row in db.scalars(select(DetectorRun).where(DetectorRun.analysis_id == aid))}
            require(set(detectors) == allowed and all(detectors[key].status == "ok" for key in allowed - {DETECTOR}),
                    "The incumbent requires real production shot/track and face measurements")
            incumbent = detectors[DETECTOR]
            details = {}
            if incumbent.raw_output_artefact_id:
                asset = db.get(Artefact, incumbent.raw_output_artefact_id)
                store.verify(asset.sha256)
                details = json.loads(store.read_bytes(asset.sha256))["details"]
        table = outcome.outputs["probe"]["frame_table"]
        frozen_table = case["frame_table"]
        require(table["pts"] == frozen_table["pts"] and Fraction(table["time_base"]) == Fraction(frozen_table["time_base"])
                and (table["width"], table["height"]) == (frozen_table["width"], frozen_table["height"]),
                "Native production clock or geometry differs")
        if details.get("appearance_check_status") == "ok":
            selected = details.get("face_sequence")
            status = "incumbent_measured_no_flag"
        else:
            selected, status = None, "incumbent_unassessed"
        result = evaluation.case_result(selected, table, expected, truth, status)
        return {"group": ordinal, "role": role, **result, "elapsed_s": round(time.monotonic() - start, 4)}
    finally:
        shutdown()


if __name__ == "__main__":
    main()
