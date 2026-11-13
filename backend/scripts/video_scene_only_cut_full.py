# SPDX-License-Identifier: Apache-2.0
"""Route all matched cut fixtures through the isolated paid Full face pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from fractions import Fraction
from pathlib import Path

from scripts import video_short_return_encoded_app as paid

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.video-scene-only-cut-full/1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def scope(record, pts):
    return [pts.index(record[k]) for k in ("start_pts", "end_pts")] if record else None


def run(study, output):
    manifest = study / "media.json"
    plan = json.loads(manifest.read_bytes())
    require(plan["freeze_sha256"] == sha(study / "freeze.json") and len(plan["rows"]) == 32,
            "Existing frozen factorial study is required")
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_MIN_FREE_BYTES=str(256 * 1024**2),
                      VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT="true", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="true",
                      VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true", VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true",
                      VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true", VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true",
                      VERITAS_WEBHOOKS_ENABLED="false")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store

    require(settings.data_dir == (output / "app").resolve() and settings.video_face_spatial_cut_context,
            "The worker must use the isolated state and enabled picture context")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    require(paid.ALLOW.issubset(enabled("video", "full")), "Full appearance lane is unavailable")
    registry.enabled = lambda modality, profile: [name for name in enabled(modality, profile) if name in paid.ALLOW]
    payment = paid.PaymentStub()
    service._stripe = payment
    roster = []
    start = time.monotonic()
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("scene-only-cut-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for number, case in enumerate(plan["rows"], 1):
            path = study / "media" / case["name"]
            require(sha(path) == case["sha256"], "Frozen upload changed")
            aid = paid.checkout(client, payment, path, filename="scene-cut-study.mp4")
            _, detector = paid.read_result(aid, case["sha256"])
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            require(evidence["source"]["sha256"] == case["sha256"], "Public evidence did not bind the upload")
            with get_session() as db:
                stage = db.get(AnalysisStage, (aid, "shots"))
                require(stage.status == "done", "The worker did not complete its shot stage")
                objects = default_store()
                shots = json.loads(objects.read_bytes(db.get(Artefact, stage.output_artefact_id).sha256))
            table = probe_frames(path, timeout_s=20)
            require(len(table) == shots["frames_scanned"] == 120 and table.time_base == evidence["source"]["time_base"],
                    "Worker shot scan or public source clock differs")
            details = detector.get("details") or {}
            coarse = scope(details.get("face_sequence"), table.pts)
            brief = scope(details.get("face_return"), table.pts)
            record = details.get("face_spatial_cut")
            cut_range = scope(record, table.pts)
            packets = [p for p in evidence["packets"] if p.get("face_spatial_cut")]
            require(bool(packets) == bool(record) and len(packets) <= 1, "Private and public cut evidence differ")
            anchors = []
            if record:
                packet, = packets
                require(packet["face_spatial_cut"] == record and packet["affects_overall_confidence"] is False,
                        "The public cut record or confidence scope differs")
                items = packet["face_spatial_cut_frames"]
                by_index = {p["anchor"]["frame_index"]: p for p in items}
                indices = [cut_range[0] - 1, sum(cut_range) // 2, cut_range[1]]
                require(list(by_index) == indices and len(items) == 3, "Wrong public cut frames")
                for frame in decode_frames(path, table, indices, timeout_s=20):
                    item = by_index[frame.index]["anchor"]
                    rgb = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                    require(frame.pts_verified and frame.pts == item["pts"] and rgb == item["decoded_rgb_sha256"],
                            "Public cut frame or time differs from the submitted original")
                    anchors.append({"frame": frame.index, "time": str(frame.pts * table.tb), "rgb_sha256": rgb})
                require(len(anchors) == 3, "Incomplete public cut anchor readback")
                exported = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
                require(exported["evidence"] == evidence, "JSON export differs from the API evidence")
            if coarse is None:
                route = "no_coarse"
            elif brief:
                route = "brief_precedence"
            elif not 5 <= coarse[1] - coarse[0] <= 42:
                route = "coarse_duration_outside_scope"
            elif record:
                route = "published"
            else:
                route = "eligible_without_published_cut"
            roster.append({**{k: case[k] for k in ("pair", "direction", "role", "span")},
                           "appearance_status": detector["status"], "coarse": coarse, "brief": brief,
                           "route": route, "cut": cut_range, "anchors": anchors,
                           "shot_boundaries": [b["frame_index"] for b in shots["boundaries"]],
                           "authored_time": [str(Fraction(i, 30)) for i in case["span"]]})
            print(number, case["role"], route, cut_range, flush=True)
    write(output / "receipt.json", {"schema": SCHEMA, "media_sha256": sha(manifest),
                                    "simulated_paid_jobs": len(roster), "unpaid_jobs": 0,
                                    "isolated_production_full_face_pipeline": True,
                                    "runtime_s": round(time.monotonic() - start, 2), "rows": roster})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
