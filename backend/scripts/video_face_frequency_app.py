"""Verify the local face-detail fallback through paid analysis and public exports."""

from __future__ import annotations

import argparse
import math
import os
from copy import deepcopy
from pathlib import Path

from scripts import video_face_frequency_fit as study
from scripts import video_face_residual_fit as previous
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_face_paired_edge_hevc_app import sha_path, write_json
from scripts.video_face_same_frame_app import verify_deliveries
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def examples(receipt):
    rows = receipt["panels"]["held"]["rows"]
    method = receipt["freeze"]["selection_order"][0]
    point = receipt["freeze"]["points"][method]["wider"]
    negative = {row["group"]: row for row in rows if row["family"] == "ordinary" and row["delivery"] == "tandem"
                and not previous.baseline(row) and row["scores"][method] is not None and row["scores"][method] < point - .002}
    additions = [row for row in rows if row["family"] != "ordinary" and row["delivery"] == "tandem"
                 and row["group"] in negative and not previous.baseline(row) and previous.above(row["scores"][method], point + .002)]
    first = additions[0]
    second = next(row for row in additions if row["group"] != first["group"] and row["family"] != first["family"])
    existing = next(row for row in rows if row["family"] != "ordinary" and previous.baseline(row))
    return ((first, False), (first, True), (second, True), (negative[first["group"]], True), (existing, True))


def run(clips, receipt_path, output, device, *, browser):
    receipt = study.read(receipt_path)
    study.verify(receipt)
    method = receipt["freeze"]["selection_order"][0]
    point = receipt["freeze"]["points"][method]["wider"]
    output = output.resolve()
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE=device, VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"),
                      VERITAS_VIDEO_WHOLE_FACE_REVIEW="true", VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW="true",
                      VERITAS_VIDEO_WHOLE_FACE_DETAIL_REVIEW="true", VERITAS_LOG_LEVEL="WARNING")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.video.adapters.face_detail import DETECTOR
    from app.detectors.video.adapters.whole_face import DETECTOR as CURRENT
    from app.engine.registry import get_registry
    from app.evidence import face_detail_binding as detail
    from app.evidence import whole_face_binding as binding
    from app.main import app
    from app.storage.db import get_session

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert point == detail.POINT
    registry = get_registry()
    enabled = registry.enabled
    assert {CURRENT, DETECTOR}.issubset(enabled("video", "full"))
    registry.enabled = lambda modality, profile: [item for item in enabled(modality, profile) if item in (CURRENT, DETECTOR)]
    stub = PaymentStub()
    service._stripe = stub
    cases, browser_cases, baseline = [], [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("face-detail-local-test", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for row, on in examples(receipt):
            settings.video_whole_face_detail_review = on
            source = clips / row["file"]
            assert sha_path(source) == row["sha256"]
            aid = checkout(client, stub, source)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == row["sha256"]
            face_packets = [item for item in evidence["packets"] if item.get("whole_face_review")]
            check, = [item for item in face_packets if item["semantics"] != detail.METHOD]
            added = [item for item in face_packets if item["semantics"] == detail.METHOD]
            cap = previous.baseline(row)
            assert check["whole_face_review"]["flagged"] == cap
            assert len(added) == (on and not cap)
            assessment = tokens_assessed(aid)
            if not on:
                baseline[row["sha256"]] = assessment
            if on and row["sha256"] in baseline:
                assert assessment == baseline[row["sha256"]]
            name = f"{source.stem}-{'on' if on else 'off'}"
            case = {"name": name, "source_sha256": row["sha256"], "existing_flag": cap, "detail_enabled": on}
            if added:
                packet, = added
                record = packet["whole_face_review"]
                assert record["prior"] == check["whole_face_review"]
                assert record["schema"] == detail.METHOD and packet["affects_overall_confidence"] is False
                expected = row["scores"][method]
                assert math.isclose(float.fromhex(record["score_hex"]), expected, abs_tol=.00002)
                assert record["flagged"] == (expected > point)
                assert [[r["frame_index"], r["pts"], r["decoded_rgb_sha256"], r["aligned_rgb_sha256"]] for r in record["samples"]] == [
                    [r["index"], r["pts"], r["rgb_sha256"], r["crop_sha256"]] for r in row["frames"]]
                case.update(detail_flag=record["flagged"], measured_frames=len(record["samples"]))
                altered = deepcopy(packet)
                altered["whole_face_review"]["terms_hex"][0] = (float.fromhex(record["terms_hex"][0]) + .1).hex()
                try:
                    binding.validate_packet(altered, evidence["source"])
                except ValueError as error:
                    assert "contributions" in str(error)
                else:
                    raise AssertionError("Changing a bound contribution should fail verification")
                case.update(verify_deliveries(client, aid, source, evidence, packet, output / name))
                browser_cases.append({"id": aid, "name": name, "packet": packet["id"], "title": packet["title"],
                                      "target": packet["frames"][1]["anchor"], "frames": packet["window"]["frame_indices"]})
            cases.append(case)
            print(name, case.get("detail_flag"), flush=True)
        browser_receipt = browser_check(output, app, key, browser_cases, scenario="face-detail-replay.mjs") if browser else None
    write_json(output / "receipt.json", {"cases": cases, "browser": browser_receipt, "unpaid_jobs": 0,
                                         "simulated_paid_jobs": len(cases), "unchanged_overall_for_toggle": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clips", "receipt", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.clips, args.receipt, args.output, args.device, browser=args.browser)
