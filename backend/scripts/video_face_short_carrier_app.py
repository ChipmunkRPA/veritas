# SPDX-License-Identifier: Apache-2.0
"""Verify carrier-qualified face windows through the local prepaid app."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path

from scripts.video_face_paired_edge_hevc_app import sha_path, write_json
from scripts.video_face_short_windows_app import EDGES, REPRESENTATIVES, WINDOW_STARTS, verify_surfaces
from scripts.video_short_return_encoded_app import (
    ALLOW,
    PaymentStub,
    body,
    checkout,
    configure,
    read_result,
)

METHOD = "face-short-return/11"
VERSION = "video-evidence-24"
HEVC_METHOD = "face-short-return/10"
HEVC_VERSION = "video-evidence-23"
REPLAY_DETAIL = "Recomputed each carrier-qualified window shortlist and both native face boundaries"
CARRIERS = {
    "avc": {"extension": "mp4", "mime": "video/mp4", "encoding": ["h264", "yuv420p", "avc1"], "container": "mp4", "base": "1/30000"},
    "vp9": {"extension": "webm", "mime": "video/webm", "encoding": ["vp9", "yuv420p", "[0][0][0][0]"], "container": "webm", "base": "1/1000"},
}


def read(path):
    return json.loads(path.read_bytes())


def fixture_plan(study, names):
    fixtures = {}
    for name in names:
        fixture = study / f"{name}-held-1"
        held_path = study / f"{name}-held.json"
        held = read(held_path)
        assert held["freeze_sha256"] == sha_path(study / "freeze.json")
        frozen = {row["sha256"]: row for row in held["rows"]}
        local = {row["role"]: row for row in read(fixture / "results.json")}
        roster = {}
        spec = CARRIERS[name]
        for role in ("two_swaps", "genuine_cut"):
            row = local[role]
            source = fixture / f"{role}.{spec['extension']}"
            assert row["sha256"] == sha_path(source) and row == frozen[row["sha256"]]
            assert row["carrier"] == name and row["clock"]["time_base"] == spec["base"]
            expected = EDGES if role == "two_swaps" else ()
            assert row["truth"] == [list(pair) for pair in expected]
            assert [window["window_start"] for window in row["windows"]] == list(WINDOW_STARTS)
            assert all(window["assessed"] for window in row["windows"])
            assert tuple(tuple(window["match"]["frames"]) for window in row["windows"] if window["match"]) == expected
            if role == "two_swaps":
                anchors = [[item["frame_index"] for item in window["match"]["anchors"]] for window in row["windows"] if window["match"]]
                assert anchors == [list(indices) for indices in REPRESENTATIVES]
            assert len(row["clock"]["pts"]) == 240 and row["clock"]["pts"][0] == 0
            assert row["clock"]["pts"] == [(100 * i + 1) // 3 if name == "vp9" else i * 1000 for i in range(240)]
            roster[role] = source, row
        fixtures[name] = roster, sha_path(held_path)
    return fixtures


def coverage_windows(coverage, spec, commitment, *, positive):
    assert coverage["affects_overall_confidence"] is False
    assert coverage["source_encoding"] == spec["encoding"]
    assert coverage["source_carrier"] == spec["container"]
    assert coverage["source_time_base"] == spec["base"]
    assert coverage["source_first_pts"] == commitment["clock"]["pts"][0]
    assert [(row["start_frame_index"], row["end_frame_index_exclusive"], row["pixel_frames"])
            for row in coverage["windows"]] == [(start, start + 120, 120) for start in WINDOW_STARTS]
    wanted = ["localized", "localized", "no_distinct_range"] if positive else ["no_distinct_range"] * 3
    assert [row["status"] for row in coverage["windows"]] == wanted
    assert all(row["reason"] is None and 0 < row["measured_face_frames"] <= row["scheduled_frames"] for row in coverage["windows"])


def carrier_cases(client, stub, settings, name, fixtures, output, key, template):
    spec = CARRIERS[name]
    roster, held_sha = fixtures[name]
    other = next(records[0]["two_swaps"][0] for label, records in fixtures.items() if label != name)
    cases, decisions = {}, {}
    carrier_path = output / name
    carrier_path.mkdir(mode=0o700)
    for case in ("two_swaps", "genuine_cut", "feature_disabled"):
        selected = "genuine_cut" if case == "genuine_cut" else "two_swaps"
        source, commitment = roster[selected]
        settings.video_face_short_carrier_review = case != "feature_disabled"
        aid = checkout(client, stub, source, media_type=spec["mime"], filename=source.name)
        decisions[case], detector = read_result(aid, commitment["sha256"])
        evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
        assert evidence["source"]["sha256"] == commitment["sha256"]
        assert evidence["source"]["time_base"] == spec["base"]
        header = client.get(f"/api/analyses/{aid}/source", headers={"Range": "bytes=0-31"})
        assert header.status_code == 206 and header.content == source.read_bytes()[:32]
        assert header.headers["content-type"] == spec["mime"]
        packets = [packet for packet in evidence["packets"] if packet.get("face_return", {}).get("method") == METHOD]
        coverage = evidence["coverage"].get("brief_face_windows")
        record = {"source_sha256": commitment["sha256"], "version": evidence["version"], "new_packets": len(packets),
                  "coverage": coverage, "original_media_mime_and_byte_range_verified": True}
        print(name, case, evidence["version"], len(packets), flush=True)
        if case == "feature_disabled":
            assert evidence["version"] != VERSION and not packets and coverage is None
        else:
            assert evidence["version"] == VERSION and len(packets) == (2 if case == "two_swaps" else 0)
            coverage_windows(coverage, spec, commitment, positive=case == "two_swaps")
            assert [packet["face_return"] for packet in packets] == detector["details"].get("face_short_windows", [])
            proof, browser = verify_surfaces(client, aid, source, commitment, evidence, packets, carrier_path / case,
                                             replay_detail=REPLAY_DETAIL, wrong_source=other, expected_template=template)
            record.update(proof)
            if packets:
                pts, tb = commitment["clock"]["pts"], Fraction(spec["base"])
                record.update(native_ranges=[{"end_exclusive_frames": list(edges), "ticks": [pts[i] for i in edges],
                                             "time_s": [round(float(pts[i] * tb), 6) for i in edges],
                                             "representative_frames": list(anchors),
                                             "representative_ticks": [pts[i] for i in anchors]}
                                            for edges, anchors in zip(EDGES, REPRESENTATIVES, strict=True)])
                write_json(carrier_path / "browser-input.json", {"analysis_id": aid, "packets": browser, "verify_browser_seek": True})
                key_path = carrier_path / "browser-key.txt"
                key_path.write_text(key)
                key_path.chmod(0o600)
        cases[case] = record
    assert decisions["two_swaps"] == decisions["feature_disabled"]
    return {"cases": cases, "held_receipt_sha256": held_sha, "overall_decision_identical_flag_on_off": True}


def run(study, hevc_study, output, names, template):
    fixtures = fixture_plan(study, CARRIERS)
    configure(output, Path(__file__).resolve().parents[2])
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW="true",
                      VERITAS_VIDEO_FACE_SHORT_WINDOW_REVIEW="true", VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true",
                      VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT="true", VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true",
                      VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true", VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true",
                      VERITAS_MIN_FREE_BYTES=str(256 * 1024**2), VERITAS_API_KEY_PEPPER="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session

    assert settings.data_dir == (output / "app").resolve() and settings.video_face_short_carrier_review
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert set(enabled("video", "full")) >= ALLOW
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("face-carrier-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        verified = {name: carrier_cases(client, stub, settings, name, fixtures, output, key, template) for name in names}
        settings.video_face_short_carrier_review = True
        source = hevc_study / "two_swaps.mp4"
        commitment = next(row for row in read(hevc_study / "results.json") if row["role"] == "two_swaps")
        assert sha_path(source) == commitment["sha256"]
        aid = checkout(client, stub, source, filename=source.name)
        read_result(aid, commitment["sha256"])
        legacy = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
        assert legacy["version"] == HEVC_VERSION
        prior_ranges = [packet["face_return"] for packet in legacy["packets"] if packet.get("face_return", {}).get("method") == HEVC_METHOD]
        assert [[row[key] for key in ("start_frame_index", "end_frame_index")] for row in prior_ranges] == [list(pair) for pair in EDGES]
    write_json(output / "receipt.json", {"carriers": verified, "unpaid_jobs": 0,
                                         "simulated_paid_jobs": 3 * len(names) + 1, "existing_hevc_preserved": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Frozen AVC and VP9 held fixture root")
    parser.add_argument("hevc_study", type=Path, help="Frozen eight-second HEVC held fixture directory")
    parser.add_argument("output", type=Path, help="New isolated output directory")
    parser.add_argument("--carriers", nargs="+", choices=CARRIERS, default=list(CARRIERS))
    parser.add_argument("--template", default="veritas-report-2.0.r113")
    args = parser.parse_args()
    run(args.study.resolve(), args.hevc_study.resolve(), args.output.resolve(), tuple(dict.fromkeys(args.carriers)), args.template)


if __name__ == "__main__":
    main()
