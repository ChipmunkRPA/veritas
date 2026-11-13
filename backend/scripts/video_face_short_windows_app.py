# SPDX-License-Identifier: Apache-2.0
"""Verify overlapping face-review windows in an isolated prepaid local analysis."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts.video_face_paired_edge_hevc_app import image_signature, sha, sha_path, write_json
from scripts.video_face_unrouted_short_app import METHOD as LEGACY_METHOD
from scripts.video_face_unrouted_short_app import VERSION as LEGACY_VERSION
from scripts.video_face_unrouted_short_app import fixture_plan as legacy_plan
from scripts.video_short_return_encoded_app import (
    ALLOW,
    PaymentStub,
    body,
    checkout,
    configure,
    fetch,
    read_result,
    wait,
)

METHOD = "face-short-return/10"
VERSION = "video-evidence-23"
WINDOW_STARTS = (0, 60, 120)
EDGES = ((71, 77), (128, 134))
REPRESENTATIVES = ((70, 74, 77), (127, 131, 134))
ROLES = ("before", "during", "after")


def fixture_plan(study: Path, legacy: Path) -> dict:
    results = {row["role"]: row for row in json.loads((study / "results.json").read_bytes())}
    held = json.loads((study.parent / "held.json").read_bytes())
    assert held["freeze_sha256"] == sha_path(study.parent / "freeze.json")
    frozen = {row["sha256"]: row for row in held["rows"]}
    fixtures = {}
    for case, role in (("two_swaps", "two_swaps"), ("genuine_cut", "genuine_cut"), ("feature_disabled", "two_swaps")):
        record = results[role]
        source = study / f"{role}.mp4"
        assert sha_path(source) == record["sha256"]
        assert record == frozen[record["sha256"]]
        assert (record["clock"]["width"], record["clock"]["height"]) == (1280, 720)
        assert record["clock"]["time_base"] == "1/30000" and len(record["clock"]["pts"]) == 240
        assert [row["window_start"] for row in record["windows"]] == list(WINDOW_STARTS)
        assert all(row["assessed"] for row in record["windows"])
        expected = EDGES if role == "two_swaps" else ()
        assert record["truth"] == [list(pair) for pair in expected]
        assert tuple(tuple(row["match"]["frames"]) for row in record["windows"] if row["match"]) == expected
        fixtures[case] = (source, record)
    fixtures["legacy_four_seconds"] = (legacy / "source_edits-g1-two_swaps.mp4", legacy_plan(legacy)["short_swap"])
    return fixtures


def verify_images(client, source, record, packets, folder):
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(source)
    assert list(table.pts) == record["clock"]["pts"] and table.rotation == 0 and not table.interpolated and not table.truncated
    wanted = sorted(index for indices in REPRESENTATIVES for index in indices)
    frames = {frame.index: frame for frame in decode_frames(source, table, wanted)}
    signatures, assets, browser = set(), [], []
    for packet, indices, edges, window_start in zip(packets, REPRESENTATIVES, EDGES, WINDOW_STARTS[:2], strict=True):
        measured = packet["face_return"]
        assert packet["affects_overall_confidence"] is False
        assert [measured[k] for k in ("start_frame_index", "end_frame_index")] == list(edges)
        assert [measured[k] * table.tb for k in ("start_pts", "end_pts")] == [Fraction(index, 30) for index in edges]
        assert measured["window_start_frame_index"] == window_start and measured["anchors"] == dict(zip(ROLES, indices, strict=True))
        assert [measured[k] for k in ("start_stream_s", "end_stream_s")] == [round(index / 30, 6) for index in edges]
        browser.append({"packet_id": packet["id"], "frames": [row["id"] for row in packet["frames"]],
                        "indices": list(indices), "times_s": [round(index / 30, 3) for index in edges]})
        for row, role, index in zip(packet["frames"], ROLES, indices, strict=True):
            anchor = row["anchor"]
            frame = frames[index]
            assert row["interval_role"] == role and row["score"] is None
            assert frame.pts_verified and anchor["frame_index"] == index and frame.pts == table.pts[index] == anchor["pts"]
            assert anchor["stream_time_s"] == round(float(table.pts[index] * table.tb), 6)
            rgb_hash = sha(frame.rgb.tobytes())
            assert rgb_hash == record["native_hashes"][index] == anchor["decoded_rgb_sha256"]
            box = row["context_region"]
            x, y, width, height = (box[key] for key in "xywh")
            expected = {"original": ((table.width, table.height), rgb_hash),
                        "crop": ((width, height), sha(frame.rgb[y:y + height, x:x + width].tobytes()))}
            for kind, signature in expected.items():
                asset = packet["artifacts"][row["artifact_refs"][kind]]
                data = fetch(client, asset["url"])
                assert sha(data) == asset["sha256"] and image_signature(data) == signature
                signatures.add(signature)
                (folder / f"{index}-{kind}.png").write_bytes(data)
                assets.append({"role": role, "kind": kind, "frame": index, "pts": table.pts[index], "sha256": asset["sha256"]})
    assert len(signatures) == len(assets) == 12
    return signatures, assets, browser


def signed_replay(client, aid, source, evidence, report, pdf, folder):
    from app.reports.verify import verify_package

    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                              json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    signed = folder / "signed-package.zip"
    signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    signer = body(client.get("/api/v2/instance/public-key"))
    key = signer["public_key_pem"].encode()
    replay = verify_package(signed, public_key_pem=key, expect_key_id=signer["key_id"], source_video=source)
    assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
    with ZipFile(signed) as archive:
        bundle = json.loads(archive.read("bundle.json"))
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read(bundle["reports"][0]["file"]) == pdf
    altered = folder / "altered-original.mp4"
    altered.write_bytes(source.read_bytes() + b"\n")
    assert not verify_package(signed, public_key_pem=key, source_video=altered).ok
    return replay


def verify_surfaces(client, aid, source, commitment, evidence, packets, folder):
    import pymupdf

    from app.reports import engine
    from scripts.verify_multimedia_exports import scan

    folder.mkdir(mode=0o700)
    images, assets, browser = verify_images(client, source, commitment, packets, folder) if packets else (set(), [], [])
    scan(fetch(client, f"/api/v2/analyses/{aid}").decode())
    scan(fetch(client, f"/api/v2/analyses/{aid}/findings").decode())
    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(exported)["evidence"] == evidence
    scan(exported.decode())
    (folder / "export.json").write_bytes(exported)
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        assert "Window" in words
        if packets:
            assert "Brief face-change range to review" in words and "with an exclusive end" in words
            assert all(f"{start / 30:.3f}-{end / 30:.3f} s" in words for start, end in EDGES)
            assert all(f"source frame {index}" in words for indices in REPRESENTATIVES for index in indices)
            xrefs = {item[0] for page in document for item in page.get_images(full=True)}
            assert images <= {image_signature(document.extract_image(xref)["image"]) for xref in xrefs}
        else:
            assert "supplied no additional brief face-change range" in words
    (folder / "report.pdf").write_bytes(pdf)
    replay = signed_replay(client, aid, source, evidence, report, pdf, folder)
    detail = "Recomputed each overlapping window shortlist and both native face boundaries"
    assert any(item.ok and item.name == "original video frames" and (not packets or detail in item.detail)
               for item in replay.checks), replay.as_dict()
    assert engine.TEMPLATE_SET == "veritas-report-2.0.r113"
    receipt = {"exact_source_images": assets, "pdf_api_json_agree": True, "signed_original_replay": True,
               "different_original_rejected": True, "public_model_names_absent": True, "template_set": engine.TEMPLATE_SET}
    return receipt, browser


def run(study, legacy, output):
    fixtures = fixture_plan(study, legacy)
    configure(output, Path(__file__).resolve().parents[2])
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_SHORT_WINDOW_REVIEW="true",
                      VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true", VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT="true",
                      VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true", VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true",
                      VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true", VERITAS_MIN_FREE_BYTES=str(256 * 1024**2),
                      VERITAS_API_KEY_PEPPER="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session

    assert settings.data_dir == (output / "app").resolve() and settings.video_face_short_window_review
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert set(enabled("video", "full")) >= ALLOW
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    cases, decisions = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("face-window-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for case, (source, commitment) in fixtures.items():
            settings.video_face_short_window_review = case != "feature_disabled"
            aid = checkout(client, stub, source)
            decisions[case], detector = read_result(aid, commitment["sha256"])
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == commitment["sha256"]
            windows = [packet for packet in evidence["packets"] if packet.get("face_return", {}).get("method") == METHOD]
            legacy_packets = [packet for packet in evidence["packets"] if packet.get("face_return", {}).get("method") == LEGACY_METHOD]
            coverage = evidence["coverage"].get("brief_face_windows")
            row = {"file_sha256": commitment["sha256"], "version": evidence["version"], "window_packets": len(windows),
                   "legacy_packets": len(legacy_packets), "window_coverage": coverage}
            print(case, evidence["version"], len(windows), len(legacy_packets), coverage, flush=True)
            if case in {"two_swaps", "genuine_cut"}:
                assert evidence["version"] == VERSION and coverage["affects_overall_confidence"] is False
                assert [(item["start_frame_index"], item["end_frame_index_exclusive"], item["pixel_frames"])
                        for item in coverage["windows"]] == [(start, start + 120, 120) for start in WINDOW_STARTS]
                assert all(item["scheduled_frames"] > 0 and item["measured_face_frames"] > 0 for item in coverage["windows"])
                expected_statuses = ["localized", "localized", "no_distinct_range"] if case == "two_swaps" else ["no_distinct_range"] * 3
                assert [item["status"] for item in coverage["windows"]] == expected_statuses
                assert len(coverage["observations"]) == len(WINDOW_STARTS)
            else:
                assert coverage is None and not windows
            if case == "two_swaps":
                assert len(windows) == 2 and not legacy_packets
                assert [packet["face_return"] for packet in windows] == detector["details"]["face_short_windows"]
                proof, browser = verify_surfaces(client, aid, source, commitment, evidence, windows, output / case)
                row.update(proof, known_edit_frames=[list(pair) for pair in EDGES],
                           representative_frames=[list(indices) for indices in REPRESENTATIVES],
                           source_time_s=[[round(index / 30, 3) for index in pair] for pair in EDGES])
                write_json(output / "browser-input.json", {"analysis_id": aid, "packets": browser})
                (output / "browser-key.txt").write_text(key)
                (output / "browser-key.txt").chmod(0o600)
            else:
                assert not windows and not detector["details"].get("face_short_windows")
                assert len(legacy_packets) == (case == "legacy_four_seconds")
                if case == "genuine_cut":
                    proof, _ = verify_surfaces(client, aid, source, commitment, evidence, [], output / case)
                    row.update(proof)
                if case == "legacy_four_seconds":
                    assert evidence["version"] == LEGACY_VERSION
                    assert [legacy_packets[0]["face_return"][field] for field in ("start_frame_index", "end_frame_index")] == [62, 68]
            cases[case] = row
    assert decisions["two_swaps"] == decisions["feature_disabled"]
    write_json(output / "receipt.json", {"cases": cases, "unpaid_jobs": 0, "simulated_paid_jobs": len(cases),
                                         "held_receipt_sha256": sha_path(study.parent / "held.json"),
                                         "overall_decision_identical_flag_on_off": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Frozen eight-second HEVC held fixture directory")
    parser.add_argument("legacy", type=Path, help="Frozen four-second HEVC fixture directory")
    parser.add_argument("output", type=Path, help="New isolated output directory")
    args = parser.parse_args()
    run(args.study.resolve(), args.legacy.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
